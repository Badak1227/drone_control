import csv
import math
import random

import airsim
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dask.graph_manipulation import checkpoint
from matplotlib import cm
from torch.distributions import Normal
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from collections import deque

from torchvision import models


# 기본 설정 클래스
class Config:
    # 환경 설정
    max_episode_steps = 1000
    max_episodes = 3000
    goal_threshold = 3.0  # 목표 도달 판정 거리 (미터)
    max_drone_speed = 5.0  # 최대 허용 속도 (m/s)

    # 라이다 및 깊이 이미지 설정
    depth_image_height = 84
    depth_image_width = 84
    max_lidar_distance = 50.0  # 최대 라이다 감지 거리 (미터)

    # 신경망 및 학습 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr_actor = 3e-4
    lr_critic = 3e-4
    lr_alpha = 3e-4
    gamma = 0.99
    tau = 0.005
    alpha = 0.2

    # 학습 관련 설정
    prior_sigma = 0.45  # 논문에서 최적으로 제시된 값
    prior_decay_rate = 0.00005  # 경험 감쇠율

    batch_size = 16
    burn_in = 16
    seq_length = 32
    action_dim = 3
    max_action = 5.0
    target_entropy = -3  # -action_dim으로 수정 (엔트로피 타겟)

    # 저장 및 평가 설정
    save_interval = 20  # 에피소드마다 모델 저장
    eval_interval = 20  # 에피소드마다 평가
    log_interval = 1  # 에피소드마다 로그 출력

    # 경로 설정
    model_dir = "models"
    log_dir = "logs"


# --- Prioritized Experience Replay for Sequences ---
class PrioritizedSequenceReplayBuffer:
    def __init__(self, capacity, alpha=0.6, burn_in=16):
        """
        capacity: Maximum number of episodes
        alpha: Priority scaling factor
        burn_in: Warm-up steps
        """
        self.capacity = capacity
        self.alpha = alpha
        self.burn_in = burn_in

        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def push(self, episode_transitions):
        """Add one episode (list of transitions) to the buffer"""
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(episode_transitions)
            self.priorities[len(self.buffer) - 1] = max_prio
        else:
            self.buffer[self.pos] = episode_transitions
            self.priorities[self.pos] = max_prio
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, seq_length):
        """
        Sample batch_size sequences of (burn_in + learning) length with PER
        Returns:
          sequences:   list of lists of transitions, len=batch_size
          indices:     list of indices in buffer (batch_size)
          is_weights:  importance sampling weights (batch_size,)
        """
        if len(self.buffer) == 0:
            return [], [], []

        # PER probability distribution
        prios = self.priorities[:len(self.buffer)] ** self.alpha
        total = prios.sum()

        if not np.isfinite(total) or total <= 0:
            probs = np.ones_like(prios, dtype=np.float32) / len(prios)
        else:
            probs = prios / total
        # Sample episode indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        is_weights = (len(self.buffer) * probs[indices]) ** -1
        is_weights /= is_weights.max()  # Normalize to [0,1]

        sequences = []
        for idx in indices:
            ep = self.buffer[idx]
            L = len(ep)
            # If sequence length is insufficient, use only burn-in part and pad the rest
            if L >= seq_length:
                start = random.randint(0, L - seq_length)
                seq = ep[start: start + seq_length]
            else:
                # Pad with last transition
                pad_n = seq_length - L
                seq = ep + [ep[-1]] * pad_n

            sequences.append(seq)

        return sequences, indices.tolist(), is_weights

    def update_priorities(self, indices, td_errors, eps=1e-6):
        """Update priorities after sampling using max TD-error of each sequence"""
        for idx, errors in zip(indices, td_errors):
            arr = np.array(errors, dtype=np.float32).flatten()
            max_err = float(np.max(np.abs(arr)))
            self.priorities[idx] = max_err + eps

    def __len__(self):
        return len(self.buffer)


# --- CNN Feature Extractor ---
class SimpleCNNFeatureExtractor(nn.Module):
    def __init__(self, output_dim=256):
        super().__init__()
        # 더 간단한 CNN 구조
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # 특징 맵 크기 계산 (84x84 입력 기준)
        # 첫 번째 층 이후: (84-8)/4+1 = 20
        # 두 번째 층 이후: (20-4)/2+1 = 9
        # 세 번째 층 이후: (9-3)/1+1 = 7
        # 따라서 최종 특징 맵 크기는 7x7x64 = 3136

        self.fc = nn.Linear(7 * 7 * 64, output_dim)
        self.bn = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # 평탄화
        x = self.fc(x)
        x = self.bn(x)
        return F.relu(x)


# --- Actor Network (Policy) ---
class PolicyNetwork(nn.Module):
    def __init__(self, in_dim, action_dim, max_action):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)
        self.max_action = max_action
        self.log_std_min, self.log_std_max = -20, 2

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        return mean, std

    def sample(self, x):
        mean, std = self.forward(x)
        normal = Normal(mean, std)
        # 재매개화 트릭을 사용한 샘플링
        x_t = normal.rsample()

        # Tanh 스케일링
        action = torch.tanh(x_t)

        # 실제 로그 확률 계산
        log_prob = normal.log_prob(x_t)

        # Tanh 변환에 의한 로그 확률 보정
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        # 액션 스케일링
        action = action * self.max_action

        return action, log_prob

    def get_action(self, x):
        mean, _ = self.forward(x)
        return torch.tanh(mean) * self.max_action


# --- Critic Network (Twin Q) ---
class QNetwork(nn.Module):
    def __init__(self, in_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.q = nn.Linear(256, 1)

    def forward(self, x, a):
        xu = torch.cat([x, a], dim=-1)
        x1 = F.relu(self.fc1(xu))
        x1 = F.relu(self.fc2(x1))
        return self.q(x1)

# --- Agent ---
class RSACAgent(nn.Module):
    def __init__(self, state_dim=6):  # 자세 정보 제외한 상태 차원
        super().__init__()
        # 특징 추출
        self.cnn = SimpleCNNFeatureExtractor(output_dim=256)
        self.cnn_bn = nn.BatchNorm1d(256)

        # 상태 처리
        self.state_fc = nn.Linear(state_dim, 128)
        self.state_bn = nn.BatchNorm1d(128)

        # GRU 사용
        self.rnn = nn.GRU(256 + 128, 256, batch_first=True)

        # 액터 & 크리틱
        self.actor = PolicyNetwork(256, Config.action_dim, Config.max_action)
        self.q1 = QNetwork(256, Config.action_dim)
        self.q2 = QNetwork(256, Config.action_dim)

        # 타겟 크리틱
        self.target_q1 = QNetwork(256, Config.action_dim)
        self.target_q2 = QNetwork(256, Config.action_dim)
        self._copy_weights()

        # 최적화기
        self.opt_critic = optim.Adam(
            list(self.cnn.parameters()) +
            list(self.state_fc.parameters()) +
            list(self.rnn.parameters()) +
            list(self.q1.parameters()) +
            list(self.q2.parameters()),
            lr=Config.lr_critic
        )
        self.opt_actor = optim.Adam(self.actor.parameters(), lr=Config.lr_actor)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=Config.device)
        self.opt_alpha = optim.Adam([self.log_alpha], lr=Config.lr_alpha)

    def _copy_weights(self):
        for p, tp in zip(self.q1.parameters(), self.target_q1.parameters()):
            tp.data.copy_(p.data)
        for p, tp in zip(self.q2.parameters(), self.target_q2.parameters()):
            tp.data.copy_(p.data)

    def forward(self, obs_seq, state_seq, h=None, evaluate=False):
        B, L, C, H, W = obs_seq.shape

        # CNN 특징 추출 및 정규화
        x_img = obs_seq.reshape(B * L, C, H, W)
        feats_img = self.cnn(x_img)
        feats_img = self.cnn_bn(feats_img)
        feats_img = feats_img.reshape(B, L, -1)

        # 상태 특징 추출 및 정규화
        x_st = state_seq.reshape(B * L, -1)
        feats_st = self.state_fc(x_st)
        feats_st = self.state_bn(feats_st)
        feats_st = F.relu(feats_st).reshape(B, L, -1)

        # 특징 결합
        feats = torch.cat([feats_img, feats_st], dim=-1)

        # GRU 처리
        if h is None:
            h = torch.zeros(1, B, 256, device=Config.device)
        out, h_new = self.rnn(feats, h)
        last = out[:, -1]

        if evaluate:
            action = self.actor.get_action(last)
            return action, h_new
        else:
            action, logp = self.actor.sample(last)
            return action, logp, last, h_new


    def select_action(self, obs_seq, state_seq, h=None, evaluate=False):
        with torch.no_grad():
            if evaluate:
                action, h_new = self.forward(obs_seq, state_seq, h, True)
                return action, h_new
            else:
                action, _, _, h_new = self.forward(obs_seq, state_seq, h, False)
                return action, h_new

    def update(self, buffer):
        seqs, idxs, isw = buffer.sample(Config.batch_size, Config.seq_length)
        if not seqs: return

        # 텐서 변환 시 명시적으로 float32 타입 사용
        obs_seq = torch.stack(
            [torch.stack([torch.from_numpy(np.array(t.depth_image, dtype=np.float32)) for t in s], 0) for s in seqs])
        obs_seq = obs_seq.unsqueeze(2).to(Config.device)
        state_seq = torch.stack(
            [torch.stack([torch.from_numpy(np.array(t.state, dtype=np.float32)) for t in s], 0) for s in seqs])
        state_seq = state_seq.to(Config.device)

        bi = Config.burn_in
        obs_b, obs_l = obs_seq[:, :bi], obs_seq[:, bi:]
        st_b, st_l = state_seq[:, :bi], state_seq[:, bi:]

        # burn-in GRU - 배치 정규화 적용
        h0 = torch.zeros(1, Config.batch_size, 256, device=Config.device)

        # CNN 특징 처리 (배치 정규화 포함)
        cnn_feats_b = self.cnn(obs_b.reshape(-1, 1, 84, 84))
        cnn_feats_b = self.cnn_bn(cnn_feats_b)
        cnn_feats_b = cnn_feats_b.reshape(Config.batch_size, bi, -1)

        # 상태 특징 처리 - 6차원으로 수정
        state_feats_b = self.state_fc(st_b.reshape(-1, 6))
        state_feats_b = self.state_bn(state_feats_b)
        state_feats_b = F.relu(state_feats_b)
        state_feats_b = state_feats_b.reshape(Config.batch_size, bi, -1)

        # 특징 결합 및 GRU 처리
        _, h0 = self.rnn(torch.cat([cnn_feats_b, state_feats_b], -1), h0)

        # learning pass - 배치 정규화 적용
        # CNN 특징 처리
        cnn_feats_l = self.cnn(obs_l.reshape(-1, 1, 84, 84))
        cnn_feats_l = self.cnn_bn(cnn_feats_l)
        cnn_feats_l = cnn_feats_l.reshape(Config.batch_size, Config.seq_length - bi, -1)

        # 상태 특징 처리 - 6차원으로 수정
        state_feats_l = self.state_fc(st_l.reshape(-1, 6))
        state_feats_l = self.state_bn(state_feats_l)
        state_feats_l = F.relu(state_feats_l)
        state_feats_l = state_feats_l.reshape(Config.batch_size, Config.seq_length - bi, -1)

        # 특징 결합 및 GRU 처리
        out, _ = self.rnn(torch.cat([cnn_feats_l, state_feats_l], -1), h0)

        # 마지막 출력 저장 (이후에 여러 번 복사하여 사용)
        last = out[:, -1].clone()

        # 액션, 보상, 완료 상태 준비 - 단일 보상 값 사용
        actions = torch.stack([torch.tensor(s[-1].action, dtype=torch.float32) for s in seqs]).to(Config.device)
        r = torch.tensor([float(s[-1].reward) for s in seqs], dtype=torch.float32, device=Config.device)
        done = torch.tensor([float(s[-1].done) for s in seqs], dtype=torch.float32, device=Config.device)

        # 1. 크리틱 업데이트
        with torch.no_grad():
            a_next, logp_next = self.actor.sample(last)
            q1n = self.target_q1(last, a_next).squeeze(-1)
            q2n = self.target_q2(last, a_next).squeeze(-1)
            alpha = torch.clamp(self.log_alpha.exp(), min=1e-10, max=10.0)  # 안정성을 위한 클램핑
            target = r + (1 - done) * Config.gamma * (torch.min(q1n, q2n) - alpha * logp_next.squeeze(-1))

        # 크리틱 손실 계산 및 업데이트
        q1_cur = self.q1(last, actions).squeeze(-1)
        q2_cur = self.q2(last, actions).squeeze(-1)
        critic_loss = F.mse_loss(q1_cur, target) + F.mse_loss(q2_cur, target)

        self.opt_critic.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.q1.parameters()) + list(self.q2.parameters()) +
            list(self.cnn.parameters()) + list(self.state_fc.parameters()) +
            list(self.rnn.parameters()),
            max_norm=1.0
        )
        self.opt_critic.step()

        # 2. 액터 업데이트 - 새로운 계산 그래프 사용
        last_actor = last.detach().clone()  # 완전히 분리된 복사본

        action_pi, logp_pi = self.actor.sample(last_actor)
        q1_pi = self.q1(last_actor, action_pi).squeeze(-1)
        q2_pi = self.q2(last_actor, action_pi).squeeze(-1)
        q_pi = torch.min(q1_pi, q2_pi)

        # 알파 값은 계산 그래프에서 분리
        alpha_detached = self.log_alpha.exp().detach()

        actor_loss = (alpha_detached * logp_pi.squeeze(-1) - q_pi).mean()

        self.opt_actor.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.opt_actor.step()

        # 3. 알파 업데이트 - 또 다른 독립적인 계산 그래프
        logp_pi_detached = logp_pi.detach().squeeze(-1)  # 그래프에서 분리

        alpha_loss = -(self.log_alpha * (logp_pi_detached + Config.target_entropy)).mean()

        self.opt_alpha.zero_grad()
        alpha_loss.backward()
        self.opt_alpha.step()

        # 4. 타겟 네트워크 소프트 업데이트
        for p, tp in zip(self.q1.parameters(), self.target_q1.parameters()):
            tp.data.copy_(tp.data * (1 - Config.tau) + p.data * Config.tau)
        for p, tp in zip(self.q2.parameters(), self.target_q2.parameters()):
            tp.data.copy_(tp.data * (1 - Config.tau) + p.data * Config.tau)

        # TD 오류를 계산하여 PER 업데이트
        with torch.no_grad():
            td_errors = torch.abs(q1_cur - target).cpu().numpy().tolist()

        buffer.update_priorities(idxs, td_errors)

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha': self.log_alpha.exp().item()
        }



# Transition 클래스 정의 - 분리된 보상 구조를 위해 수정
class Transition:
    def __init__(self, depth_image, state, action, reward, next_depth_image, next_state, done):
        self.depth_image = depth_image
        self.state = state
        self.action = action
        self.reward = reward  # 단일 보상 값
        self.next_depth_image = next_depth_image
        self.next_state = next_state
        self.done = done


# 학습 진행 상황을 추적하는 클래스
class TrainingTracker:
    def __init__(self, window_size=100):
        self.rewards_history = []
        self.episode_lengths = []
        self.success_history = []
        self.collision_history = []
        self.timeout_history = []
        self.recent_rewards = deque(maxlen=window_size)
        self.recent_successes = deque(maxlen=window_size)

    def add_episode_stats(self, episode_reward, episode_length, status):
        self.rewards_history.append(episode_reward)
        self.episode_lengths.append(episode_length)

        success = status == "goal_reached"
        collision = status == "collision"
        timeout = status == "timeout"

        self.success_history.append(1 if success else 0)
        self.collision_history.append(1 if collision else 0)
        self.timeout_history.append(1 if timeout else 0)

        self.recent_rewards.append(episode_reward)
        self.recent_successes.append(1 if success else 0)

    def get_recent_stats(self):
        avg_reward = sum(self.recent_rewards) / max(len(self.recent_rewards), 1)
        success_rate = sum(self.recent_successes) / max(len(self.recent_successes), 1)
        return avg_reward, success_rate

    def plot_training_progress(self, save_path=None):
        """학습 진행 상황을 그래프로 시각화"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 보상 그래프
        axes[0, 0].plot(self.rewards_history)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')

        # 에피소드 길이 그래프
        axes[0, 1].plot(self.episode_lengths)
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')

        # 성공률 그래프
        window = 20  # 이동 평균 윈도우 크기
        success_rate = [sum(self.success_history[max(0, i - window):i + 1]) /
                        min(i + 1, window) for i in range(len(self.success_history))]

        axes[1, 0].plot(success_rate)
        axes[1, 0].set_title('Success Rate (Moving Average)')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Success Rate')
        axes[1, 0].set_ylim(0, 1)

        # 결과 분포 그래프
        episodes = list(range(len(self.success_history)))
        axes[1, 1].stackplot(episodes,
                             self.success_history,
                             self.collision_history,
                             self.timeout_history,
                             labels=['Success', 'Collision', 'Timeout'],
                             colors=['green', 'red', 'orange'])

        axes[1, 1].set_title('Episode Outcomes')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].legend(loc='upper left')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)

        plt.close()


# 학습 함수
def train_rsac():
    # 환경, 에이전트, 리플레이 버퍼 초기화
    env = DroneEnv()
    agent = RSACAgent().to(Config.device)

    # 리플레이 버퍼 사용
    buffer = PrioritizedSequenceReplayBuffer(capacity=10000, alpha=0.6, burn_in=Config.burn_in)

    # 학습 진행 추적기
    tracker = TrainingTracker()

    # 모델, 로그 디렉토리 생성
    os.makedirs(Config.model_dir, exist_ok=True)
    os.makedirs(Config.log_dir, exist_ok=True)
    log_csv_path = os.path.join(Config.log_dir, "training_log.csv")

    # CSV 헤더 작성
    with open(log_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([
            'Episode', 'Steps', 'Duration', 'Reward', 'Avg_Reward',
            'Status', 'Success_Rate', 'Critic_Loss', 'Actor_Loss', 'Alpha'
        ])


    # 체크포인트 로드 (있을 경우)
    latest_model_path = os.path.join(Config.model_dir, f"rsac_latest.pt")
    start_episode = 0
    if os.path.exists(latest_model_path):
        try:
            checkpoint = torch.load(latest_model_path, map_location=Config.device)
            agent.load_state_dict(checkpoint['model_state'])
            start_episode = checkpoint.get('episode', 0) + 1
            print(f"체크포인트 로드 완료: 에피소드 {start_episode}부터 계속")
        except Exception as e:
            print(f"체크포인트 로드 실패: {e}")

    # 에피소드 반복
    for episode in range(start_episode, Config.max_episodes):
        try:
            # 환경 초기화
            depth_image, drone_state = env.reset()

            # 에피소드 상태 저장 변수
            episode_reward = 0
            episode_steps = 0
            episode_transitions = []  # 현재 에피소드의 전이 저장
            episode_critic_losses = []
            episode_actor_losses = []
            episode_alphas = []


            # 히든 스테이트 초기화
            h = None

            done = False
            episode_start_time = time.time()

            # 에피소드 실행
            while not done and episode_steps < Config.max_episode_steps:
                # 에이전트로부터 액션 획득
                with torch.no_grad():
                    # 깊이 이미지 변환 및 상태 전처리
                    depth_tensor = torch.FloatTensor(depth_image).unsqueeze(0).unsqueeze(0).to(Config.device)
                    state_tensor = torch.FloatTensor(drone_state).unsqueeze(0).unsqueeze(0).to(Config.device)

                    # 시퀀스 형태로 변환
                    if episode_steps == 0:
                        # 처음 단계인 경우 이미지와 상태를 반복
                        depth_seq = depth_tensor.repeat(1, Config.seq_length, 1, 1, 1)
                        state_seq = state_tensor.repeat(1, Config.seq_length, 1)
                    else:
                        # 아닌 경우 이전 시퀀스에 현재 이미지와 상태 추가
                        depth_seq = torch.cat([depth_seq[:, 1:], depth_tensor.unsqueeze(1)], dim=1)
                        state_seq = torch.cat([state_seq[:, 1:], state_tensor], dim=1)

                    # 액션 선택
                    action_tensor, h = agent.select_action(depth_seq, state_seq, h)
                    action = action_tensor.cpu().numpy()[0]

                # 환경에서 한 스텝 진행 - 단일 보상 값 반환
                next_depth_image, next_state, reward, done, info = env.step(action)

                # 전이 저장 - 단일 보상 값 사용
                transition = Transition(
                    depth_image, drone_state, action,
                    float(reward),  # 단일 보상 값
                    next_depth_image, next_state, done
                )
                episode_transitions.append(transition)

                # 보상 누적
                episode_reward += reward
                episode_steps += 1

                # 다음 상태로 업데이트
                depth_image = next_depth_image
                drone_state = next_state

                # 에이전트 업데이트 (정기적으로)
                if len(buffer) > Config.batch_size:
                    if episode_steps % 10 == 0:  # 10스텝마다 업데이트
                        update_info = agent.update(buffer)
                        if update_info:
                            episode_critic_losses.append(update_info['critic_loss'])
                            episode_actor_losses.append(update_info['actor_loss'])
                            episode_alphas.append(update_info['alpha'])

            # 에피소드가 끝나면 전이를 리플레이 버퍼에 추가
            if episode_transitions:
                buffer.push(episode_transitions)

            # 에피소드 통계 기록
            avg_critic_loss = np.mean(episode_critic_losses) if episode_critic_losses else 0
            avg_actor_loss = np.mean(episode_actor_losses) if episode_actor_losses else 0
            avg_alpha = np.mean(episode_alphas) if episode_alphas else 0
            episode_duration = time.time() - episode_start_time
            status = info.get("status", "unknown")

            tracker.add_episode_stats(episode_reward, episode_steps, status)
            avg_reward, success_rate = tracker.get_recent_stats()

            # 로그 출력
            if episode % Config.log_interval == 0:
                print(f"Episode {episode}: " +
                      f"Time={episode_duration:.2f}s, Steps={episode_steps}, " +
                      f"Reward={episode_reward:.2f}, Avg Reward={avg_reward:.2f}, Status={status}")

                # 로그 파일에 기록 (CSV)
                with open(log_csv_path, 'a', newline='', encoding='utf-8') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow([
                        episode, episode_steps, f"{episode_duration:.2f}",
                        f"{episode_reward:.2f}", f"{avg_reward:.2f}",
                        status, f"{success_rate:.4f}",
                        f"{avg_critic_loss:.4f}", f"{avg_actor_loss:.4f}", f"{avg_alpha:.4f}"
                    ])
            print()

            # 모델 저장
            if episode % Config.save_interval == 0:
                # 모델 저장
                checkpoint = {
                    'model_state': agent.state_dict(),
                    'episode': episode
                }
                latest_filepath = os.path.join(Config.model_dir, f"rsac_latest.pt")
                torch.save(checkpoint, latest_filepath)

                # 특정 에피소드 체크포인트 저장
                ep_filepath = os.path.join(Config.model_dir, f"rsac_ep{episode}.pt")
                torch.save(checkpoint, ep_filepath)

                print(f"모델 체크포인트 저장 완료: {ep_filepath}")

                # 학습 진행 그래프 저장
                graph_path = os.path.join(Config.log_dir, f"training_progress_ep{episode}.png")
                tracker.plot_training_progress(save_path=graph_path)

        except Exception as e:
            print(f"에피소드 {episode} 실행 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            # 오류 발생해도 계속 진행

    # 최종 모델 저장
    checkpoint = {
        'model_state': agent.state_dict(),
        'episode': Config.max_episodes - 1
    }
    final_filepath = os.path.join(Config.model_dir, f"rsac_final.pt")
    torch.save(checkpoint, final_filepath)

    # 최종 학습 진행 그래프 저장
    final_graph_path = os.path.join(Config.log_dir, "training_progress_final.png")
    tracker.plot_training_progress(save_path=final_graph_path)

    print("Training completed!")
    return agent, tracker


# 메인 함수
if __name__ == "__main__":
    # 학습 시작
    trained_agent, training_stats = train_rsac()

    # 최종 결과 출력
    final_avg_reward, final_success_rate = training_stats.get_recent_stats()
    print(f"Final Results: Average Reward = {final_avg_reward:.2f}, Success Rate = {final_success_rate:.2f}")
