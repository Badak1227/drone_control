import math
import random

import airsim
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dask.graph_manipulation import checkpoint
from torch.distributions import Normal
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from collections import deque


# 기본 설정 클래스
class Config:
    # 환경 설정
    max_episode_steps = 1000
    max_episodes = 1000
    goal_threshold = 1.0  # 목표 도달 판정 거리 (미터)
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
class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)  # → 32×20×20
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)  # → 64×9×9
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)  # → 64×7×7

        # Fully-connected layer
        self.fc = nn.Linear(64 * 7 * 7, 512)

    def forward(self, x):
        """
        x: Tensor of shape [B, 1, 84, 84]
        returns: features of shape [B, 512]
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.shape[0], -1)  # flatten
        features = F.relu(self.fc(x))
        return features


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
    def __init__(self, state_dim=9):
        super().__init__()
        # shared feature
        self.cnn = CNNFeatureExtractor()
        self.state_fc = nn.Linear(state_dim, 128)
        self.gru = nn.GRU(512 + 128, 256, batch_first=True)

        # actor & critics
        self.actor = PolicyNetwork(256, Config.action_dim, Config.max_action)
        self.q1 = QNetwork(256, Config.action_dim)
        self.q2 = QNetwork(256, Config.action_dim)

        # target critics
        self.target_q1 = QNetwork(256, Config.action_dim)
        self.target_q2 = QNetwork(256, Config.action_dim)
        self._copy_weights()

        # optimizers
        self.opt_critic = optim.Adam(
            list(self.cnn.parameters()) +
            list(self.state_fc.parameters()) +
            list(self.gru.parameters()) +
            list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=Config.lr_critic)
        self.opt_actor = optim.Adam(self.actor.parameters(), lr=Config.lr_actor)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=Config.device)
        self.opt_alpha = optim.Adam([self.log_alpha], lr=Config.lr_alpha)

    def _copy_weights(self):
        for p, tp in zip(self.q1.parameters(), self.target_q1.parameters()): tp.data.copy_(p.data)
        for p, tp in zip(self.q2.parameters(), self.target_q2.parameters()): tp.data.copy_(p.data)

    def forward(self, obs_seq, state_seq, h=None, evaluate=False):
        B, L, C, H, W = obs_seq.shape
        x_img = obs_seq.view(B * L, C, H, W)
        feats_img = self.cnn(x_img).view(B, L, -1)
        x_st = state_seq.view(B * L, -1)
        feats_st = F.relu(self.state_fc(x_st)).view(B, L, -1)
        feats = torch.cat([feats_img, feats_st], dim=-1)

        if h is None:
            h = torch.zeros(1, B, 256, device=Config.device)
        out, h_new = self.gru(feats, h)
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
        # build tensors
        obs_seq = torch.stack([torch.stack([torch.from_numpy(t.depth_image) for t in s], 0) for s in seqs])
        obs_seq = obs_seq.unsqueeze(2).to(Config.device).float()
        state_seq = torch.stack([torch.stack([torch.from_numpy(t.state) for t in s], 0) for s in seqs])
        state_seq = state_seq.to(Config.device).float()

        bi = Config.burn_in
        obs_b, obs_l = obs_seq[:, :bi], obs_seq[:, bi:]
        st_b, st_l = state_seq[:, :bi], state_seq[:, bi:]

        # burn-in GRU
        h0 = torch.zeros(1, Config.batch_size, 256, device=Config.device)
        _, h0 = self.gru(torch.cat([self.cnn(obs_b.view(-1, 1, 84, 84)).view(Config.batch_size, bi, -1),
                                    F.relu(self.state_fc(st_b.view(-1, 9))).view(Config.batch_size, bi, -1)], -1), h0)
        # learning pass
        out, _ = self.gru(
            torch.cat([self.cnn(obs_l.view(-1, 1, 84, 84)).view(Config.batch_size, Config.seq_length - bi, -1),
                       F.relu(self.state_fc(st_l.view(-1, 9))).view(Config.batch_size, Config.seq_length - bi, -1)],
                      -1), h0)
        last = out[:, -1]

        # extract actions, rewards
        actions = torch.stack([torch.tensor(s[-1].action) for s in seqs]).to(Config.device).float()
        r_ev = torch.tensor([s[-1].reward_evade for s in seqs], device=Config.device)
        r_ap = torch.tensor([s[-1].reward_approach for s in seqs], device=Config.device)
        done = torch.tensor([s[-1].done for s in seqs], device=Config.device).float()

        # critic update
        with torch.no_grad():
            a_next, logp_next = self.actor.sample(last)
            q1n = self.target_q1(last, a_next).squeeze(-1)
            q2n = self.target_q2(last, a_next).squeeze(-1)
            alpha = self.log_alpha.exp()
            target = r_ev + (1 - done) * Config.gamma * (torch.min(q1n, q2n) - alpha * logp_next.squeeze(-1))
        q1_cur = self.q1(last, actions).squeeze(-1)
        q2_cur = self.q2(last, actions).squeeze(-1)
        critic_loss = F.mse_loss(q1_cur, target) + F.mse_loss(q2_cur, target)
        self.opt_critic.zero_grad()
        critic_loss.backward()
        self.opt_critic.step()

        # actor update
        action_pi, logp_pi = self.actor.sample(last)
        q1_pi = self.q1(last, action_pi).squeeze(-1)
        q2_pi = self.q2(last, action_pi).squeeze(-1)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (alpha.detach() * logp_pi.squeeze(-1) - q_pi).mean()
        self.opt_actor.zero_grad()
        actor_loss.backward()
        self.opt_actor.step()

        # alpha update
        alpha_loss = -(self.log_alpha * (logp_pi.detach().squeeze(-1) + Config.target_entropy)).mean()
        self.opt_alpha.zero_grad()
        alpha_loss.backward()
        self.opt_alpha.step()

        # soft update targets
        for p, tp in zip(self.q1.parameters(), self.target_q1.parameters()): tp.data.copy_(
            tp.data * (1 - Config.tau) + p.data * Config.tau)
        for p, tp in zip(self.q2.parameters(), self.target_q2.parameters()): tp.data.copy_(
            tp.data * (1 - Config.tau) + p.data * Config.tau)

        return {'critic_loss': critic_loss.item(), 'actor_loss': actor_loss.item(),
                'alpha': self.log_alpha.exp().item()}

# --- Layer 1: Obstacle Avoidance Module ---
class ObstacleAvoidanceModule(nn.Module):
    def __init__(self, drone_state_dim=9):
        super().__init__()
        self.cnn = CNNFeatureExtractor()

        # Add FC layer to process drone state
        self.state_fc = nn.Linear(drone_state_dim, 128)

        # Adjust GRU input dimension to combine CNN features and state features
        self.gru = nn.GRU(512 + 128, 256, batch_first=True)

        self.actor = PolicyNetwork(256, Config.action_dim, Config.max_action)
        self.q1 = QNetwork(256, Config.action_dim)
        self.q2 = QNetwork(256, Config.action_dim)

    def process_features(self, obs_seq, drone_state_seq):
        """
        Process observations and drone states to extract features
        obs_seq: [B, L, 1, H, W]
        drone_state_seq: [B, L, drone_state_dim]
        returns: combined_feats [B, L, 512+128]
        """
        B, L, C, H, W = obs_seq.shape

        # Extract features through CNN
        x_img = obs_seq.reshape(B * L, C, H, W)
        img_feats = self.cnn(x_img).reshape(B, L, 512)

        # Process drone state
        x_state = drone_state_seq.reshape(B * L, -1)
        state_feats = F.relu(self.state_fc(x_state)).reshape(B, L, 128)

        # Combine image features and state features
        combined_feats = torch.cat([img_feats, state_feats], dim=2)
        return combined_feats

    def forward(self, obs_seq, drone_state_seq, h=None):
        """
        obs_seq: [B, L, C, H, W] tensor of observation sequence batch
        drone_state_seq: [B, L, drone_state_dim] tensor of drone state sequence
        h: GRU hidden state
        """
        B = obs_seq.shape[0]

        # Initialize hidden state if not provided
        if h is None:
            h = torch.zeros(1, B, 256, device=Config.device)

        # Process features
        combined_feats = self.process_features(obs_seq, drone_state_seq)

        # Process temporal information through GRU
        out_seq, h_new = self.gru(combined_feats, h)

        # Use output from last timestep
        last = out_seq[:, -1]

        # Sample action
        a, logp = self.actor.sample(last)

        # Calculate Q values
        q1 = self.q1(last, a)
        q2 = self.q2(last, a)

        return out_seq, h_new, a, logp, q1, q2


# --- Layer 2: Navigation Module ---
class NavigationModule(nn.Module):
    def __init__(self, drone_state_dim=9):
        super().__init__()
        self.cnn = CNNFeatureExtractor()

        # Add FC layer to process drone state
        self.state_fc = nn.Linear(drone_state_dim, 128)

        # Adjust GRU input dimension for all combined features
        self.gru = nn.GRU(512 + 128, 256, batch_first=True)

        self.actor = PolicyNetwork(256, Config.action_dim, Config.max_action)
        self.q1 = QNetwork(256, Config.action_dim)
        self.q2 = QNetwork(256, Config.action_dim)

    def process_features(self, obs_seq, drone_state_seq):
        """
        Process observations and drone states to extract features
        obs_seq: [B, L, 1, H, W]
        drone_state_seq: [B, L, drone_state_dim]
        returns: combined_feats [B, L, 512+128]
        """
        B, L, C, H, W = obs_seq.shape

        # Extract features through CNN
        x_img = obs_seq.reshape(B * L, C, H, W)
        img_feats = self.cnn(x_img).reshape(B, L, 512)

        # Process drone state
        x_state = drone_state_seq.reshape(B * L, -1)
        state_feats = F.relu(self.state_fc(x_state)).reshape(B, L, 128)

        # Combine image features and state features
        combined_feats = torch.cat([img_feats, state_feats], dim=2)
        return combined_feats

    def forward(self, obs_seq, drone_state_seq, h=None):
        """
        obs_seq: [B, L, C, H, W] tensor of observation sequence batch
        drone_state_seq: [B, L, drone_state_dim] tensor of drone state sequence
        h: GRU hidden state
        """
        B = obs_seq.shape[0]

        # Initialize hidden state if not provided
        if h is None:
            h = torch.zeros(1, B, 256, device=Config.device)

        # Process features
        combined_feats = self.process_features(obs_seq, drone_state_seq)

        # Process temporal information through GRU
        out_seq, h_new = self.gru(combined_feats, h)

        # Use output from last timestep
        last = out_seq[:, -1]

        # Sample action
        a, logp = self.actor.sample(last)

        # Calculate Q values
        q1 = self.q1(last, a)
        q2 = self.q2(last, a)

        return out_seq, h_new, a, logp, q1, q2


# --- Integrated Network to select between the two sub-actions ---
class IntegratedNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256 + 256, 256)  # Takes outputs from both modules
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)  # Outputs a value between 0 and 1

    def forward(self, x1, x2):
        """
        x1: output from obstacle avoidance module [B, 256]
        x2: output from navigation module [B, 256]
        returns: mixing coefficient [B, 1] in range [0, 1]
        """
        x = torch.cat([x1, x2], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # Output between 0 and 1
        return x


# --- Layered RSAC Agent ---
class LayeredRSACAgent(nn.Module):
    def __init__(self):
        super().__init__()
        # 모듈 초기화
        self.layer1 = ObstacleAvoidanceModule().to(Config.device)  # 장애물 회피
        self.layer2 = NavigationModule().to(Config.device)  # 내비게이션
        self.integrated = IntegratedNetwork().to(Config.device)  # 통합 결정

        # 타겟 네트워크 초기화
        self.target_q1_1 = QNetwork(256, Config.action_dim).to(Config.device)
        self.target_q1_2 = QNetwork(256, Config.action_dim).to(Config.device)
        self.target_q2_1 = QNetwork(256, Config.action_dim).to(Config.device)
        self.target_q2_2 = QNetwork(256, Config.action_dim).to(Config.device)

        # 가중치 복사
        self._copy_weights()

        # 옵티마이저 초기화
        # 1) Critic1 optimizer: q1, q2, (필요 시 cnn/state_fc/gru 포함)
        critic1_params = (
                list(self.layer1.cnn.parameters()) +
                list(self.layer1.state_fc.parameters()) +
                list(self.layer1.gru.parameters()) +
                list(self.layer1.q1.parameters()) +
                list(self.layer1.q2.parameters())
        )
        self.opt1 = optim.Adam(critic1_params, lr=Config.lr_critic)

        # Actor1 optimizer: 오직 actor 네트워크만
        self.opt_actor1 = optim.Adam(self.layer1.actor.parameters(), lr=Config.lr_actor)

        # 2) Critic2 optimizer
        critic2_params = (
                list(self.layer2.cnn.parameters()) +
                list(self.layer2.state_fc.parameters()) +
                list(self.layer2.gru.parameters()) +
                list(self.layer2.q1.parameters()) +
                list(self.layer2.q2.parameters())
        )
        self.opt2 = optim.Adam(critic2_params, lr=Config.lr_critic)

        # Actor2 optimizer
        self.opt_actor2 = optim.Adam(self.layer2.actor.parameters(), lr=Config.lr_actor)

        self.opt_integrated = optim.Adam(self.integrated.parameters(), lr=Config.lr_critic)

        # 엔트로피 관련
        self.log_alpha1 = torch.zeros(1, requires_grad=True, device=Config.device)
        self.log_alpha2 = torch.zeros(1, requires_grad=True, device=Config.device)
        self.alpha_opt1 = optim.Adam([self.log_alpha1], lr=Config.lr_alpha)
        self.alpha_opt2 = optim.Adam([self.log_alpha2], lr=Config.lr_alpha)

    def _copy_weights(self):
        """타겟 네트워크에 초기 가중치 복사"""
        for t, p in zip(self.target_q1_1.parameters(), self.layer1.q1.parameters()):
            t.data.copy_(p.data)
        for t, p in zip(self.target_q1_2.parameters(), self.layer1.q2.parameters()):
            t.data.copy_(p.data)
        for t, p in zip(self.target_q2_1.parameters(), self.layer2.q1.parameters()):
            t.data.copy_(p.data)
        for t, p in zip(self.target_q2_2.parameters(), self.layer2.q2.parameters()):
            t.data.copy_(p.data)

    def select_action(self, obs_seq, drone_state_seq, h1=None, h2=None, evaluate=False):
        """
        Select action based on current observations
        obs_seq: [B, L, C, H, W] tensor of observation sequence batch
        drone_state_seq: [B, L, drone_state_dim] tensor of drone state sequence
        h1, h2: Hidden states for the two modules
        evaluate: If True, use deterministic action, if False, use stochastic action
        """
        B = obs_seq.size(0)

        # Proper hidden state initialization for GRU
        if h1 is None:
            h1 = torch.zeros(1, B, 256, device=Config.device)
        if h2 is None:
            h2 = torch.zeros(1, B, 256, device=Config.device)

        with torch.no_grad():
            # Get outputs from both modules
            out1, h1_new, a1, _, _, _ = self.layer1(obs_seq, drone_state_seq, h1)
            out2, h2_new, a2, _, _, _ = self.layer2(obs_seq, drone_state_seq, h2)

            if evaluate:
                # Evaluation mode: deterministic actions
                a1 = self.layer1.actor.get_action(out1[:, -1])
                a2 = self.layer2.actor.get_action(out2[:, -1])

            # Integration selection (mix between the two actions)
            mix = self.integrated(out1[:, -1], out2[:, -1])

            # Choose action based on mixing coefficient
            a = mix * a1 + (1 - mix) * a2

        return a, h1_new, h2_new

    def soft_update(self, net, target_net):
        """타겟 네트워크 소프트 업데이트"""
        with torch.no_grad():
            for p, tp in zip(net.parameters(), target_net.parameters()):
                tp.copy_(tp * (1 - Config.tau) + p * Config.tau)

    def update(self, buffer):
        """
        One-step SAC update using full sequences with burn-in handled separately.
        """
        # 문제 진단을 위한 anomaly detection 활성화
        torch.autograd.set_detect_anomaly(True)

        try:
            # 1) Sample a batch of sequences
            seqs, indices, is_weights = buffer.sample(Config.batch_size, Config.seq_length)
            if not seqs:
                return None

            B = len(seqs)
            bi = Config.burn_in
            T = Config.seq_length - bi

            # print("\n===== DEBUG: STEP 1 - DATA SAMPLING =====")
            # print(f"Batch size: {B}, Burn-in: {bi}, Learn steps: {T}")

            # 2) Build tensors: observations [B, seq_len, 1, H, W], states [B, seq_len, state_dim]
            obs_seq = torch.stack([
                torch.stack([torch.from_numpy(t.depth_image) for t in s], dim=0)
                for s in seqs
            ], dim=0).unsqueeze(2).to(Config.device, dtype=torch.float32)  # [B, L, 1, H, W]

            state_seq = torch.stack([
                torch.stack([torch.from_numpy(t.state) for t in s], dim=0)
                for s in seqs
            ], dim=0).to(Config.device, dtype=torch.float32)  # [B, L, state_dim]

            # print("\n===== DEBUG: STEP 2 - INPUT TENSORS =====")
            # print(f"obs_seq shape: {obs_seq.shape}, dtype: {obs_seq.dtype}")
            # print(
            #     f"obs_seq stats: min={obs_seq.min().item():.4f}, max={obs_seq.max().item():.4f}, mean={obs_seq.mean().item():.4f}")
            #
            # print(f"state_seq shape: {state_seq.shape}, dtype: {state_seq.dtype}")
            # print(
            #     f"state_seq stats: min={state_seq.min().item():.4f}, max={state_seq.max().item():.4f}, mean={state_seq.mean().item():.4f}")

            # NaN 체크
            has_nan_obs = torch.isnan(obs_seq).any()
            has_nan_state = torch.isnan(state_seq).any()
            # print(f"NaN in observations: {has_nan_obs}")
            # print(f"NaN in states: {has_nan_state}")

            # 3) Extract action, reward, done for learning
            actions = torch.stack([
                torch.tensor(s[-1].action, dtype=torch.float32)
                for s in seqs
            ], dim=0).to(Config.device)  # [B, action_dim]

            r_ev = torch.tensor([s[-1].reward_evade for s in seqs],
                                dtype=torch.float32, device=Config.device)  # [B]
            r_ap = torch.tensor([s[-1].reward_approach for s in seqs],
                                dtype=torch.float32, device=Config.device)  # [B]
            done = torch.tensor([s[-1].done for s in seqs],
                                dtype=torch.float32, device=Config.device)  # [B]

            # print("\n===== DEBUG: STEP 3 - ACTION & REWARDS =====")
            # print(f"actions shape: {actions.shape}, dtype: {actions.dtype}")
            # print(
            #     f"actions stats: min={actions.min().item():.4f}, max={actions.max().item():.4f}, mean={actions.mean().item():.4f}")
            #
            # print(
            #     f"reward_evade stats: min={r_ev.min().item():.4f}, max={r_ev.max().item():.4f}, mean={r_ev.mean().item():.4f}")
            # print(
            #     f"reward_approach stats: min={r_ap.min().item():.4f}, max={r_ap.max().item():.4f}, mean={r_ap.mean().item():.4f}")
            # print(f"done stats: count(1)={done.sum().item()}/{B}")

            # 4) Split sequences into burn-in and learning parts
            obs_burn_in = obs_seq[:, :bi].clone()
            obs_learn = obs_seq[:, bi:].clone()
            state_burn_in = state_seq[:, :bi].clone()
            state_learn = state_seq[:, bi:].clone()

            # print("\n===== DEBUG: STEP 4 - BURN-IN & LEARNING SPLIT =====")
            # print(f"obs_burn_in shape: {obs_burn_in.shape}")
            # print(f"obs_learn shape: {obs_learn.shape}")
            #
            # # 5) Process features for burn-in
            # print("\n===== DEBUG: STEP 5 - FEATURE EXTRACTION =====")
            try:
                # Process features for burn-in
                features1_burn = self.layer1.process_features(obs_burn_in, state_burn_in)
                features2_burn = self.layer2.process_features(obs_burn_in, state_burn_in)

                # print(f"features1_burn shape: {features1_burn.shape}")
                # print(
                #     f"features1_burn stats: min={features1_burn.min().item():.4f}, max={features1_burn.max().item():.4f}, mean={features1_burn.mean().item():.4f}")
                # print(f"NaN in features1_burn: {torch.isnan(features1_burn).any()}")

                # Initialize hidden states
                h1 = torch.zeros(1, B, 256, device=Config.device)
                h2 = torch.zeros(1, B, 256, device=Config.device)

                #print(f"h1 initial shape: {h1.shape}")

                # Run GRU for burn-in sequences
                _, h1 = self.layer1.gru(features1_burn, h1)
                _, h2 = self.layer2.gru(features2_burn, h2)

                # print(
                #     f"h1 after burn-in stats: min={h1.min().item():.4f}, max={h1.max().item():.4f}, mean={h1.mean().item():.4f}")
                # print(f"NaN in h1: {torch.isnan(h1).any()}")

                # Process features for learning
                features1_learn = self.layer1.process_features(obs_learn, state_learn)
                features2_learn = self.layer2.process_features(obs_learn, state_learn)

                # print(f"features1_learn shape: {features1_learn.shape}")
                # print(f"NaN in features1_learn: {torch.isnan(features1_learn).any()}")

                # Run GRU for learning sequences
                out_seq1, _ = self.layer1.gru(features1_learn, h1)
                out_seq2, _ = self.layer2.gru(features2_learn, h2)

                # print(f"out_seq1 shape: {out_seq1.shape}")
                # print(
                #     f"out_seq1 stats: min={out_seq1.min().item():.4f}, max={out_seq1.max().item():.4f}, mean={out_seq1.mean().item():.4f}")
                # print(f"NaN in out_seq1: {torch.isnan(out_seq1).any()}")

                # Get last output features
                last_feat1 = out_seq1[:, -1].clone()
                last_feat2 = out_seq2[:, -1].clone()

                # print(f"last_feat1 shape: {last_feat1.shape}")
                # print(
                #     f"last_feat1 stats: min={last_feat1.min().item():.4f}, max={last_feat1.max().item():.4f}, mean={last_feat1.mean().item():.4f}")
                # print(f"NaN in last_feat1: {torch.isnan(last_feat1).any()}")

            except Exception as e:
                print(f"ERROR in feature extraction: {e}")
                import traceback
                traceback.print_exc()
                return None

            # 6) Critic targets and losses for layer1
            #print("\n===== DEBUG: STEP 6 - CRITIC TARGETS (Layer 1) =====")
            try:
                with torch.no_grad():
                    a1_next, logp1_next = self.layer1.actor.sample(last_feat1.detach())

                    # print(f"a1_next shape: {a1_next.shape}")
                    # print(
                    #     f"a1_next stats: min={a1_next.min().item():.4f}, max={a1_next.max().item():.4f}, mean={a1_next.mean().item():.4f}")
                    # print(
                    #     f"logp1_next stats: min={logp1_next.min().item():.4f}, max={logp1_next.max().item():.4f}, mean={logp1_next.mean().item():.4f}")
                    # print(f"NaN in a1_next: {torch.isnan(a1_next).any()}")
                    # print(f"NaN in logp1_next: {torch.isnan(logp1_next).any()}")

                    # 안전장치: logp가 NaN이면 대체
                    if torch.isnan(logp1_next).any():
                        logp1_next = torch.nan_to_num(logp1_next, nan=-20.0)
                        print("WARNING: NaN detected in logp1_next, replaced with -20.0")

                    q1_next1 = self.target_q1_1(last_feat1.detach(), a1_next)
                    q1_next2 = self.target_q1_2(last_feat1.detach(), a1_next)

                    #print(
                    #    f"q1_next1 stats: min={q1_next1.min().item():.4f}, max={q1_next1.max().item():.4f}, mean={q1_next1.mean().item():.4f}")
                    #print(
                    #    f"q1_next2 stats: min={q1_next2.min().item():.4f}, max={q1_next2.max().item():.4f}, mean={q1_next2.mean().item():.4f}")

                    q1n = torch.min(q1_next1, q1_next2).squeeze(-1)

                    # 안전장치: q1n 값 제한
                    q1n = torch.clamp(q1n, min=-100.0, max=100.0)

                    # 알파 안정화
                    alpha1 = torch.clamp(self.log_alpha1.exp(), min=1e-10, max=10.0)
                    #print(f"alpha1 value: {alpha1.item():.6f}")

                    # 타겟 계산
                    entropy_term = alpha1 * logp1_next.squeeze(-1)
                    target1 = r_ev + (1 - done) * Config.gamma * (q1n - entropy_term)

                    # 안전장치: 타겟 값 제한
                    target1 = torch.clamp(target1, min=-100.0, max=100.0)

                    # print(
                    #     f"target1 stats: min={target1.min().item():.4f}, max={target1.max().item():.4f}, mean={target1.mean().item():.4f}")
                    # print(f"NaN in target1: {torch.isnan(target1).any()}")

                # 현재 Q 값 계산
                q11 = self.layer1.q1(last_feat1, actions).squeeze(-1)
                q12 = self.layer1.q2(last_feat1, actions).squeeze(-1)

                #print(
                #    f"q11 stats: min={q11.min().item():.4f}, max={q11.max().item():.4f}, mean={q11.mean().item():.4f}")
                #print(
                #    f"q12 stats: min={q12.min().item():.4f}, max={q12.max().item():.4f}, mean={q12.mean().item():.4f}")
                #print(f"NaN in q11: {torch.isnan(q11).any()}")
                #print(f"NaN in q12: {torch.isnan(q12).any()}")

                # 손실 계산 전 NaN 체크 및 대체
                if torch.isnan(q11).any() or torch.isnan(target1).any():
                    print("WARNING: NaN detected before critic loss calculation!")
                    q11 = torch.nan_to_num(q11, nan=0.0)
                    q12 = torch.nan_to_num(q12, nan=0.0)
                    target1 = torch.nan_to_num(target1, nan=0.0)

                critic1_loss = F.mse_loss(q11, target1) + F.mse_loss(q12, target1)
                #print(f"critic1_loss: {critic1_loss.item():.6f}")
                #print(f"NaN in critic1_loss: {torch.isnan(critic1_loss).any()}")

            except Exception as e:
                print(f"ERROR in critic target calculation (Layer 1): {e}")
                import traceback
                traceback.print_exc()
                return None

            # 7) Actor and alpha losses for layer1
            #print("\n===== DEBUG: STEP 7 - ACTOR LOSS (Layer 1) =====")
            try:
                # Actor loss 계산을 위한 분리된 복제본 사용
                last_feat1_actor = last_feat1.clone().detach().requires_grad_(True)
                a1_pi, logp1 = self.layer1.actor.sample(last_feat1_actor)

                #print(f"a1_pi shape: {a1_pi.shape}")
                #print(
                #    f"a1_pi stats: min={a1_pi.min().item():.4f}, max={a1_pi.max().item():.4f}, mean={a1_pi.mean().item():.4f}")
                #print(
                #    f"logp1 stats: min={logp1.min().item():.4f}, max={logp1.max().item():.4f}, mean={logp1.mean().item():.4f}")

                # NaN 체크 및 대체
                if torch.isnan(logp1).any():
                    print("WARNING: NaN detected in logp1!")
                    logp1 = torch.nan_to_num(logp1, nan=-20.0)

                q1_pi1 = self.layer1.q1(last_feat1_actor, a1_pi)
                q1_pi2 = self.layer1.q2(last_feat1_actor, a1_pi)
                #print(
                #    f"q1_pi1 stats: min={q1_pi1.min().item():.4f}, max={q1_pi1.max().item():.4f}, mean={q1_pi1.mean().item():.4f}")

                q1_pi = torch.min(q1_pi1, q1_pi2).squeeze(-1)

                # 알파값 안정화
                alpha1_actor = torch.clamp(self.log_alpha1.exp().detach(), min=1e-10, max=10.0)

                actor1_loss = (alpha1_actor * logp1.squeeze(-1) - q1_pi).mean()
                #print(f"actor1_loss: {actor1_loss.item():.6f}")
                #print(f"NaN in actor1_loss: {torch.isnan(actor1_loss).any()}")

                # Alpha loss
                alpha1_loss = -(self.log_alpha1 * (logp1.squeeze(-1).detach() + Config.target_entropy)).mean()
                #print(f"alpha1_loss: {alpha1_loss.item():.6f}")

            except Exception as e:
                print(f"ERROR in actor loss calculation (Layer 1): {e}")
                import traceback
                traceback.print_exc()
                return None

            # 8) Layer 2 repeat steps 6-7 (간략화)
            #print("\n===== DEBUG: STEP 8 - LAYER 2 LOSSES =====")
            try:
                # Critic targets for layer2
                with torch.no_grad():
                    a2_next, logp2_next = self.layer2.actor.sample(last_feat2.detach())

                    # NaN 체크 및 대체
                    if torch.isnan(logp2_next).any():
                        logp2_next = torch.nan_to_num(logp2_next, nan=-20.0)
                        print("WARNING: NaN detected in logp2_next, replaced with -20.0")

                    q2n = torch.min(
                        self.target_q2_1(last_feat2.detach(), a2_next),
                        self.target_q2_2(last_feat2.detach(), a2_next)
                    ).squeeze(-1)

                    # 안전장치: q2n 값 제한
                    q2n = torch.clamp(q2n, min=-100.0, max=100.0)

                    # 알파 안정화
                    alpha2 = torch.clamp(self.log_alpha2.exp(), min=1e-10, max=10.0)

                    target2 = r_ap + (1 - done) * Config.gamma * (q2n - alpha2 * logp2_next.squeeze(-1))
                    target2 = torch.clamp(target2, min=-100.0, max=100.0)

                q21 = self.layer2.q1(last_feat2, actions).squeeze(-1)
                q22 = self.layer2.q2(last_feat2, actions).squeeze(-1)

                if torch.isnan(q21).any() or torch.isnan(target2).any():
                    q21 = torch.nan_to_num(q21, nan=0.0)
                    q22 = torch.nan_to_num(q22, nan=0.0)
                    target2 = torch.nan_to_num(target2, nan=0.0)

                critic2_loss = F.mse_loss(q21, target2) + F.mse_loss(q22, target2)
                #print(f"critic2_loss: {critic2_loss.item():.6f}")

                # Actor loss for layer2
                last_feat2_actor = last_feat2.clone().detach().requires_grad_(True)
                a2_pi, logp2 = self.layer2.actor.sample(last_feat2_actor)

                if torch.isnan(logp2).any():
                    print("WARNING: NaN detected in logp2!")
                    logp2 = torch.nan_to_num(logp2, nan=-20.0)

                q2_pi = torch.min(
                    self.layer2.q1(last_feat2_actor, a2_pi),
                    self.layer2.q2(last_feat2_actor, a2_pi)
                ).squeeze(-1)

                alpha2_actor = torch.clamp(self.log_alpha2.exp().detach(), min=1e-10, max=10.0)

                actor2_loss = (alpha2_actor * logp2.squeeze(-1) - q2_pi).mean()
                #print(f"actor2_loss: {actor2_loss.item():.6f}")

                alpha2_loss = -(self.log_alpha2 * (logp2.squeeze(-1).detach() + Config.target_entropy)).mean()

            except Exception as e:
                print(f"ERROR in Layer 2 loss calculation: {e}")
                import traceback
                traceback.print_exc()
                return None

            # 9) Integrated network loss
            #print("\n===== DEBUG: STEP 9 - INTEGRATED NETWORK =====")
            try:
                # 복제본 사용하여 인플레이스 연산 방지
                last_feat1_int = last_feat1.clone().detach()
                last_feat2_int = last_feat2.clone().detach()

                mix = self.integrated(last_feat1_int, last_feat2_int)
                #print(
                #    f"mix stats: min={mix.min().item():.4f}, max={mix.max().item():.4f}, mean={mix.mean().item():.4f}")

                # Simple heuristic: If evade reward is negative, prioritize avoidance action
                obstacle_danger = (r_ev < 0).float().unsqueeze(-1).clone()
                #print(f"obstacle_danger: count(1)={obstacle_danger.sum().item()}/{B}")

                integrated_loss = F.binary_cross_entropy(mix, obstacle_danger)
                #print(f"integrated_loss: {integrated_loss.item():.6f}")

            except Exception as e:
                print(f"ERROR in integrated network calculation: {e}")
                import traceback
                traceback.print_exc()
                return None

            # 10) Backprop: critics, actors+alphas, integrated
            #print("\n===== DEBUG: STEP 10 - BACKPROP & OPTIMIZATION =====")
            try:
                # 1. 크리틱 업데이트 - 분리된 그래프 사용
                self.opt1.zero_grad()
                critic1_loss.backward()  # retain_graph 제거
                critic1_params = list(self.layer1.q1.parameters()) + list(self.layer1.q2.parameters())
                torch.nn.utils.clip_grad_norm_(critic1_params, max_norm=1.0)
                self.opt1.step()
                #print("Critic1 update completed")

                # 2. 액터 업데이트 - 완전히 새로운 계산 그래프 생성
                self.opt_actor1.zero_grad()

                # 모든 텐서를 새로 생성하여 이전 그래프와 완전히 분리
                with torch.no_grad():
                    last_feat1_new = last_feat1.clone()

                # 새 텐서에 requires_grad 설정
                last_feat1_actor = last_feat1_new.requires_grad_(True)

                # 액터 네트워크로 액션과 로그 확률 계산
                a1_pi, logp1 = self.layer1.actor.sample(last_feat1_actor)

                # NaN 체크 및 처리
                if torch.isnan(logp1).any():
                    logp1 = torch.nan_to_num(logp1, nan=-20.0)
                    print("WARNING: NaN detected in logp1, replaced")

                # Q 값 계산
                q1_pi1 = self.layer1.q1(last_feat1_actor, a1_pi)
                q1_pi2 = self.layer1.q2(last_feat1_actor, a1_pi)
                q1_pi = torch.min(q1_pi1, q1_pi2).squeeze(-1)

                # 알파 값 계산 (detach 사용)
                alpha1_actor = self.log_alpha1.detach().exp()

                # 액터 손실 계산
                actor1_loss = (alpha1_actor * logp1.squeeze(-1) - q1_pi).mean()

                # 역전파
                actor1_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.layer1.actor.parameters(), max_norm=1.0)
                self.opt_actor1.step()
                #print("Actor1 update completed")

                # 3. 알파 업데이트 - 별도의 계산 그래프 사용
                self.alpha_opt1.zero_grad()

                # logp1은 이미 detach됨 (이전 그래프에서 생성됨)
                alpha1_loss = -(self.log_alpha1 * (logp1.squeeze(-1).detach() + Config.target_entropy)).mean()
                alpha1_loss.backward()
                self.alpha_opt1.step()
                #print("Alpha1 update completed")

                # 4. Layer2 크리틱 업데이트
                self.opt2.zero_grad()
                critic2_loss.backward()
                critic2_params = list(self.layer2.q1.parameters()) + list(self.layer2.q2.parameters())
                torch.nn.utils.clip_grad_norm_(critic2_params, max_norm=1.0)
                self.opt2.step()
                #print("Critic2 update completed")

                # 5. Layer2 액터 업데이트 - 새로운 계산 그래프 생성
                self.opt_actor2.zero_grad()

                # 새로운 텐서 생성
                with torch.no_grad():
                    last_feat2_new = last_feat2.clone()

                # requires_grad 설정
                last_feat2_actor = last_feat2_new.requires_grad_(True)

                # 액션과 로그 확률 계산
                a2_pi, logp2 = self.layer2.actor.sample(last_feat2_actor)

                # NaN 체크 및 처리
                if torch.isnan(logp2).any():
                    logp2 = torch.nan_to_num(logp2, nan=-20.0)
                    print("WARNING: NaN detected in logp2, replaced")

                # Q 값 계산
                q2_pi1 = self.layer2.q1(last_feat2_actor, a2_pi)
                q2_pi2 = self.layer2.q2(last_feat2_actor, a2_pi)
                q2_pi = torch.min(q2_pi1, q2_pi2).squeeze(-1)

                # 알파 값 계산
                alpha2_actor = self.log_alpha2.detach().exp()

                # 액터 손실 계산
                actor2_loss = (alpha2_actor * logp2.squeeze(-1) - q2_pi).mean()

                # 역전파
                actor2_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.layer2.actor.parameters(), max_norm=1.0)
                self.opt_actor2.step()
                #print("Actor2 update completed")

                # 6. Layer2 알파 업데이트
                self.alpha_opt2.zero_grad()
                alpha2_loss = -(self.log_alpha2 * (logp2.squeeze(-1).detach() + Config.target_entropy)).mean()
                alpha2_loss.backward()
                self.alpha_opt2.step()
                #print("Alpha2 update completed")

                # 7. 통합 네트워크 업데이트 - 별도의 계산 그래프 사용
                self.opt_integrated.zero_grad()

                # 분리된 계산 그래프 사용
                with torch.no_grad():
                    last_feat1_int = last_feat1.clone()
                    last_feat2_int = last_feat2.clone()

                mix = self.integrated(last_feat1_int, last_feat2_int)
                obstacle_danger = (r_ev < 0).float().unsqueeze(-1)

                integrated_loss = F.binary_cross_entropy(mix, obstacle_danger)
                integrated_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.integrated.parameters(), max_norm=1.0)
                self.opt_integrated.step()
                #print("Integrated network update completed")

                # 타겟 네트워크 소프트 업데이트
                self.soft_update(self.layer1.q1, self.target_q1_1)
                self.soft_update(self.layer1.q2, self.target_q1_2)
                self.soft_update(self.layer2.q1, self.target_q2_1)
                self.soft_update(self.layer2.q2, self.target_q2_2)
                #print("Target networks soft-updated")

            except Exception as e:
                print(f"ERROR in backprop & optimization: {e}")
                import traceback
                traceback.print_exc()
                return None

            # 11) PER update
            #print("\n===== DEBUG: STEP 11 - PER UPDATE =====")
            try:
                # TD 오류를 사용한 우선순위 업데이트
                td_errors = [
                    [float(abs(q11[i].item() - target1[i].item()))]
                    for i in range(B)
                ]

                buffer.update_priorities(indices, td_errors)
                #print(f"Priority update completed for {len(indices)} indices")

            except Exception as e:
                print(f"ERROR in PER update: {e}")
                import traceback
                traceback.print_exc()
                return None

            # 최종 결과 반환
            return {
                'loss_evade': critic1_loss.item(),
                'policy_evade': actor1_loss.item(),
                'alpha_evade': self.log_alpha1.exp().item(),
                'loss_approach': critic2_loss.item(),
                'policy_approach': actor2_loss.item(),
                'alpha_approach': self.log_alpha2.exp().item(),
                'loss_integrated': integrated_loss.item()
            }

        except Exception as e:
            print(f"CRITICAL ERROR in update function: {e}")
            import traceback
            traceback.print_exc()
            return None

    def save_checkpoint(self, episode):
        """모델 체크포인트 저장"""
        if not os.path.exists(Config.model_dir):
            os.makedirs(Config.model_dir)

        checkpoint = {
            'layer1': self.layer1.state_dict(),
            'layer2': self.layer2.state_dict(),
            'integrated': self.integrated.state_dict(),
            'target_q1_1': self.target_q1_1.state_dict(),
            'target_q1_2': self.target_q1_2.state_dict(),
            'target_q2_1': self.target_q2_1.state_dict(),
            'target_q2_2': self.target_q2_2.state_dict(),
            'log_alpha1': self.log_alpha1,
            'log_alpha2': self.log_alpha2,
            'episode' : episode
        }

        filename = f"rsac_ep{episode}.pt"
        filepath = os.path.join(Config.model_dir, filename)

        # 최신 모델 저장
        latest_filepath = os.path.join(Config.model_dir, f"rsac_latest.pt")
        torch.save(checkpoint, latest_filepath)

        print(f"모델 체크포인트 저장 완료: {filepath}")

    def load_state_dict(self, checkpoint):


        if isinstance(checkpoint, str):
            state_dict = torch.load(checkpoint, map_location=Config.device, weights_only=False)

        """저장된 상태 딕셔너리로부터 모델 로드"""
        self.layer1.load_state_dict(state_dict['layer1'])
        self.layer2.load_state_dict(state_dict['layer2'])
        self.integrated.load_state_dict(state_dict['integrated'])
        self.target_q1_1.load_state_dict(state_dict['target_q1_1'])
        self.target_q1_2.load_state_dict(state_dict['target_q1_2'])
        self.target_q2_1.load_state_dict(state_dict['target_q2_1'])
        self.target_q2_2.load_state_dict(state_dict['target_q2_2'])
        self.log_alpha1 = state_dict['log_alpha1']
        self.log_alpha2 = state_dict['log_alpha2']

        return state_dict['episode']


# 라이다 데이터를 깊이 이미지로 변환
def lidar_to_depth_image(lidar_points, height=Config.depth_image_height,
                         width=Config.depth_image_width, max_distance=Config.max_lidar_distance):
    """
    라이다 포인트 클라우드를 구형 좌표계 기반 depth 이미지로 변환
    로컬 좌표계 기준 (드론 중심)
    """
    # 빈 depth 이미지 초기화 (최대 거리로)
    depth_image = np.ones((height, width)) * max_distance

    if not lidar_points:
        return depth_image / max_distance  # 정규화된 이미지 반환

    for point in lidar_points:
        # 각 점은 (x, y, z) 튜플입니다
        x, y, z = point

        # 구면 좌표로 변환 (r, azimuth, elevation)
        r = math.sqrt(x * x + y * y + z * z)

        # 범위 내의 포인트만 처리
        if r <= max_distance:
            # 방위각: xy 평면에서의 각도 (-π ~ π)
            azimuth = math.atan2(y, x)

            # 고도각: z와 xy 평면 사이의 각도 (-π/2 ~ π/2)
            elevation = math.asin(z / max(r, 1e-5))

            # 구면 좌표를 이미지 픽셀로 변환
            col = int((azimuth + math.pi) / (2 * math.pi) * width)
            row = int((elevation + math.pi / 2) / math.pi * height)

            # 픽셀 범위 확인
            if 0 <= row < height and 0 <= col < width:
                # 해당 픽셀 위치에 가장 가까운 거리 저장
                depth_image[row, col] = min(depth_image[row, col], r)

    # 깊이 이미지 정규화 (0~1 범위로)
    normalized_depth = depth_image / max_distance

    return normalized_depth


# 드론 환경 클래스
class DroneEnv:
    def __init__(self):
        # AirSim 클라이언트 초기화
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)

        # 목표 위치 설정 (임의로 설정, 실제 환경에 맞게 수정 필요)
        self.goal_position = np.array([50.0, 0.0, -10.0])

        self.goal_distance = 0
        self.prev_goal_distance = 0

        # 에피소드 스텝 카운터
        self.steps = 0

    def reset(self):
        # 드론 초기화
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()

        # 초기 위치로 이동 (홈 위치, 약간의 랜덤성 추가)
        initial_x, initial_y, initial_z = [0, 0, 0]

        self.client.moveToPositionAsync(initial_x, initial_y, initial_z, 5).join()

        # 새로운 목표 위치 (랜덤)
        while True:
            self.goal_position = np.array([
                random.uniform(-19.0, 19.0),
                random.uniform(-19.0, 19.0),
                random.uniform(-19.0, 19.0)
            ])

            self.goal_distance = np.linalg.norm(self.goal_position)
            self.prev_goal_distance = self.goal_distance
            if self.goal_distance >= 15:
                print(f"New goal position: {self.goal_position}")
                break

        # 스텝 카운터 초기화
        self.steps = 0

        # 초기 상태 가져오기
        depth_image, drone_state, position = self._get_state()

        return depth_image, drone_state

    def step(self, action):
        # 액션 적용 (vx, vy, vz 속도)
        vx, vy, vz = action

        # AirSim에서는 NED 좌표계 사용
        self.client.moveByVelocityBodyFrameAsync(
            float(vx),
            float(vy),
            float(vz),  # AirSim에서 z는 아래 방향이 양수
            1.0  # 1초 동안 속도 유지
        )

        # 상태 및 라이다 데이터 가져오기
        depth_image, state, position = self._get_state()

        # 보상과 종료 여부 계산
        reward_evade, reward_approach, done, info = self._compute_reward(depth_image, state, position)

        # 스텝 카운터 증가
        self.steps += 1

        # 최대 스텝 수 초과 시 종료
        if self.steps >= Config.max_episode_steps:
            done = True
            info['timeout'] = True

        return depth_image, state, reward_evade, reward_approach, done, info

    def _get_lidar_data(self, euler):
        lidar_data = self.client.getLidarData("LidarSensor1", "HelloDrone").point_cloud

        if len(lidar_data) >= 3:
            # 롤, 피치 각도 가져오기
            roll, pitch = euler[0], euler[1]

            # 회전 행렬 계산
            Rx = np.array([
                [1, 0, 0],
                [0, np.cos(-roll), -np.sin(-roll)],
                [0, np.sin(-roll), np.cos(-roll)]
            ])

            Ry = np.array([
                [np.cos(-pitch), 0, np.sin(-pitch)],
                [0, 1, 0],
                [-np.sin(-pitch), 0, np.cos(-pitch)]
            ])

            R_horizontal = np.dot(Ry, Rx)

            # 모든 포인트를 numpy 배열로 변환 (벡터화)
            points_count = len(lidar_data) // 3
            points = np.array(lidar_data).reshape(points_count, 3)

            # 모든 포인트를 한 번에 변환 (빠른 행렬곱)
            corrected_points = np.dot(points, R_horizontal.T)

            # 변환된 포인트를 튜플 리스트로 변환
            points_list = [tuple(point) for point in corrected_points]

            return lidar_to_depth_image(points_list)

        return lidar_to_depth_image([])

    def _get_state(self):
        # 드론 상태와 방향 가져오기
        kinematics = self.client.simGetGroundTruthKinematics()
        position = np.array([kinematics.position.x_val, kinematics.position.y_val, kinematics.position.z_val])
        orientation = kinematics.orientation

        # 쿼터니언을 오일러 각도로 변환
        euler = self._quaternion_to_euler(
            np.array([orientation.w_val, orientation.x_val, orientation.y_val, orientation.z_val])
        )

        # 요(yaw)만 고려한 회전 행렬
        yaw = euler[2]
        yaw_rotation_matrix = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])

        # 라이다 데이터에서 깊이 이미지 가져오기
        depth_image = self._get_lidar_data(euler)


        # 속도 가져오기 (NED 좌표계)
        velocity_ned = np.array([
            kinematics.linear_velocity.x_val,
            kinematics.linear_velocity.y_val,
            kinematics.linear_velocity.z_val
        ])

        # NED 좌표계에서 목표까지의 상대 벡터
        relative_goal_ned = self.goal_position - position

        # NED 속도와 상대 목표 위치를 바디 프레임으로 변환
        velocity_body = np.dot(yaw_rotation_matrix.T, velocity_ned)
        relative_goal_body = np.dot(yaw_rotation_matrix.T, relative_goal_ned)

        # 바디 프레임 기준 상태 벡터: 속도(3), 목표 상대 위치(3), 드론 pose(3)
        drone_state = np.concatenate([velocity_body, relative_goal_body, euler])

        return depth_image, drone_state, position

    def _quaternion_to_rotation_matrix(self, w, x, y, z):
        # 쿼터니언을 회전 행렬로 변환
        rotation_matrix = np.array([
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]
        ])
        return rotation_matrix

    def _quaternion_to_euler(self, q):
        """쿼터니언에서 오일러 각도로 변환"""
        w, x, y, z = q

        # 롤 (x축 회전)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # 피치 (y축 회전)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)  # 90도로 제한
        else:
            pitch = math.asin(sinp)

        # 요 (z축 회전)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return np.array([roll, pitch, yaw])

    def _compute_reward(self, depth_image, drone_state, position):
        # 현재 위치와 목표 위치
        current_goal_distance = np.linalg.norm(drone_state[3:6])

        # 충돌 확인
        collision_info = self.client.simGetCollisionInfo()
        has_collided = collision_info.has_collided

        # 보상 초기화
        reward_evade = 0  # 장애물 회피 보상
        reward_approach = 0  # 내비게이션 보상

        info = {
            "distance_to_goal": current_goal_distance,
            "prev_distance": self.prev_goal_distance
        }

        done = False

        # 목표 도달 확인
        if current_goal_distance < Config.goal_threshold:
            print(f"Goal reached at position: {position}")
            reward_approach += 100.0  # 목표 도달 큰 보상
            done = True
            info["status"] = "goal_reached"

        # 충돌 확인
        elif has_collided:
            print(f"Collision at position: {position}")
            reward_evade -= 100.0  # 충돌 큰 페널티
            done = True
            info["status"] = "collision"

        # 일반 이동 중 보상
        else:
            # --- 장애물 회피(Evade) 보상 ---
            min_depth = np.min(depth_image) * Config.max_lidar_distance

            # 안전 거리 임계값
            safety_threshold = 3.0  # 미터
            danger_threshold = 1.5  # 미터

            # 장애물이 가까울수록 페널티, 멀수록 보상
            if min_depth < safety_threshold:
                # 선형 보간: 안전 거리에서는 작은 페널티, 위험 거리에서는 큰 페널티
                obstacle_penalty = -3.0 * (1.0 - (min_depth - danger_threshold) /
                                           (safety_threshold - danger_threshold))
                obstacle_penalty = max(obstacle_penalty, -3.0)  # 최대 페널티 제한
                reward_evade += obstacle_penalty
            else:
                # 장애물이 충분히 멀 경우 작은 보상
                reward_evade += 0.1

            info["obstacle_distance"] = min_depth

            # --- 내비게이션(Approach) 보상 ---
            # 목표를 향해 가까워지면 보상, 멀어지면 페널티
            distance_reward = np.clip(self.prev_goal_distance - current_goal_distance, -1.0, 1.0)
            reward_approach += distance_reward * 3.0

            # 시간 패널티 (빠른 목표 도달 장려)
            reward_approach -= 0.1

            # 에너지 효율성 (급격한 움직임 패널티)
            velocity = drone_state[:3]
            speed = np.linalg.norm(velocity)

            # 너무 빠른 속도에 페널티
            if speed > Config.max_drone_speed:
                reward_approach -= 0.2 * (speed - Config.max_drone_speed)

            info["status"] = "moving"

        # 현재 목표 거리를 이전 거리로 업데이트
        self.prev_goal_distance = current_goal_distance

        return reward_evade, reward_approach, done, info


# Transition 클래스 정의 - 분리된 보상 구조를 위해 수정
class Transition:
    def __init__(self, depth_image, state, action, reward_evade, reward_approach, next_depth_image, next_state, done):
        self.depth_image = depth_image  # 깊이 이미지
        self.state = state  # 드론 상태 벡터
        self.action = action  # 수행한 액션
        self.reward_evade = reward_evade  # 장애물 회피 보상
        self.reward_approach = reward_approach  # 목표 접근 보상
        self.next_depth_image = next_depth_image  # 다음 깊이 이미지
        self.next_state = next_state  # 다음 드론 상태
        self.done = done  # 종료 여부


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
def train_layered_rsac():
    # 환경, 에이전트, 리플레이 버퍼 초기화
    env = DroneEnv()
    agent = LayeredRSACAgent()

    # 리플레이 버퍼 사용
    buffer = PrioritizedSequenceReplayBuffer(capacity=10000, alpha=0.6, burn_in=Config.burn_in)

    # 학습 진행 추적기
    tracker = TrainingTracker()

    # 모델, 로그 디렉토리 생성
    os.makedirs(Config.model_dir, exist_ok=True)
    os.makedirs(Config.log_dir, exist_ok=True)

    latest_model_path = os.path.join(Config.model_dir, f"rsac_latest.pt")
    start_episode = 0
    if os.path.exists(latest_model_path):
        episode = agent.load_state_dict(latest_model_path)
        if episode is not None:
            start_episode = episode + 1
            print(f"체크포인트 로드 완료: 에피소드 {start_episode}부터 계속")

    # 에피소드 반복
    for episode in range(start_episode, Config.max_episodes):
        # 환경 초기화
        depth_image, drone_state = env.reset()

        # 에피소드 상태 저장 변수
        episode_reward_evade = 0
        episode_reward_approach = 0
        episode_reward_total = 0
        episode_steps = 0
        episode_transitions = []  # 현재 에피소드의 전이 저장

        # 히든 스테이트 초기화
        h1 = None
        h2 = None

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

                # 액션 선택 (드론 상태 정보 포함)
                action_tensor, h1, h2 = agent.select_action(depth_seq, state_seq, h1, h2)
                action = action_tensor.cpu().numpy()[0]

            # 환경에서 한 스텝 진행
            next_depth_image, next_state, reward_evade, reward_approach, done, info = env.step(action)

            # 총 보상 계산 (논문의 정의에 따라)
            reward_total = reward_evade + reward_approach

            # 전이 저장
            transition = Transition(
                depth_image, drone_state, action,
                reward_evade, reward_approach,
                next_depth_image, next_state, done
            )
            episode_transitions.append(transition)

            # 보상 누적
            episode_reward_evade += reward_evade
            episode_reward_approach += reward_approach
            episode_reward_total += reward_total
            episode_steps += 1

            # 다음 상태로 업데이트
            depth_image = next_depth_image
            drone_state = next_state

            # 에이전트 업데이트 (정기적으로)
            if len(buffer) > Config.batch_size:
                if episode_steps % 10 == 0:  # 10스텝마다 업데이트
                    update_info = agent.update(buffer)

        # 에피소드가 끝나면 전이를 리플레이 버퍼에 추가
        if episode_transitions:
            buffer.push(episode_transitions)

        # 에피소드 통계 기록
        episode_duration = time.time() - episode_start_time
        status = info.get("status", "unknown")

        tracker.add_episode_stats(episode_reward_total, episode_steps, status)
        avg_reward, success_rate = tracker.get_recent_stats()

        # 로그 출력
        if episode % Config.log_interval == 0:
            print(f"Episode {episode}: " +
                  f"Time={episode_duration:.2f}s, Steps ={episode_steps}, " +
                  f"Avg Reward={avg_reward:.2f}, Status={info['status']}")
            print()

        # 모델 저장
        if episode % Config.save_interval == 0:
            agent.save_checkpoint(episode)

            # 학습 진행 그래프 저장
            graph_path = os.path.join(Config.log_dir, f"training_progress_ep{episode}.png")
            tracker.plot_training_progress(save_path=graph_path)

    # 최종 모델 저장
    agent.save_checkpoint(Config.max_episodes)

    # 최종 학습 진행 그래프 저장
    final_graph_path = os.path.join(Config.log_dir, "training_progress_final.png")
    tracker.plot_training_progress(save_path=final_graph_path)

    print("Training completed!")
    return agent, tracker


# 메인 함수
if __name__ == "__main__":
    # 학습 시작
    trained_agent, training_stats = train_layered_rsac()

    # 최종 결과 출력
    final_avg_reward, final_success_rate = training_stats.get_recent_stats()
    print(f"Final Results: Average Reward = {final_avg_reward:.2f}, Success Rate = {final_success_rate:.2f}")
