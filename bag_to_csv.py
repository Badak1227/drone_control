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
    max_episodes = 1000
    goal_threshold = 1.0  # 목표 도달 판정 거리 (미터)
    max_drone_speed = 5.0  # 최대 허용 속도 (m/s)

    # 라이다 및 깊이 이미지 설정
    depth_image_height = 42
    depth_image_width = 42
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

    batch_size = 32
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

# --- Prioritized Experience Replay for Sequences ---
class PrioritizedTransitionReplayBuffer:
    """
    Prioritized Experience Replay for single-step Transitions.
    Stores transitions (depth_image, state, action, reward, next_depth_image, next_state, done)
    """
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def push(self, depth, state, action, reward, next_depth, next_state, done):
        depth = torch.from_numpy(depth).float()
        state = torch.from_numpy(state).float()
        next_depth = torch.from_numpy(next_depth).float()
        next_state = torch.from_numpy(next_state).float()
        transition = Transition(depth, state, action, reward, next_depth, next_state, done)

        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.4):
        """
        Sample a batch of transitions with Prioritized Replay.
        Returns:
          batch: list of transitions
          indices: list of sampled indices
          weights: numpy array of shape [batch_size]
        """
        if len(self.buffer) == 0:
            return [], [], []
        prios = self.priorities[:len(self.buffer)] ** self.alpha
        probs = prios / prios.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        return samples, indices, weights

    def update_priorities(self, indices, td_errors, eps: float = 1e-6):
        """
        Update priorities of sampled transitions.
        td_errors: list or array of floats (absolute TD errors)
        """
        for idx, err in zip(indices, td_errors):
            self.priorities[idx] = abs(err) + eps

    def __len__(self):
        return len(self.buffer)


# --- CNN Feature Extractor ---
class CNNFeatureExtractor(nn.Module):
    def __init__(self, in_channels=4, output_dim=128):  # 출력 차원 축소
        super().__init__()
        # 더 효율적인 커널 크기와 채널 수
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=5, stride=2, padding=2)  # 유지 크기
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

        # 특징 풀링으로 차원 축소
        self.pool = nn.AdaptiveAvgPool2d((5, 5))

        # 최종 FC 레이어
        self.fc = nn.Linear(32 * 5 * 5, output_dim)
        self.bn = nn.LayerNorm(output_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
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
    def __init__(self, state_dim=6):
        super().__init__()
        # 더 작은 출력 차원으로 CNN 특징 추출기 초기화
        self.cnn = CNNFeatureExtractor(output_dim=128)
        self.cnn_bn = nn.LayerNorm(128)

        # 상태 처리 - 더 작은 차원
        self.state_fc = nn.Linear(state_dim, 64)
        self.state_bn = nn.LayerNorm(64)

        # 결합 특징 차원 축소
        self.shared_fc1 = nn.Linear(128 + 64, 128)
        self.shared_fc2 = nn.Linear(128, 128)

        # 액터 & 크리틱 (더 작은 입력 차원)
        self.actor = PolicyNetwork(128, Config.action_dim, Config.max_action)
        self.q1 = QNetwork(128, Config.action_dim)
        self.q2 = QNetwork(128, Config.action_dim)

        # 타겟 크리틱
        self.target_q1 = QNetwork(128, Config.action_dim)
        self.target_q2 = QNetwork(128, Config.action_dim)
        self._copy_weights()

        # 최적화기
        self.opt_critic = optim.Adam(
            list(self.cnn.parameters()) +
            list(self.state_fc.parameters()) +
            list(self.shared_fc1.parameters()) +
            list(self.shared_fc2.parameters()) +
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

    def forward(self, depth, state, evaluate=False):
        # CNN 특징
        feat_img = self.cnn(depth)  # (B,256)
        feat_img = self.cnn_bn(feat_img)
        # State 특징
        feat_st = F.relu(self.state_bn(self.state_fc(state)))  # (B,128)
        # 결합 → 공유 MLP
        x = torch.cat([feat_img, feat_st], dim=-1)  # (B,384)
        x = F.relu(self.shared_fc1(x))
        shared_feat = F.relu(self.shared_fc2(x))  # (B,256)

        if evaluate:
            action = self.actor.get_action(shared_feat)
            return action
        else:
            action, logp = self.actor.sample(shared_feat)
            return action, logp


    def select_action(self, depth, state, evaluate=False):
        with torch.no_grad():
            if evaluate:
                action = self.forward(depth, state, True)
                return action
            else:
                action, _= self.forward(depth, state, False)
                return action

    def update(self, buffer, beta=0.4):
        """
        SAC 알고리즘의 최적화된 업데이트 함수
        - 메모리 효율성 개선
        - 텐서 차원 일관성 유지
        - 계산 그래프 분리로 역전파 오류 방지
        """
        transitions, indices, is_weights = buffer.sample(Config.batch_size, beta)
        if not transitions:
            return {}

        device = Config.device

        # 1. 효율적인 배치 텐서 생성
        depths = torch.stack([t.depth_image for t in transitions]).to(device)
        states = torch.stack([t.state for t in transitions]).to(device)
        next_depths = torch.stack([t.next_depth_image for t in transitions]).to(device)
        next_states = torch.stack([t.next_state for t in transitions]).to(device)

        # NumPy 배열 중간 변환으로 효율성 향상
        actions_np = np.array([t.action for t in transitions])
        rewards_np = np.array([t.reward for t in transitions])
        dones_np = np.array([t.done for t in transitions])

        actions = torch.tensor(actions_np, dtype=torch.float32, device=device)
        rewards = torch.tensor(rewards_np, dtype=torch.float32, device=device).unsqueeze(1)
        dones = torch.tensor(dones_np, dtype=torch.float32, device=device).unsqueeze(1)

        # 2. 특징 추출 - 현재 상태
        with torch.no_grad():  # 그래디언트 계산 방지로 메모리 사용 최적화
            # 비주얼 특징 (CNN)
            img_feat = self.cnn(depths)
            img_feat = self.cnn_bn(img_feat)

            # 상태 특징 (FC)
            st_feat = F.relu(self.state_bn(self.state_fc(states)))

            # 특징 결합
            feat = torch.cat([img_feat, st_feat], dim=-1)
            shared_feat = F.relu(self.shared_fc1(feat))
            shared_feat = F.relu(self.shared_fc2(shared_feat))

            # 다음 상태 특징 추출
            img_feat_n = self.cnn(next_depths)
            img_feat_n = self.cnn_bn(img_feat_n)
            st_feat_n = F.relu(self.state_bn(self.state_fc(next_states)))
            feat_n = torch.cat([img_feat_n, st_feat_n], dim=-1)
            shared_n = F.relu(self.shared_fc1(feat_n))
            shared_n = F.relu(self.shared_fc2(shared_n))

            # 다음 상태 액션 및 가치 계산
            a_next, logp_n = self.actor.sample(shared_n)
            q1n = self.target_q1(shared_n, a_next)
            q2n = self.target_q2(shared_n, a_next)

            # 차원 일치 보장
            q1n = q1n.squeeze(-1)
            q2n = q2n.squeeze(-1)
            logp_n = logp_n.squeeze(-1)

            # 엔트로피 보정된 Q 타겟 계산
            alpha = self.log_alpha.exp().detach()
            target_q = rewards + (1 - dones) * Config.gamma * (
                        torch.min(q1n, q2n).unsqueeze(-1) - alpha * logp_n.unsqueeze(-1))
            target_q = target_q.squeeze(-1)  # 일관된 차원 보장

        # 3. Critic 업데이트 (Twin Q)
        # 새 계산 그래프 시작
        img_feat = self.cnn(depths)
        img_feat = self.cnn_bn(img_feat)
        st_feat = F.relu(self.state_bn(self.state_fc(states)))
        feat = torch.cat([img_feat, st_feat], dim=-1)
        shared_feat = F.relu(self.shared_fc1(feat))
        shared_feat = F.relu(self.shared_fc2(shared_feat))

        q1_cur = self.q1(shared_feat, actions).squeeze(-1)
        q2_cur = self.q2(shared_feat, actions).squeeze(-1)

        # MSE 손실 계산 (차원 일치 확인)
        critic_loss = F.mse_loss(q1_cur, target_q) + F.mse_loss(q2_cur, target_q)
        self.opt_critic.zero_grad()
        critic_loss.backward()
        self.opt_critic.step()

        # 4. Actor 업데이트 (분리된 계산 그래프 사용)
        with torch.no_grad():
            shared_feat_detached = shared_feat.detach()  # 그래디언트 흐름 차단

        action_pi, logp_pi = self.actor.sample(shared_feat_detached)
        q1_pi = self.q1(shared_feat_detached, action_pi).squeeze(-1)
        q2_pi = self.q2(shared_feat_detached, action_pi).squeeze(-1)
        q_pi = torch.min(q1_pi, q2_pi)
        logp_pi = logp_pi.squeeze(-1)

        # Actor 손실: 기대 Q값 최대화 + 엔트로피 보너스
        alpha_det = self.log_alpha.exp().detach()
        actor_loss = (alpha_det * logp_pi - q_pi).mean()
        self.opt_actor.zero_grad()
        actor_loss.backward()
        self.opt_actor.step()

        # 5. 온도 파라미터(알파) 업데이트
        with torch.no_grad():
            logp_pi_detached = logp_pi.detach()

        alpha_loss = -(self.log_alpha * (logp_pi_detached + Config.target_entropy)).mean()
        self.opt_alpha.zero_grad()
        alpha_loss.backward()
        self.opt_alpha.step()

        # 6. 타겟 네트워크 소프트 업데이트
        for p, tp in zip(self.q1.parameters(), self.target_q1.parameters()):
            tp.data.copy_(tp.data * (1 - Config.tau) + p.data * Config.tau)
        for p, tp in zip(self.q2.parameters(), self.target_q2.parameters()):
            tp.data.copy_(tp.data * (1 - Config.tau) + p.data * Config.tau)

        # 7. 우선순위 업데이트
        td_errors = (q1_cur - target_q).abs().detach().cpu().numpy().tolist()
        buffer.update_priorities(indices, td_errors)

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha': self.log_alpha.exp().item()
        }



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

        # 목표 위치 설정
        self.goal_position = np.array([50.0, 0.0, -10.0])

        # 거리 기록
        self.goal_distance = 0.0
        self.prev_goal_distance = 0.0

        # 에피소드 스텝 카운터
        self.steps = 0

        # 최근 4프레임 깊이 이미지 저장
        self.depth_queue = deque(maxlen=4)

    def reset(self):
        # 드론 초기화
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()

        # 홈 위치 이동
        self.client.moveToPositionAsync(0, 0, 0, 5).join()

        # 랜덤 목표 설정
        while True:
            self.goal_position = np.array([
                random.uniform(-19.0, 19.0),
                random.uniform(-19.0, 19.0),
                random.uniform(-19.0, 19.0)
            ])
            self.goal_distance = np.linalg.norm(self.goal_position)
            self.prev_goal_distance = self.goal_distance
            if self.goal_distance >= 15.0:
                break

        print(f"Goal: {self.goal_position}")

        # 스텝 초기화
        self.steps = 0

        # 초기 깊이 이미지 가져와 큐에 채우기
        depth, state, _, _, _ = self._get_state()
        self.depth_queue.clear()
        for _ in range(4):
            self.depth_queue.append(depth)

        # 스택된 상태 반환
        depth_stack = np.stack(self.depth_queue, axis=0)  # shape [4,H,W]
        return depth_stack, state

    def step(self, action):
        # 액션 적용 (vx, vy, vz)
        # 다음 상태 및 깊이
        depth, state, action_simple, dist, position = self._get_state()

        vx, vy, vz = action_simple * 0.2 + 0.8 * action

        self.client.moveByVelocityAsync(
            float(vx), float(vy), float(vz), 1.0
        )



        # 큐 업데이트
        self.depth_queue.append(depth)
        depth_stack = np.stack(self.depth_queue, axis=0)

        # 보상·종료 계산
        reward, done, info = self._compute_reward(depth, state, dist, position)
        self.steps += 1
        if self.steps >= Config.max_episode_steps:
            done = True
            info['timeout'] = True

        return depth_stack, state, reward, done, info

    def visualize_3d_lidar(self, depth_image, frame_count=0, show=True, save_path=None):
        """
        깊이 이미지의 3D 시각화 (학습 중 실시간 모니터링용)
        depth_image: 현재 라이다 깊이 이미지
        frame_count: 현재 프레임 번호 (표시용)
        show: 화면에 표시 여부
        save_path: 저장 경로 (None이면 저장하지 않음)
        """
        if not hasattr(self, 'lidar_fig') or self.lidar_fig is None:
            # 처음 호출 시 그림 초기화
            plt.ion()  # 대화형 모드 활성화
            self.lidar_fig = plt.figure(figsize=(10, 8))
            self.lidar_ax = self.lidar_fig.add_subplot(111, projection='3d')
            self.lidar_scatter = None
            self.drone_point = None

        # 이전 산점도 제거
        if self.lidar_scatter:
            self.lidar_scatter.remove()
        if self.drone_point:
            self.drone_point.remove()

        # 각도 설정
        x_size, y_size = Config.depth_image_width, Config.depth_image_height
        az = np.linspace(-np.pi, np.pi, x_size)
        el = np.linspace(np.pi / 2, -np.pi / 2, y_size)
        AZ, EL = np.meshgrid(az, el)

        # 깊이 값 변환
        R = depth_image * Config.max_lidar_distance

        # 표시할 데이터 수 줄이기 (성능 향상)
        skip = 4  # 4개마다 하나씩 표시

        # 구면 좌표를 3D 직교 좌표로 변환
        X = R[::skip, ::skip] * np.cos(EL[::skip, ::skip]) * np.cos(AZ[::skip, ::skip])
        Y = R[::skip, ::skip] * np.cos(EL[::skip, ::skip]) * np.sin(AZ[::skip, ::skip])
        Z = R[::skip, ::skip] * np.sin(EL[::skip, ::skip])

        # 데이터 평탄화
        X_flat = X.flatten()
        Y_flat = Y.flatten()
        Z_flat = Z.flatten()
        depth_flat = depth_image[::skip, ::skip].flatten()

        mask = depth_flat < 1.0
        X_plot = X_flat[mask]
        Y_plot = Y_flat[mask]
        Z_plot = Z_flat[mask]
        colors = depth_flat[mask]

        self.lidar_scatter = self.lidar_ax.scatter(
            X_plot, Y_plot, Z_plot,
            c=colors,
            cmap='viridis_r',
            s=3,
            alpha=0.8,
            marker='.'
        )

        # 드론 위치 표시
        self.drone_point = self.lidar_ax.scatter([0], [0], [0], color='red', s=50, marker='o')

        # 축 설정
        self.lidar_ax.set_title(f'Training Step: {frame_count}')
        self.lidar_ax.set_xlabel('North (m)')
        self.lidar_ax.set_ylabel('East (m)')
        self.lidar_ax.set_zlabel('Down (m)')

        # 축 범위 설정
        max_vis_range = Config.max_lidar_distance / 2
        self.lidar_ax.set_xlim([-max_vis_range, max_vis_range])
        self.lidar_ax.set_ylim([-max_vis_range, max_vis_range])
        self.lidar_ax.set_zlim([-max_vis_range, max_vis_range])

        # 화면 업데이트 (학습 속도에 영향을 최소화하기 위해 빠르게 처리)
        if show:
            self.lidar_fig.canvas.draw_idle()
            plt.pause(0.001)  # 매우 짧은 일시 중지

        # 프레임 저장
        if save_path:
            plt.savefig(save_path)

        return self.lidar_fig

    def _get_lidar_data(self):
        lidar_data = self.client.getLidarData("LidarSensor1", "HelloDrone")
        lidar_points = lidar_data.point_cloud
        pose = lidar_data.pose

        if len(lidar_points) >= 3:

            # 모든 포인트를 numpy 배열로 변환 (벡터화)
            points_count = len(lidar_points) // 3
            points = np.array(lidar_points).reshape(points_count, 3)

            qw, qx, qy, qz = pose.orientation.w_val, pose.orientation.x_val, pose.orientation.y_val, pose.orientation.z_val

            # ③ 쿼터니언 → 회전 행렬
            R = self._quaternion_to_rotation_matrix(qw, qx, qy, qz)

            # 역행렬(전치행렬)을 사용하여 body에서 NED로 변환
            pts_ned = points @ R.T

            # 변환된 포인트를 튜플 리스트로 변환
            points_list = [tuple(point) for point in pts_ned]

            return lidar_to_depth_image(points_list)

        return lidar_to_depth_image([])

    def _get_state(self):
        # 드론 상태와 방향 가져오기
        kinematics = self.client.simGetGroundTruthKinematics()
        position = np.array([kinematics.position.x_val, kinematics.position.y_val, kinematics.position.z_val])

        # 라이다 데이터에서 깊이 이미지 가져오기
        depth_image = self._get_lidar_data()

        # 속도 가져오기 (NED 좌표계)
        velocity_ned = np.array([
            kinematics.linear_velocity.x_val,
            kinematics.linear_velocity.y_val,
            kinematics.linear_velocity.z_val
        ])

        # NED 좌표계에서 목표까지의 상대 벡터
        rel = self.goal_position - position
        dist = np.linalg.norm(rel) + 1e-8
        dir_to_goal = rel / dist
        action_simple = dir_to_goal * Config.max_action

        # 바디 프레임 기준 상태 벡터: 속도(3), 목표 상대 위치(3), 드론 pose(3)
        drone_state = np.concatenate([velocity_ned, dir_to_goal])

        return depth_image, drone_state, action_simple, dist, position

    def _quaternion_to_rotation_matrix(self, w, x, y, z):
        # 쿼터니언을 회전 행렬로 변환
        rotation_matrix = np.array([
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]
        ])
        return rotation_matrix

    def _rotate_by_quaternion(self, point, q):
        """쿼터니언을 사용하여 점을 회전"""
        w, x, y, z = q
        point_q = np.array([0, point[0], point[1], point[2]])

        # 쿼터니언 곱셈을 사용한 회전
        q_conj = np.array([w, -x, -y, -z])
        rotated = self._quaternion_multiply(self._quaternion_multiply(q, point_q), q_conj)

        return rotated[1:]  # 벡터 부분만 반환

    def _quaternion_multiply(self, q1, q2):
        """두 쿼터니언의 곱셈"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return np.array([w, x, y, z])

    def _compute_reward(self, depth_image, drone_state, dist, position):
        # 현재 위치와 목표 위치 거리
        current_goal_distance = dist

        # 충돌 정보
        collision_info = self.client.simGetCollisionInfo()
        has_collided = collision_info.has_collided

        # 최소 깊이 계산 (가장 가까운 장애물 거리)
        min_depth = np.min(depth_image) * Config.max_lidar_distance

        # 보상 초기화
        reward = 0.0
        done = False

        # 터미널 상태 처리
        if current_goal_distance < Config.goal_threshold:
            reward = 50.0  # 목표 도달 보너스 (더 큰 값)
            done = True
            info = {"status": "goal_reached", "distance_to_goal": current_goal_distance}
            print(f"Reached: {position}")
        elif has_collided:
            reward = -50.0  # 충돌 페널티 (더 큰 부정값)
            done = True
            info = {"status": "collision", "distance_to_goal": current_goal_distance}
            print(f"Collided: {position}")
        else:
            # 지속적인 보상 계산 (비터미널 상태)

            # 1. 목표 접근 보상: 이전 거리와 현재 거리의 차이 (진전 보상)
            progress_reward = (self.prev_goal_distance - current_goal_distance) * 10.0

            # 3. 장애물 회피 보상: 안전 거리에 따른 보상/페널티
            danger_zone = 2.0
            caution_zone = 5.0

            if min_depth < danger_zone:
                # 매우 위험한 상황 (강한 페널티)
                obstacle_reward = -2.0 * ((danger_zone - min_depth) / danger_zone)
            elif min_depth < caution_zone:
                # 주의 구역 (약한 페널티)
                obstacle_reward = -0.5 * ((caution_zone - min_depth) / (caution_zone - danger_zone))
            else:
                # 안전 구역 (작은 보상)
                obstacle_reward = 0.1

            # 4. 에너지 효율성 보상: 최적 속도 유지 장려
            speed = np.linalg.norm(drone_state[:3])
            optimal_speed = Config.max_drone_speed * 0.7  # 최대 속도의 70%를 최적으로 가정
            efficiency_reward = -0.1 * abs(speed - optimal_speed) / optimal_speed

            # 5. 시간 패널티: 시간이 지날수록 작은 패널티
            time_penalty = -0.001 * self.steps

            # 모든 보상 요소 결합
            reward = progress_reward + obstacle_reward + efficiency_reward + time_penalty

            info = {
                "status": "moving",
                "distance_to_goal": current_goal_distance,
                "obstacle_distance": min_depth,
                "progress_reward": progress_reward,
                "obstacle_reward": obstacle_reward
            }

        # 이전 거리 업데이트
        self.prev_goal_distance = current_goal_distance

        return reward, done, info

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
    # 환경, 에이전트, 버퍼 초기화
    env = DroneEnv()
    agent = RSACAgent().to(Config.device)
    buffer = PrioritizedTransitionReplayBuffer(capacity=10000, alpha=0.6)
    tracker = TrainingTracker()

    # 로그 디렉토리 설정
    os.makedirs(Config.model_dir, exist_ok=True)
    os.makedirs(Config.log_dir, exist_ok=True)
    log_csv = os.path.join(Config.log_dir, 'training_log.csv')
    with open(log_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Episode','Steps','Reward','AvgReward','Status','CriticLoss','ActorLoss','Alpha'])

    # 체크포인트 로드
    latest = os.path.join(Config.model_dir, 'rsac_latest.pt')
    start_ep = 0
    if os.path.exists(latest):
        cp = torch.load(latest, map_location=Config.device)
        agent.load_state_dict(cp['model_state'])
        start_ep = cp.get('episode',0) + 1
        print(f"Resuming from episode {start_ep}")

    # 에피소드 루프
    for ep in range(start_ep, Config.max_episodes):
        depth_stack, state = env.reset()
        episode_reward = 0.0
        done = False
        steps = 0
        critic_loss = actor_loss = alpha = 0.0

        t0 = time.time()
        while not done and steps < Config.max_episode_steps:
            # 액션 선택
            d_tensor = torch.FloatTensor(depth_stack).unsqueeze(0).to(Config.device)  # (1,4,H,W)
            s_tensor = torch.FloatTensor(state).unsqueeze(0).to(Config.device)        # (1,state_dim)
            action = agent.select_action(d_tensor, s_tensor).cpu().numpy()[0]

            # 환경 스텝
            next_depth, next_state, reward, done, info = env.step(action)

            # 버퍼 저장
            buffer.push(depth_stack, state, action, reward, next_depth, next_state, done)

            episode_reward += reward
            steps += 1
            depth_stack, state = next_depth, next_state

            # 업데이트
            if len(buffer) > Config.batch_size and steps % 10 == 0:
                upd = agent.update(buffer)
                critic_loss = upd.get('critic_loss', critic_loss)
                actor_loss = upd.get('actor_loss', actor_loss)
                alpha = upd.get('alpha', alpha)

        # 에피소드 종료
        duration = time.time() - t0
        status = info.get('status', 'unknown')
        tracker.add_episode_stats(episode_reward, steps, status)
        avg_reward, _ = tracker.get_recent_stats()

        # 로그 기록
        with open(log_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([ep, steps, f"{episode_reward:.2f}", f"{avg_reward:.2f}", status,
                             f"{critic_loss:.4f}", f"{actor_loss:.4f}", f"{alpha:.4f}"])

        # 출력
        if ep % Config.log_interval == 0:
            print(f"Ep {ep} | Steps {steps} | Duration {duration} | Rwd {episode_reward:.2f} | AvgR {avg_reward:.2f} | {status}")

        # 모델 저장
        if ep % Config.save_interval == 0:
            ckpt = {'model_state': agent.state_dict(), 'episode': ep}
            torch.save(ckpt, latest)
            torch.save(ckpt, os.path.join(Config.model_dir, f"rsac_ep{ep}.pt"))

    # 최종 저장
    final = os.path.join(Config.model_dir, 'rsac_final.pt')
    torch.save({'model_state': agent.state_dict(), 'episode': Config.max_episodes-1}, final)
    print("Training complete.")

    return agent, tracker



# 메인 함수
if __name__ == "__main__":
    # 학습 시작
    trained_agent, training_stats = train_rsac()

    # 최종 결과 출력
    final_avg_reward, final_success_rate = training_stats.get_recent_stats()
    print(f"Final Results: Average Reward = {final_avg_reward:.2f}, Success Rate = {final_success_rate:.2f}")
