import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Normal
import math
import random
from collections import deque, namedtuple
import matplotlib.pyplot as plt
from copy import deepcopy


# Configuration for SAC with GRU
class SACConfig:
    # Environment
    max_episode_steps = 1000
    goal_threshold = 2.0  # Distance to goal considered as reached
    seq_length = 8  # 시퀀스 길이

    # Neural Network
    hidden_dim = 256
    cnn_features = 64
    gru_hidden_dim = 128  # GRU hidden dimension
    gru_num_layers = 2  # Number of GRU layers

    # SAC Parameters
    batch_size = 64
    buffer_size = 1000000
    gamma = 0.99  # Discount factor
    tau = 0.005  # For soft update of target network
    lr_actor = 3e-4  # Learning rate for actor
    lr_critic = 3e-4  # Learning rate for critic
    lr_alpha = 3e-4  # Learning rate for alpha parameter
    alpha_init = 0.2  # Initial temperature parameter
    target_entropy = -3  # Target entropy for automatic temperature tuning

    # Training
    start_steps = 1000  # Steps to take random actions for exploration
    update_interval = 1  # How often to update the networks

    # Dual Experience Buffer
    success_buffer_size = 300000
    fail_buffer_size = 700000
    sample_success_ratio = 0.4  # Ratio of samples from success buffer


# Convolutional Self-Attention Layer
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        # Query, Key, Value convolutions
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # Learnable scaling factor (gamma)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, height, width = x.size()

        # Compute query, key, value tensors
        query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)  # B x (H*W) x C'
        key = self.key(x).view(batch_size, -1, height * width)  # B x C' x (H*W)
        value = self.value(x).view(batch_size, -1, height * width)  # B x C x (H*W)

        # Calculate attention map
        attention = torch.bmm(query, key)  # B x (H*W) x (H*W)
        attention = F.softmax(attention, dim=2)

        # Compute output
        out = torch.bmm(value, attention.permute(0, 2, 1))  # B x C x (H*W)
        out = out.view(batch_size, C, height, width)

        # Apply gamma and add residual connection
        out = self.gamma * out + x

        return out


# CNN for processing depth images
class CNNFeatureExtractor(nn.Module):
    def __init__(self, input_channels=1, output_dim=SACConfig.cnn_features):
        super(CNNFeatureExtractor, self).__init__()

        # First convolutional block
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.attention1 = SelfAttention(32)

        # Second convolutional block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.attention2 = SelfAttention(64)

        # Third convolutional block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        # Fourth convolutional block
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        # For 42x42 resolution:
        self.fc = nn.Linear(128 * 5 * 5, output_dim)

    def forward(self, x):
        """
        CNN 특징 추출기

        Args:
            x: 입력 이미지 [B, C, H, W] 형태만 지원
        """
        # 입력 형태 검증 및 수정
        if len(x.shape) != 4:
            raise ValueError(f"Input must be [B, C, H, W], got {x.shape}")

        if x.shape[1] != 1:
            raise ValueError(f"Expected 1 channel, got {x.shape[1]}")

        # First block
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = self.attention1(x)

        # Second block
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = self.attention2(x)

        # Third and fourth blocks
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        # Flatten and pass through FC layer
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))

        return x


# GRU-based Actor Network
class GRUActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, max_action=1.0):
        super(GRUActorNetwork, self).__init__()

        # CNN for feature extraction from each frame
        self.cnn = CNNFeatureExtractor(input_channels=1)

        # GRU for temporal processing
        self.gru = nn.GRU(
            input_size=SACConfig.cnn_features,
            hidden_size=SACConfig.gru_hidden_dim,
            num_layers=SACConfig.gru_num_layers,
            batch_first=True
        )

        # Fully connected layers for policy
        self.fc1 = nn.Linear(SACConfig.gru_hidden_dim + state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Output layers for Gaussian policy
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

        self.max_action = max_action
        self.action_dim = action_dim

    def forward(self, depth_images, state, hidden=None):
        """
        시퀀스 데이터를 처리하여 정책 네트워크 출력 생성

        Args:
            depth_images: 깊이 이미지 시퀀스 [batch_size, seq_len, H, W]
            state: 현재 상태 [batch_size, state_dim]
            hidden: GRU 히든 스테이트 (선택 사항)

        Returns:
            mean: 행동 평균값 [batch_size, action_dim]
            log_std: 행동 로그 표준편차 [batch_size, action_dim]
            gru_hidden: 다음 스텝을 위한 GRU 히든 스테이트
        """
        batch_size = depth_images.shape[0]
        seq_len = depth_images.shape[1]

        # CNN 특징 추출
        cnn_features = []
        for i in range(seq_len):
            # 시퀀스에서 단일 프레임 추출
            frame = depth_images[:, i]  # [batch_size, H, W]

            # CNN 입력을 위한 채널 차원 추가
            frame = frame.unsqueeze(1)  # [batch_size, 1, H, W]

            # CNN으로 처리
            frame_features = self.cnn(frame)  # [batch_size, cnn_features]
            cnn_features.append(frame_features)

        # CNN 특징들을 시퀀스로 쌓기
        cnn_features = torch.stack(cnn_features, dim=1)  # [batch_size, seq_len, cnn_features]

        # GRU로 시퀀스 처리
        if hidden is None:
            gru_out, gru_hidden = self.gru(cnn_features)
        else:
            gru_out, gru_hidden = self.gru(cnn_features, hidden)

        # GRU 시퀀스의 마지막 출력 사용
        gru_features = gru_out[:, -1]  # [batch_size, gru_hidden_dim]

        # GRU 특징과 상태 벡터 합치기
        combined = torch.cat([gru_features, state], dim=1)  # [batch_size, gru_hidden_dim + state_dim]

        # 완전 연결 레이어
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))

        # 정책 출력 (평균 및 로그 표준편차)
        mean = self.mean(x)
        log_std = self.log_std(x)

        # 로그 표준편차 제한
        log_std = torch.clamp(log_std, min=-20, max=2)

        return mean, log_std, gru_hidden

    def sample(self, depth_images, state, hidden=None):
        """정책에서 행동 샘플링"""
        mean, log_std, gru_hidden = self.forward(depth_images, state, hidden)
        std = log_std.exp()

        # 정규 분포 생성
        normal = Normal(mean, std)

        # 리파라미터화 트릭을 사용한 샘플링
        x_t = normal.rsample()
        log_prob = normal.log_prob(x_t).sum(dim=1, keepdim=True)

        # tanh 스쿼싱 적용
        action = torch.tanh(x_t)

        # 스쿼싱에 따른 로그 확률 보정
        log_prob -= torch.sum(torch.log(1 - action.pow(2) + 1e-6), dim=1, keepdim=True)

        # 행동 스케일링
        action = action * self.max_action

        return action, log_prob, torch.tanh(mean) * self.max_action, gru_hidden


# GRU-based Critic Network
class GRUCriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(GRUCriticNetwork, self).__init__()

        # CNN for feature extraction from each frame
        self.cnn = CNNFeatureExtractor(input_channels=1)

        # GRU for temporal processing
        self.gru = nn.GRU(
            input_size=SACConfig.cnn_features,
            hidden_size=SACConfig.gru_hidden_dim,
            num_layers=SACConfig.gru_num_layers,
            batch_first=True
        )

        # Q1 Network
        self.q1_fc1 = nn.Linear(SACConfig.gru_hidden_dim + state_dim + action_dim, hidden_dim)
        self.q1_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_out = nn.Linear(hidden_dim, 1)

        # Q2 Network
        self.q2_fc1 = nn.Linear(SACConfig.gru_hidden_dim + state_dim + action_dim, hidden_dim)
        self.q2_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_out = nn.Linear(hidden_dim, 1)

    def forward(self, depth_images, state, action, hidden=None):
        """
        시퀀스 데이터를 처리하여 Q값 생성

        Args:
            depth_images: 깊이 이미지 시퀀스 [batch_size, seq_len, H, W]
            state: 현재 상태 [batch_size, state_dim]
            action: 행동 [batch_size, action_dim]
            hidden: GRU 히든 스테이트 (선택 사항)

        Returns:
            q1: 첫 번째 Q 값 [batch_size, 1]
            q2: 두 번째 Q 값 [batch_size, 1]
            gru_hidden: 다음 스텝을 위한 GRU 히든 스테이트
        """
        batch_size = depth_images.shape[0]
        seq_len = depth_images.shape[1]

        # CNN 특징 추출
        cnn_features = []
        for i in range(seq_len):
            # 시퀀스에서 단일 프레임 추출
            frame = depth_images[:, i]  # [batch_size, H, W]

            # CNN 입력을 위한 채널 차원 추가
            frame = frame.unsqueeze(1)  # [batch_size, 1, H, W]

            # CNN으로 처리
            frame_features = self.cnn(frame)  # [batch_size, cnn_features]
            cnn_features.append(frame_features)

        # CNN 특징들을 시퀀스로 쌓기
        cnn_features = torch.stack(cnn_features, dim=1)  # [batch_size, seq_len, cnn_features]

        # GRU로 시퀀스 처리
        if hidden is None:
            gru_out, gru_hidden = self.gru(cnn_features)
        else:
            gru_out, gru_hidden = self.gru(cnn_features, hidden)

        # GRU 시퀀스의 마지막 출력 사용
        gru_features = gru_out[:, -1]  # [batch_size, gru_hidden_dim]

        # GRU 특징, 상태, 행동 벡터 합치기
        combined = torch.cat([gru_features, state, action], dim=1)

        # Q1 네트워크
        q1 = F.relu(self.q1_fc1(combined))
        q1 = F.relu(self.q1_fc2(q1))
        q1 = self.q1_out(q1)

        # Q2 네트워크
        q2 = F.relu(self.q2_fc1(combined))
        q2 = F.relu(self.q2_fc2(q2))
        q2 = self.q2_out(q2)

        return q1, q2, gru_hidden

    def q1(self, depth_images, state, action, hidden=None):
        """Return only Q1 value for action selection"""
        q1, _, gru_hidden = self.forward(depth_images, state, action, hidden)
        return q1, gru_hidden


# Sequence-based Experience Buffer
class SequenceExperienceBuffer:
    def __init__(self, success_buffer_size, fail_buffer_size, state_dim, action_dim, seq_length=SACConfig.seq_length):
        self.success_buffer = deque(maxlen=success_buffer_size)
        self.fail_buffer = deque(maxlen=fail_buffer_size)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.seq_length = seq_length

        # 현재 에피소드 임시 저장소
        self.current_episode = []

        # 시퀀스 데이터 구조 정의
        self.experience = namedtuple("Experience",
                                     field_names=["depth_sequence", "state_sequence",
                                                  "action_sequence", "reward_sequence",
                                                  "next_depth_sequence", "next_state_sequence",
                                                  "done_sequence", "is_success"])

    def add(self, depth_image, state, action, reward, next_depth_image, next_state, done, is_success):
        """
        환경에서 단일 스텝 데이터 추가

        Args:
            depth_image: 현재 깊이 이미지 [H, W] 또는 시퀀스 [seq, H, W]
            state: 현재 상태 벡터 [state_dim]
            action: 행동 벡터 [action_dim]
            reward: 보상 값 (스칼라)
            next_depth_image: 다음 깊이 이미지 [H, W] 또는 시퀀스 [seq, H, W]
            next_state: 다음 상태 벡터 [state_dim]
            done: 에피소드 종료 여부 (불리언)
            is_success: 성공 여부 (불리언)
        """
        # 시퀀스 입력을 처리하는 경우, 마지막 프레임만 사용
        if len(depth_image.shape) > 2:  # 시퀀스 감지
            depth_image = depth_image[-1]  # 마지막 프레임만 사용

        if len(next_depth_image.shape) > 2:  # 시퀀스 감지
            next_depth_image = next_depth_image[-1]  # 마지막 프레임만 사용

        # 현재 에피소드에 스텝 데이터 추가
        self.current_episode.append(
            (depth_image, state, action, reward, next_depth_image, next_state, done, is_success))

        # 에피소드가 종료되면 시퀀스 생성 및 버퍼에 추가
        if done:
            self._process_episode(is_success)
            self.current_episode = []  # 에피소드 초기화

    def _process_episode(self, final_is_success):
        """에피소드 완료 시 시퀀스로 분할하여 버퍼에 추가"""
        if len(self.current_episode) < self.seq_length:
            return  # 너무 짧은 에피소드는 무시

        # 에피소드를 시퀀스로 분할 (sliding window 방식)
        for i in range(0, len(self.current_episode) - self.seq_length + 1):
            seq_slice = self.current_episode[i:i + self.seq_length]

            # 시퀀스 데이터 추출
            depth_sequence = np.array([step[0] for step in seq_slice])  # [seq_len, H, W]
            state_sequence = np.array([step[1] for step in seq_slice])
            action_sequence = np.array([step[2] for step in seq_slice])
            reward_sequence = np.array([step[3] for step in seq_slice])
            next_depth_sequence = np.array([step[4] for step in seq_slice])
            next_state_sequence = np.array([step[5] for step in seq_slice])
            done_sequence = np.array([step[6] for step in seq_slice])

            # 마지막 스텝만 done=True로 설정
            seq_is_done = done_sequence[-1]

            # 에피소드의 최종 결과에 따라 성공/실패 결정
            is_success_seq = final_is_success

            # 경험 객체 생성 및 버퍼에 추가
            e = self.experience(
                depth_sequence, state_sequence, action_sequence,
                reward_sequence, next_depth_sequence, next_state_sequence,
                np.array([False] * (self.seq_length - 1) + [seq_is_done]), is_success_seq
            )

            if is_success_seq:
                self.success_buffer.append(e)
            else:
                self.fail_buffer.append(e)

    def sample(self, batch_size, success_ratio=0.4):
        """
        버퍼에서 시퀀스 데이터 샘플링

        Args:
            batch_size: 샘플링할 배치 크기
            success_ratio: 성공 버퍼에서 샘플링할 비율

        Returns:
            각 필드의 텐서 배치
        """
        # 성공/실패 버퍼에서 샘플 수 계산
        success_samples = int(batch_size * success_ratio)
        fail_samples = batch_size - success_samples

        # 버퍼 크기에 따라 샘플 수 조정
        if len(self.success_buffer) < success_samples:
            success_samples = len(self.success_buffer)
            fail_samples = batch_size - success_samples

        if len(self.fail_buffer) < fail_samples:
            fail_samples = len(self.fail_buffer)
            success_samples = batch_size - fail_samples

        # 버퍼에서 샘플링
        success_experiences = random.sample(self.success_buffer, min(success_samples, len(self.success_buffer))) if len(
            self.success_buffer) > 0 else []
        fail_experiences = random.sample(self.fail_buffer, min(fail_samples, len(self.fail_buffer))) if len(
            self.fail_buffer) > 0 else []

        # 성공/실패 샘플 결합 및 셔플
        experiences = success_experiences + fail_experiences
        random.shuffle(experiences)

        # 샘플이 없는 경우 처리
        if len(experiences) == 0:
            return None, None, None, None, None, None, None

        # 텐서 변환 (배치 및 시퀀스 차원 유지)
        depth_sequences = torch.from_numpy(np.array([e.depth_sequence for e in experiences if e is not None])).float()
        state_sequences = torch.from_numpy(np.array([e.state_sequence for e in experiences if e is not None])).float()
        action_sequences = torch.from_numpy(np.array([e.action_sequence for e in experiences if e is not None])).float()
        reward_sequences = torch.from_numpy(np.array([e.reward_sequence for e in experiences if e is not None])).float()
        next_depth_sequences = torch.from_numpy(
            np.array([e.next_depth_sequence for e in experiences if e is not None])).float()
        next_state_sequences = torch.from_numpy(
            np.array([e.next_state_sequence for e in experiences if e is not None])).float()
        done_sequences = torch.from_numpy(
            np.array([e.done_sequence for e in experiences if e is not None]).astype(np.uint8)).float()

        # 각 시퀀스의 마지막 요소만 반환 (현재 상태와 다음 상태)
        states = state_sequences[:, -1]
        actions = action_sequences[:, -1]
        rewards = reward_sequences[:, -1].unsqueeze(1)
        next_states = next_state_sequences[:, -1]
        dones = done_sequences[:, -1].unsqueeze(1)

        return depth_sequences, states, actions, rewards, next_depth_sequences, next_states, dones

    def __len__(self):
        return len(self.success_buffer) + len(self.fail_buffer)

    def success_buffer_len(self):
        return len(self.success_buffer)

    def fail_buffer_len(self):
        return len(self.fail_buffer)


# GRU-based SAC Agent
class GRUSACAgent:
    def __init__(self, state_dim, action_dim, max_action=1.0, device="cpu"):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.device = device

        # Initialize networks
        self.actor = GRUActorNetwork(state_dim, action_dim, SACConfig.hidden_dim, max_action).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=SACConfig.lr_actor)

        self.critic = GRUCriticNetwork(state_dim, action_dim, SACConfig.hidden_dim).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=SACConfig.lr_critic)

        self.critic_target = GRUCriticNetwork(state_dim, action_dim, SACConfig.hidden_dim).to(device)

        # Copy weights from critic to target
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        # Initialize temperature parameter alpha (for entropy)
        self.log_alpha = torch.tensor(np.log(SACConfig.alpha_init)).to(device)
        self.log_alpha.requires_grad = True
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=SACConfig.lr_alpha)

        # Target entropy
        self.target_entropy = SACConfig.target_entropy

        # Experience buffer - 시퀀스 버퍼로 변경
        self.memory = SequenceExperienceBuffer(
            SACConfig.success_buffer_size,
            SACConfig.fail_buffer_size,
            state_dim,
            action_dim,
            SACConfig.seq_length
        )

        # GRU hidden states
        self.actor_hidden = None
        self.critic_hidden = None

        # Training step counter
        self.training_steps = 0

        # Frame history for sequential processing
        self.frame_history = deque(maxlen=SACConfig.seq_length)

    def reset_hidden(self):
        """Reset hidden states between episodes"""
        self.actor_hidden = None
        self.critic_hidden = None
        self.frame_history.clear()

    def select_action(self, depth_image, state, evaluate=False):
        """
        현재 정책으로 행동 선택

        Args:
            depth_image: 현재 깊이 이미지 [H, W]
            state: 현재 상태 [state_dim]
            evaluate: 평가 모드 여부 (결정론적 정책 사용)

        Returns:
            선택된 행동 [action_dim]
        """
        # 프레임 히스토리에 이미지 추가
        if len(self.frame_history) < self.frame_history.maxlen:
            # 초기 히스토리 채우기
            for _ in range(self.frame_history.maxlen - len(self.frame_history)):
                self.frame_history.append(depth_image.copy())
        else:
            self.frame_history.append(depth_image.copy())

        # 프레임 히스토리를 시퀀스로 변환
        depth_sequence = np.array(list(self.frame_history))  # [seq_len, H, W]

        # 텐서로 변환
        depth_sequence = torch.FloatTensor(depth_sequence).to(self.device)
        state = torch.FloatTensor(state).to(self.device)

        # 배치 차원 추가
        if len(state.shape) == 1:
            state = state.unsqueeze(0)  # [1, state_dim]

        # 시퀀스에 배치 차원 추가
        if len(depth_sequence.shape) == 3:  # [seq_len, H, W]
            depth_sequence = depth_sequence.unsqueeze(0)  # [1, seq_len, H, W]

        # 행동 선택 (탐색 있음/없음)
        with torch.no_grad():
            if evaluate:
                # 평가 모드: 평균 행동 사용
                _, _, mean_action, self.actor_hidden = self.actor.sample(depth_sequence, state, self.actor_hidden)
                action = mean_action
            else:
                # 학습 모드: 확률적 행동 샘플링
                action, _, _, self.actor_hidden = self.actor.sample(depth_sequence, state, self.actor_hidden)

        return action.detach().cpu().numpy()[0]

    def update_parameters(self, delayed_update=True):
        """
        SAC 알고리즘으로 네트워크 파라미터 업데이트

        Args:
            delayed_update: 지연 업데이트 여부

        Returns:
            업데이트 정보 딕셔너리 또는 None (업데이트가 없는 경우)
        """
        # 충분한 샘플이 있는 경우에만 업데이트
        if len(self.memory) < SACConfig.batch_size:
            return None

        # 카운터 증가
        self.training_steps += 1

        # 지연 학습을 사용하고 아직 업데이트 시간이 아닌 경우 스킵
        if delayed_update and self.training_steps % SACConfig.update_interval != 0:
            return None

        # 재생 버퍼에서 샘플링
        sampled_data = self.memory.sample(SACConfig.batch_size, SACConfig.sample_success_ratio)

        # 데이터가 없는 경우 처리
        if sampled_data[0] is None:
            return None

        # 샘플 데이터 언패킹
        depth_sequences, states, actions, rewards, next_depth_sequences, next_states, dones = sampled_data

        # 텐서를 디바이스로 이동
        depth_sequences = depth_sequences.to(self.device)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_depth_sequences = next_depth_sequences.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # 현재 온도 파라미터
        alpha = self.log_alpha.exp()

        # ---------- Critic 업데이트 ----------
        with torch.no_grad():
            # 타겟 정책에서 행동 샘플링
            next_actions, next_log_probs, _, _ = self.actor.sample(next_depth_sequences, next_states)

            # 타겟 Q 값 계산
            target_q1, target_q2, _ = self.critic_target(next_depth_sequences, next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - alpha * next_log_probs
            target_q = rewards + (1 - dones) * SACConfig.gamma * target_q

        # 현재 Q 값 계산
        current_q1, current_q2, _ = self.critic(depth_sequences, states, actions)

        # Critic 손실 계산
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        # Critic 업데이트
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------- Actor 업데이트 ----------
        # 현재 정책에서 행동 샘플링
        actions_new, log_probs_new, _, _ = self.actor.sample(depth_sequences, states)

        # Actor 손실 계산
        q1_new, q2_new, _ = self.critic(depth_sequences, states, actions_new)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (alpha * log_probs_new - q_new).mean()

        # Actor 업데이트
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ---------- 온도 파라미터 업데이트 ----------
        alpha_loss = -(self.log_alpha * (log_probs_new.detach() + self.target_entropy)).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # ---------- 타겟 네트워크 소프트 업데이트 ----------
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - SACConfig.tau) + param.data * SACConfig.tau
            )

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha': alpha.item(),
            'q_value': q_new.mean().item(),
            'log_prob': log_probs_new.mean().item()
        }

    def save(self, filename):
        """Save agent state"""
        try:
            torch.save({
                'actor_state_dict': self.actor.state_dict(),
                'critic_state_dict': self.critic.state_dict(),
                'critic_target_state_dict': self.critic_target.state_dict(),
                'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
                'log_alpha': self.log_alpha,
                'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict(),
                'training_steps': self.training_steps
            }, filename)
            print(f"Model saved to {filename}")
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False

    def load(self, filename):
        """Load agent state"""
        try:
            # Try with newer PyTorch version using weights_only=True for security
            try:
                checkpoint = torch.load(filename, map_location=self.device, weights_only=True)
            except TypeError:
                # Fall back to original method if weights_only not supported
                checkpoint = torch.load(filename, map_location=self.device)

            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.log_alpha = checkpoint['log_alpha']
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
            self.training_steps = checkpoint['training_steps']

            # Reset hidden states
            self.reset_hidden()

            print(f"Model loaded from {filename}")
            print(f"Training steps: {self.training_steps}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False