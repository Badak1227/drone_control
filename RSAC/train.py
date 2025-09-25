"""
Recurrent SAC (Soft Actor-Critic) 구현
드론 내비게이션을 위한 안정적인 구현
"""

import os
import time
import csv
import copy
import random
import numpy as np
from collections import deque
import threading
import traceback
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt
import argparse

# 시드 설정
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.cuda.manual_seed_all(SEED)

#############################################################################
# 설정 및 하이퍼파라미터
#############################################################################

class SACConfig:
    # 네트워크 아키텍처
    cnn_features = 256
    hidden_dim = 256
    gru_hidden_dim = 256
    gru_layers = 1

    # SAC 하이퍼파라미터
    gamma = 0.99
    tau = 0.005
    alpha = 0.2
    learning_rate = 3e-4

    # 경험 리플레이
    buffer_size = 100000
    batch_size = 64
    sequence_length = 40

    # 학습
    update_every = 1
    update_freq = 1

    # 액션 공간
    action_dim = 3
    action_scale = 3.0

    # 학습 설정
    num_episodes = 1500
    save_interval = 50
    log_interval = 10

    # 디버깅
    debug_prints = False  # 디버깅 출력 활성화 여부

    # 디바이스
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#############################################################################
# 유틸리티 함수
#############################################################################

def debug_print(*args, **kwargs):
    """디버깅 출력 함수 (설정에 따라 활성화/비활성화)"""
    if SACConfig.debug_prints:
        print(*args, **kwargs)

def safe_concat(tensors, dim=-1):
    """안전한 텐서 결합 함수"""
    # 차원 확인
    dims = [t.dim() for t in tensors]
    max_dim = max(dims)

    # 모든 텐서를 동일한 차원으로 맞춤
    expanded_tensors = []
    for i, tensor in enumerate(tensors):
        if tensor.dim() < max_dim:
            # 필요한 차원 추가
            if max_dim - tensor.dim() == 1:
                expanded_tensors.append(tensor.unsqueeze(1))
            else:
                # 더 복잡한 경우 (예외 처리)
                raise ValueError(f"Cannot safely expand tensor {i} from {tensor.dim()}D to {max_dim}D")
        else:
            expanded_tensors.append(tensor)

    # 차원 크기 확인
    for i, t in enumerate(expanded_tensors):
        debug_print(f"Tensor {i} shape before concat: {t.shape}")

    # 배치와 시퀀스 차원이 일치하는지 확인
    shapes = [t.shape for t in expanded_tensors]
    for i in range(len(shapes) - 1):
        if shapes[i][0] != shapes[i+1][0]:  # 배치 크기 확인
            raise ValueError(f"Batch dimensions do not match: {shapes[i][0]} vs {shapes[i+1][0]}")
        if len(shapes[i]) > 1 and len(shapes[i+1]) > 1 and shapes[i][1] != shapes[i+1][1]:  # 시퀀스 길이 확인
            # 같은 배치 크기의 시퀀스를 expand로 맞춤
            seq_len = max(shapes[i][1], shapes[i+1][1])
            if shapes[i][1] < seq_len:
                expanded_tensors[i] = expanded_tensors[i].expand(-1, seq_len, -1)
            if shapes[i+1][1] < seq_len:
                expanded_tensors[i+1] = expanded_tensors[i+1].expand(-1, seq_len, -1)

    # 결합
    return torch.cat(expanded_tensors, dim=dim)

#############################################################################
# 신경망 모델
#############################################################################

class CNNFeatureExtractor(nn.Module):
    """깊이 이미지에서 특징 추출하는 CNN"""

    def __init__(self, input_channels=1, output_dim=SACConfig.cnn_features):
        super(CNNFeatureExtractor, self).__init__()

        # CNN 레이어
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        # 완전 연결 레이어
        self.fc = nn.Linear(128 * 5 * 5, output_dim)

        # 입력 차원 변환 처리 위한 플래그
        self.batch_seq_mode = False

    def forward(self, x):
        # 원본 형태 저장
        original_shape = x.shape
        debug_print(f"CNN input shape: {original_shape}")

        # 입력 형태 처리
        batch_size = original_shape[0]

        if len(original_shape) == 5:  # [B, S, C, H, W]
            seq_len = original_shape[1]
            self.batch_seq_mode = True
            # 배치와 시퀀스 차원을 합침
            x = x.view(batch_size * seq_len, *original_shape[2:])  # [B*S, C, H, W]
        elif len(original_shape) == 4:  # [B, C, H, W]
            self.batch_seq_mode = False
            seq_len = 1
        else:
            raise ValueError(f"Expected input with shape [B, C, H, W] or [B, S, C, H, W], got {original_shape}")

        # 채널 검증
        if x.shape[1] != 1:
            raise ValueError(f"Expected 1 channel, got {x.shape[1]}")

        # CNN 처리
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        # 완전 연결 레이어를 위해 펼치기
        x = x.view(x.size(0), -1)
        features = F.relu(self.fc(x))

        # 원래 형태로 복원
        if self.batch_seq_mode:
            # [B*S, F] -> [B, S, F]
            features = features.view(batch_size, seq_len, -1)
        else:
            # [B, F] -> [B, 1, F] (시퀀스 차원 추가)
            features = features.unsqueeze(1)

        debug_print(f"CNN output shape: {features.shape}")
        return features


class RecurrentActor(nn.Module):
    """리커런트 액터 네트워크"""

    def __init__(self, state_dim, action_dim, hidden_dim, gru_hidden_dim, gru_layers):
        super(RecurrentActor, self).__init__()

        # 특징 추출기
        self.cnn = CNNFeatureExtractor()
        self.state_fc = nn.Linear(state_dim, hidden_dim)

        # GRU 레이어
        self.gru = nn.GRU(
            input_size=SACConfig.cnn_features + hidden_dim,
            hidden_size=gru_hidden_dim,
            num_layers=gru_layers,
            batch_first=True
        )

        # 액션 헤드
        self.mean_fc = nn.Linear(gru_hidden_dim, action_dim)
        self.log_std_fc = nn.Linear(gru_hidden_dim, action_dim)

        # 가중치 초기화
        self._init_weights()

    def _init_weights(self):
        """가중치 초기화 메서드"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, depth, state, hidden=None):
        """순전파 메서드"""
        # GRU 메모리 최적화 - try-except로 감싸서 에러 방지
        try:
            self.gru.flatten_parameters()
        except RuntimeError:
            # 이미 최적화되었거나 CUDA 컨텍스트 문제 등으로 실패할 경우 무시
            pass

        debug_print(f"Actor input - depth: {depth.shape}, state: {state.shape}")

        # 입력 형태 추적
        if len(depth.shape) == 5:  # [B, S, C, H, W]
            batch_size, seq_len = depth.shape[0], depth.shape[1]
        elif len(depth.shape) == 4:  # [B, C, H, W]
            batch_size = depth.shape[0]
            seq_len = 1
            # 시퀀스 차원 추가
            depth = depth.unsqueeze(1)  # [B, 1, C, H, W]
        else:
            raise ValueError(f"Unexpected depth shape: {depth.shape}")

        # 상태 형태 확인 및 조정
        if len(state.shape) == 2:  # [B, state_dim]
            state = state.unsqueeze(1)  # [B, 1, state_dim]

        # 시퀀스 길이 일치 확인 및 조정
        if state.shape[1] != seq_len:
            if state.shape[1] == 1:
                # 단일 상태를 시퀀스 길이에 맞게 확장
                state = state.expand(-1, seq_len, -1)
            elif seq_len == 1:
                # 시퀀스의 첫 번째 상태만 사용
                state = state[:, 0:1, :]
            else:
                # 기타 불일치 처리
                raise ValueError(f"Sequence length mismatch: depth {seq_len}, state {state.shape[1]}")

        # CNN 특징 추출 - 수정된 CNNFeatureExtractor는 항상 [B, S, F] 형태 반환
        img_feat = self.cnn(depth)

        # 상태 특징 추출
        state_flat = state.reshape(-1, state.shape[-1])
        state_feat_flat = F.relu(self.state_fc(state_flat))
        state_feat = state_feat_flat.reshape(batch_size, seq_len, -1)

        debug_print(f"CNN feature shape: {img_feat.shape}")
        debug_print(f"State feature shape: {state_feat.shape}")

        # 차원 일관성 확인
        if img_feat.shape[0] != state_feat.shape[0] or img_feat.shape[1] != state_feat.shape[1]:
            raise ValueError(f"Feature dimensions mismatch: img_feat {img_feat.shape}, state_feat {state_feat.shape}")

        # 특징 결합
        combo = torch.cat([img_feat, state_feat], dim=-1)

        # GRU 처리
        output, hidden = self.gru(combo, hidden)

        # 액션 분포 매개변수 계산
        mean = self.mean_fc(output)
        log_std = torch.clamp(self.log_std_fc(output), -20, 2)

        debug_print(f"Actor output - mean: {mean.shape}, log_std: {log_std.shape}")
        return mean, log_std, hidden

    def sample(self, depth, state, hidden=None):
        """액션 샘플링 메서드"""
        mean, log_std, hidden = self.forward(depth, state, hidden)

        # 정규 분포에서 샘플링
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()

        # tanh 변환
        action = torch.tanh(x_t)

        # 로그 확률 계산 (tanh 변환 보정 포함)
        log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)

        # 액션 스케일링
        scaled_action = action * SACConfig.action_scale

        # 역전파 그래프 최적화를 위해 hidden 분리
        return scaled_action, log_prob, hidden.detach()


class RecurrentCritic(nn.Module):
    """리커런트 크리틱 네트워크"""

    def __init__(self, state_dim, action_dim, hidden_dim, gru_hidden_dim, gru_layers):
        super(RecurrentCritic, self).__init__()

        # 특징 추출기
        self.cnn = CNNFeatureExtractor()
        self.state_fc = nn.Linear(state_dim, hidden_dim)
        self.action_fc = nn.Linear(action_dim, hidden_dim)

        # GRU 레이어
        self.gru = nn.GRU(
            input_size=SACConfig.cnn_features + hidden_dim * 2,
            hidden_size=gru_hidden_dim,
            num_layers=gru_layers,
            batch_first=True
        )

        # Q값 헤드
        self.q_fc = nn.Linear(gru_hidden_dim, 1)

        # 가중치 초기화
        self._init_weights()

    def _init_weights(self):
        """가중치 초기화 메서드"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, depth, state, action, hidden=None):
        """순전파 메서드"""
        # GRU 메모리 최적화 - try-except로 감싸서 에러 방지
        try:
            self.gru.flatten_parameters()
        except RuntimeError:
            # 이미 최적화되었거나 CUDA 컨텍스트 문제 등으로 실패할 경우 무시
            pass

        debug_print(f"Critic input - depth: {depth.shape}, state: {state.shape}, action: {action.shape}")

        # CNN 특징 추출
        img_feat = self.cnn(depth)

        # 상태와 액션 특징 추출
        if len(state.shape) == 3:  # [B, S, state_dim]
            batch_size, seq_len = state.shape[0], state.shape[1]

            # 상태 처리
            state_flat = state.reshape(-1, state.shape[-1])
            s_feat = F.relu(self.state_fc(state_flat))
            s_feat = s_feat.reshape(batch_size, seq_len, -1)

            # 액션 처리
            action_flat = action.reshape(-1, action.shape[-1])
            a_feat = F.relu(self.action_fc(action_flat))
            a_feat = a_feat.reshape(batch_size, seq_len, -1)
        else:  # [B, state_dim/action_dim]
            s_feat = F.relu(self.state_fc(state))
            a_feat = F.relu(self.action_fc(action))

        # 특징 결합 (안전한 concat 사용)
        combo = safe_concat([img_feat, s_feat, a_feat], dim=-1)

        # GRU 입력 준비
        if combo.dim() == 2:  # [B, F]
            combo = combo.unsqueeze(1)  # [B, 1, F]

        # GRU 처리
        output, hidden = self.gru(combo, hidden)
        q_value = self.q_fc(output)

        # 출력 형태 정리
        if q_value.dim() == 3:  # [B, S, 1]
            return_q = q_value
        else:  # 차원 추가 필요
            return_q = q_value.unsqueeze(-1)

        debug_print(f"Critic output - q_value: {return_q.shape}")
        return return_q, hidden.detach()


#############################################################################
# 리플레이 버퍼
#############################################################################

class RecurrentReplayBuffer:
    """시퀀스 데이터를 위한 리플레이 버퍼"""

    def __init__(self, buffer_size, batch_size, sequence_length, device):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.device = device

        # 데이터 저장소
        self.depth_images = []
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_depth_images = []
        self.next_states = []
        self.dones = []

        # 에피소드 경계 (종료 인덱스)
        self.episode_boundaries = []

        # 현재 에피소드 버퍼
        self.reset_episode_buffer()

    def reset_episode_buffer(self):
        """에피소드 버퍼 초기화"""
        self.current_episode = {
            'depth_images': [],
            'states': [],
            'actions': [],
            'rewards': [],
            'next_depth_images': [],
            'next_states': [],
            'dones': []
        }

    def add(self, depth_image, state, action, reward, next_depth_image, next_state, done):
        """경험 추가"""
        self.current_episode['depth_images'].append(depth_image)
        self.current_episode['states'].append(state)
        self.current_episode['actions'].append(action)
        self.current_episode['rewards'].append(reward)
        self.current_episode['next_depth_images'].append(next_depth_image)
        self.current_episode['next_states'].append(next_state)
        self.current_episode['dones'].append(done)

        # 에피소드 종료 시 메인 버퍼에 저장
        if done:
            self._store_episode()

    def _store_episode(self):
        """에피소드를 메인 버퍼에 저장"""
        episode_length = len(self.current_episode['states'])

        if episode_length > 0:
            # 메인 버퍼에 에피소드 추가
            self.depth_images.extend(self.current_episode['depth_images'])
            self.states.extend(self.current_episode['states'])
            self.actions.extend(self.current_episode['actions'])
            self.rewards.extend(self.current_episode['rewards'])
            self.next_depth_images.extend(self.current_episode['next_depth_images'])
            self.next_states.extend(self.current_episode['next_states'])
            self.dones.extend(self.current_episode['dones'])

            # 에피소드 경계 업데이트
            if not self.episode_boundaries:
                self.episode_boundaries.append(episode_length - 1)
            else:
                self.episode_boundaries.append(self.episode_boundaries[-1] + episode_length)

            # 버퍼 크기 제한
            if len(self.states) > self.buffer_size:
                self._trim_buffer()

            # 디버깅 정보
            debug_print(f"Stored episode of length {episode_length}. Buffer size: {len(self.states)}")

        # 에피소드 버퍼 초기화
        self.reset_episode_buffer()

    def _trim_buffer(self):
        """버퍼 크기가 초과되면 오래된 데이터 제거"""
        overflow = len(self.states) - self.buffer_size

        # 오래된 데이터 제거
        self.depth_images = self.depth_images[overflow:]
        self.states = self.states[overflow:]
        self.actions = self.actions[overflow:]
        self.rewards = self.rewards[overflow:]
        self.next_depth_images = self.next_depth_images[overflow:]
        self.next_states = self.next_states[overflow:]
        self.dones = self.dones[overflow:]

        # 에피소드 경계 업데이트
        while self.episode_boundaries and self.episode_boundaries[0] < overflow:
            self.episode_boundaries.pop(0)

        # 남은 에피소드 경계 조정
        self.episode_boundaries = [b - overflow for b in self.episode_boundaries]

    def sample(self):
        """시퀀스 배치 샘플링"""
        # 배치 텐서 초기화
        batch_tensors = {
            'depth_images': [],
            'states': [],
            'actions': [],
            'rewards': [],
            'next_depth_images': [],
            'next_states': [],
            'dones': [],
            'masks': []  # 패딩 마스크
        }

        # 에피소드 시작 인덱스 계산
        if not self.episode_boundaries:
            raise ValueError("Buffer is empty or no complete episodes")

        episode_start_indices = [0] + [b + 1 for b in self.episode_boundaries[:-1]]

        # 무작위 에피소드 샘플링
        sampled_episodes = random.choices(range(len(episode_start_indices)), k=self.batch_size)

        for episode_idx in sampled_episodes:
            # 에피소드 경계 결정
            start_idx = episode_start_indices[episode_idx]
            end_idx = self.episode_boundaries[episode_idx]
            episode_length = end_idx - start_idx + 1

            # 시퀀스 선택
            if episode_length <= self.sequence_length:
                # 에피소드가 시퀀스 길이보다 짧음
                seq_start_idx = start_idx
                seq_length = episode_length
            else:
                # 시작점 무작위 선택
                max_start_idx = start_idx + episode_length - self.sequence_length
                seq_start_idx = random.randint(start_idx, max_start_idx)
                seq_length = self.sequence_length

            # 시퀀스 추출
            seq_depth_images = self.depth_images[seq_start_idx:seq_start_idx + seq_length]
            seq_states = self.states[seq_start_idx:seq_start_idx + seq_length]
            seq_actions = self.actions[seq_start_idx:seq_start_idx + seq_length]
            seq_rewards = self.rewards[seq_start_idx:seq_start_idx + seq_length]
            seq_next_depth_images = self.next_depth_images[seq_start_idx:seq_start_idx + seq_length]
            seq_next_states = self.next_states[seq_start_idx:seq_start_idx + seq_length]
            seq_dones = self.dones[seq_start_idx:seq_start_idx + seq_length]

            # 마스크 생성 (실제 데이터는 1, 패딩은 0)
            seq_mask = [1.0] * seq_length

            # 필요시 패딩
            if seq_length < self.sequence_length:
                seq_depth_images, seq_states, seq_actions, seq_rewards, \
                seq_next_depth_images, seq_next_states, seq_dones, seq_mask = \
                    self._pad_sequence(
                        seq_depth_images, seq_states, seq_actions, seq_rewards,
                        seq_next_depth_images, seq_next_states, seq_dones, seq_mask
                    )

            # 배치에 추가
            batch_tensors['depth_images'].append(seq_depth_images)
            batch_tensors['states'].append(seq_states)
            batch_tensors['actions'].append(seq_actions)
            batch_tensors['rewards'].append(seq_rewards)
            batch_tensors['next_depth_images'].append(seq_next_depth_images)
            batch_tensors['next_states'].append(seq_next_states)
            batch_tensors['dones'].append(seq_dones)
            batch_tensors['masks'].append(seq_mask)

        # 텐서로 변환
        processed_tensors = self._process_batch(batch_tensors)

        return (
            processed_tensors['depth_images'],
            processed_tensors['states'],
            processed_tensors['actions'],
            processed_tensors['rewards'],
            processed_tensors['next_depth_images'],
            processed_tensors['next_states'],
            processed_tensors['dones'],
            processed_tensors['masks']
        )

    def _pad_sequence(self, seq_depth_images, seq_states, seq_actions, seq_rewards,
                     seq_next_depth_images, seq_next_states, seq_dones, seq_mask):
        """시퀀스 패딩"""
        pad_length = self.sequence_length - len(seq_depth_images)

        # 패딩 값 생성
        depth_img_shape = seq_depth_images[0].shape
        zero_depth_img = np.zeros(depth_img_shape, dtype=np.float32)

        state_shape = seq_states[0].shape
        zero_state = np.zeros(state_shape, dtype=np.float32)

        action_shape = seq_actions[0].shape
        zero_action = np.zeros(action_shape, dtype=np.float32)

        # 패딩 적용
        seq_depth_images.extend([zero_depth_img] * pad_length)
        seq_states.extend([zero_state] * pad_length)
        seq_actions.extend([zero_action] * pad_length)
        seq_rewards.extend([0.0] * pad_length)
        seq_next_depth_images.extend([zero_depth_img] * pad_length)
        seq_next_states.extend([zero_state] * pad_length)
        seq_dones.extend([1.0] * pad_length)
        seq_mask.extend([0.0] * pad_length)

        return seq_depth_images, seq_states, seq_actions, seq_rewards, \
               seq_next_depth_images, seq_next_states, seq_dones, seq_mask

    def _process_batch(self, batch_tensors):
        """배치 데이터를 텐서로 변환"""
        # 깊이 이미지 처리
        depth_array = self._prepare_depth_array(batch_tensors['depth_images'])
        next_depth_array = self._prepare_depth_array(batch_tensors['next_depth_images'])

        # NumPy 배열로 변환
        states_array = np.array(batch_tensors['states'], dtype=np.float32)
        actions_array = np.array(batch_tensors['actions'], dtype=np.float32)
        rewards_array = np.array(batch_tensors['rewards'], dtype=np.float32)
        next_states_array = np.array(batch_tensors['next_states'], dtype=np.float32)
        dones_array = np.array(batch_tensors['dones'], dtype=np.float32)
        masks_array = np.array(batch_tensors['masks'], dtype=np.float32)

        # 텐서로 변환
        processed = {
            'depth_images': torch.from_numpy(depth_array).to(self.device),
            'states': torch.from_numpy(states_array).to(self.device),
            'actions': torch.from_numpy(actions_array).to(self.device),
            'rewards': torch.from_numpy(rewards_array).unsqueeze(-1).to(self.device),
            'next_depth_images': torch.from_numpy(next_depth_array).to(self.device),
            'next_states': torch.from_numpy(next_states_array).to(self.device),
            'dones': torch.from_numpy(dones_array).unsqueeze(-1).to(self.device),
            'masks': torch.from_numpy(masks_array).unsqueeze(-1).to(self.device)
        }

        return processed

    def _prepare_depth_array(self, depth_images):
        """깊이 이미지를 NumPy 배열로 변환"""
        batch_size = len(depth_images)
        seq_len = len(depth_images[0])
        height, width = depth_images[0][0].shape

        # NumPy 배열 생성
        depth_array = np.zeros((batch_size, seq_len, 1, height, width), dtype=np.float32)

        # 데이터 복사
        for b in range(batch_size):
            for s in range(seq_len):
                depth_array[b, s, 0] = depth_images[b][s]

        return depth_array

    def __len__(self):
        """버퍼 크기 반환"""
        return len(self.states)


#############################################################################
# SAC 에이전트
#############################################################################

class RecurrentSAC:
    """리커런트 SAC 에이전트"""

    def __init__(self, state_dim, action_dim=SACConfig.action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = SACConfig.device

        # 액터 네트워크
        self.actor = RecurrentActor(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=SACConfig.hidden_dim,
            gru_hidden_dim=SACConfig.gru_hidden_dim,
            gru_layers=SACConfig.gru_layers
        ).to(self.device)

        # 크리틱 네트워크 (쌍둥이 Q-네트워크)
        self.critic1 = RecurrentCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=SACConfig.hidden_dim,
            gru_hidden_dim=SACConfig.gru_hidden_dim,
            gru_layers=SACConfig.gru_layers
        ).to(self.device)

        self.critic2 = RecurrentCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=SACConfig.hidden_dim,
            gru_hidden_dim=SACConfig.gru_hidden_dim,
            gru_layers=SACConfig.gru_layers
        ).to(self.device)

        # 타겟 네트워크
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)

        # 타겟 네트워크 고정
        for param in self.critic1_target.parameters():
            param.requires_grad = False

        for param in self.critic2_target.parameters():
            param.requires_grad = False

        # 옵티마이저
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=SACConfig.learning_rate)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=SACConfig.learning_rate)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=SACConfig.learning_rate)

        # 엔트로피 계수 (알파)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = SACConfig.alpha
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=SACConfig.learning_rate)

        # 타겟 엔트로피
        self.target_entropy = -torch.prod(torch.Tensor([action_dim]).to(self.device)).item()

        # 은닉 상태
        self.reset_hidden_states()

        # 리플레이 버퍼
        self.replay_buffer = RecurrentReplayBuffer(
            buffer_size=SACConfig.buffer_size,
            batch_size=SACConfig.batch_size,
            sequence_length=SACConfig.sequence_length,
            device=self.device
        )

        # 학습 스텝 카운터
        self.steps = 0

        # 스레드 안전을 위한 락
        self.train_lock = threading.Lock()

        debug_print(f"RecurrentSAC initialized on device: {self.device}")

    def reset_hidden_states(self, batch_size=1):
        """은닉 상태 초기화"""
        self.actor_hidden = torch.zeros(
            SACConfig.gru_layers,
            batch_size,
            SACConfig.gru_hidden_dim,
            device=self.device
        )

        self.critic1_hidden = torch.zeros(
            SACConfig.gru_layers,
            batch_size,
            SACConfig.gru_hidden_dim,
            device=self.device
        )

        self.critic2_hidden = torch.zeros(
            SACConfig.gru_layers,
            batch_size,
            SACConfig.gru_hidden_dim,
            device=self.device
        )

    def select_action(self, depth_image, state, evaluate=False):
        """액션 선택"""
        with torch.no_grad():
            # 입력 전처리
            depth_tensor = torch.FloatTensor(depth_image).unsqueeze(0).unsqueeze(0).to(self.device)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            if evaluate:
                # 결정적 정책 (평가 모드)
                mean, _, self.actor_hidden = self.actor(depth_tensor, state_tensor, self.actor_hidden)
                action = torch.tanh(mean) * SACConfig.action_scale
                return action.cpu().data.numpy().flatten()
            else:
                # 확률적 정책 (학습 모드)
                action, _, self.actor_hidden = self.actor.sample(depth_tensor, state_tensor, self.actor_hidden)
                return action.cpu().data.numpy().flatten()

    def store_transition(self, depth_image, state, action, reward, next_depth_image, next_state, done):
        """트랜지션 저장"""
        self.replay_buffer.add(depth_image, state, action, reward, next_depth_image, next_state, done)

    def train(self):
        """정책 및 가치 함수 업데이트"""
        self.steps += 1

        # 지정된 주기에만 업데이트
        if self.steps % SACConfig.update_every != 0:
            return 0, 0, 0

        # 버퍼에 충분한 데이터 확인
        if len(self.replay_buffer) < SACConfig.batch_size:
            return 0, 0, 0

        # 스레드 안전을 위한 락 사용
        with self.train_lock:
            try:
                # 배치 샘플링
                depth_imgs, states, actions, rewards, next_depth_imgs, next_states, dones, masks = self.replay_buffer.sample()

                # 크리틱 업데이트
                critic_loss = self._update_critics(depth_imgs, states, actions, rewards, next_depth_imgs, next_states, dones, masks)

                # 액터 및 알파 업데이트
                actor_loss, alpha_loss = self._update_actor_and_alpha(depth_imgs, states, masks)

                # 타겟 네트워크 소프트 업데이트
                self._soft_update_targets()

                return critic_loss, actor_loss, alpha_loss

            except Exception as e:
                # 예외 발생 시 디버그 정보 출력
                print("\nExcepion caught during training:")
                print(traceback.format_exc())
                return 0, 0, 0

    def _update_critics(self, depth_imgs, states, actions, rewards, next_depth_imgs, next_states, dones, masks):
        """크리틱 네트워크 업데이트"""
        batch_size = states.shape[0]

        with torch.no_grad():
            # 타겟 액션 및 로그 확률 샘플링
            next_actions, next_log_probs, _ = self._get_next_actions_and_log_probs(next_depth_imgs, next_states,
                                                                                   batch_size)

            # 타겟 Q-값 계산
            target_q1, target_q2 = self._get_target_q_values(next_depth_imgs, next_states, next_actions, batch_size)

            # 두 Q-값 중 최소값 사용
            target_q_values = torch.min(target_q1, target_q2)

            # 엔트로피 항 계산
            entropy_term = self.alpha * next_log_probs

            # target_q_values의 차원을 3차원으로 맞춤
            if target_q_values.dim() == 2:
                target_q_values = target_q_values.unsqueeze(-1)  # [64, 40] -> [64, 40, 1]

            # entropy_term의 차원을 맞춤
            if entropy_term.dim() != target_q_values.dim():
                if entropy_term.dim() > target_q_values.dim():
                    # 차원 감소 (예: [64, 40, 1] -> [64, 40])
                    entropy_term = entropy_term.squeeze(-1)
                else:
                    # 차원 증가 (예: [64, 40] -> [64, 40, 1])
                    entropy_term = entropy_term.unsqueeze(-1)

            # TD 타겟 계산
            targets = rewards + (1 - dones) * SACConfig.gamma * (target_q_values - entropy_term)

        # 현재 Q-값 계산
        current_q1, current_q2 = self._get_current_q_values(depth_imgs, states, actions, batch_size)

        # 크리틱 손실 계산
        critic1_loss = F.mse_loss(current_q1 * masks, targets * masks)
        critic2_loss = F.mse_loss(current_q2 * masks, targets * masks)
        critic_loss = critic1_loss + critic2_loss

        # 크리틱 네트워크 업데이트
        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()

        return critic_loss.item()

    def _get_next_actions_and_log_probs(self, next_depth_imgs, next_states, batch_size):
        """다음 상태에 대한 액션 및 로그 확률 계산"""
        # 배치 모드 초기화
        actor_hidden = torch.zeros(
            SACConfig.gru_layers, batch_size, SACConfig.gru_hidden_dim, device=self.device)

        # 액션 및 로그 확률 샘플링
        next_actions, next_log_probs, _ = self.actor.sample(next_depth_imgs, next_states, actor_hidden)

        return next_actions, next_log_probs, actor_hidden

    def _get_target_q_values(self, next_depth_imgs, next_states, next_actions, batch_size):
        """타겟 Q-값 계산"""
        # 배치 모드 초기화
        critic1_target_hidden = torch.zeros(
            SACConfig.gru_layers, batch_size, SACConfig.gru_hidden_dim, device=self.device)
        critic2_target_hidden = torch.zeros(
            SACConfig.gru_layers, batch_size, SACConfig.gru_hidden_dim, device=self.device)

        # 타겟 Q-값 계산
        target_q1, _ = self.critic1_target(next_depth_imgs, next_states, next_actions, critic1_target_hidden)
        target_q2, _ = self.critic2_target(next_depth_imgs, next_states, next_actions, critic2_target_hidden)

        return target_q1, target_q2

    def _get_current_q_values(self, depth_imgs, states, actions, batch_size):
        """현재 Q-값 계산"""
        # 배치 모드 초기화
        critic1_hidden = torch.zeros(
            SACConfig.gru_layers, batch_size, SACConfig.gru_hidden_dim, device=self.device)
        critic2_hidden = torch.zeros(
            SACConfig.gru_layers, batch_size, SACConfig.gru_hidden_dim, device=self.device)

        # 현재 Q-값 계산
        current_q1, _ = self.critic1(depth_imgs, states, actions, critic1_hidden)
        current_q2, _ = self.critic2(depth_imgs, states, actions, critic2_hidden)

        return current_q1, current_q2

    def _update_actor_and_alpha(self, depth_imgs, states, masks):
        """액터 네트워크 및 엔트로피 계수 업데이트"""
        batch_size = states.shape[0]

        # 배치 모드 초기화
        actor_hidden = torch.zeros(
            SACConfig.gru_layers, batch_size, SACConfig.gru_hidden_dim, device=self.device)
        critic1_hidden = torch.zeros(
            SACConfig.gru_layers, batch_size, SACConfig.gru_hidden_dim, device=self.device)
        critic2_hidden = torch.zeros(
            SACConfig.gru_layers, batch_size, SACConfig.gru_hidden_dim, device=self.device)

        # 액션 및 로그 확률 샘플링
        sampled_actions, log_probs, _ = self.actor.sample(depth_imgs, states, actor_hidden)

        # Q-값 계산
        q1, _ = self.critic1(depth_imgs, states, sampled_actions, critic1_hidden)
        q2, _ = self.critic2(depth_imgs, states, sampled_actions, critic2_hidden)
        min_q = torch.min(q1, q2)

        # 액터 손실 계산 (정책 그래디언트)
        actor_loss = ((self.alpha * log_probs) - min_q) * masks
        actor_loss = actor_loss.mean()

        # 액터 업데이트
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 알파 업데이트
        alpha_loss = -self.log_alpha * ((log_probs + self.target_entropy).detach() * masks)
        alpha_loss = alpha_loss.mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # 알파 값 업데이트
        self.alpha = self.log_alpha.exp().item()

        return actor_loss.item(), alpha_loss.item()

    def _soft_update_targets(self):
        """타겟 네트워크 소프트 업데이트"""
        tau = SACConfig.tau

        # 크리틱1 타겟 업데이트
        for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        # 크리틱2 타겟 업데이트
        for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save(self, directory, filename="rsac"):
        """모델 저장"""
        if not os.path.exists(directory):
            os.makedirs(directory)

        file_path = os.path.join(directory, f"{filename}.pt")

        # 모델 상태 저장
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'critic1_target': self.critic1_target.state_dict(),
            'critic2_target': self.critic2_target.state_dict(),
            'log_alpha': self.log_alpha,
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic1_optimizer': self.critic1_optimizer.state_dict(),
            'critic2_optimizer': self.critic2_optimizer.state_dict(),
            'alpha_optimizer': self.alpha_optimizer.state_dict()
        }, file_path)

        debug_print(f"Model saved to {file_path}")
        return file_path

    def load(self, filepath):
        """모델 로드"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        # GPU/CPU 호환성 처리
        map_location = self.device

        # 체크포인트 로드
        checkpoint = torch.load(filepath, map_location=map_location)

        # 모델 상태 복원
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.critic1_target.load_state_dict(checkpoint['critic1_target'])
        self.critic2_target.load_state_dict(checkpoint['critic2_target'])

        # 알파 복원 (그래디언트 정보 유지)
        self.log_alpha.data.copy_(checkpoint['log_alpha'].data)
        self.alpha = self.log_alpha.exp().item()

        # 옵티마이저 상태 복원
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer'])
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])

        debug_print(f"Model loaded from {filepath}")


#############################################################################
# 로깅 및 학습 시스템
#############################################################################

class Logger:
    """학습 로깅 클래스"""

    def __init__(self, log_dir="logs", window_size=100):
        self.log_dir = log_dir
        self.window_size = window_size
        self.episode_counter = 0

        # 통계 추적
        self.rewards = deque(maxlen=window_size)
        self.steps = deque(maxlen=window_size)
        self.statuses = deque(maxlen=window_size)

        # 로그 파일 준비
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, "training_log.csv")
        self.summary_file = os.path.join(log_dir, "training_summary.txt")

        # CSV 헤더 작성
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Episode", "Reward", "Steps", "Status",
                "SuccessRate", "CollisionRate", "AvgSteps"
            ])

    def log_episode(self, reward, steps, status):
        """에피소드 정보 로깅"""
        self.episode_counter += 1
        self.rewards.append(reward)
        self.steps.append(steps)
        self.statuses.append(status)

        # 통계 계산
        success_rate = self._calculate_success_rate()
        collision_rate = self._calculate_collision_rate()
        avg_steps = np.mean(self.steps) if self.steps else 0

        # CSV 로그 기록
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.episode_counter,
                round(reward, 2),
                steps,
                status,
                round(success_rate * 100, 2),
                round(collision_rate * 100, 2),
                round(avg_steps, 2)
            ])

        # 주기적 요약 정보 작성
        if self.episode_counter % SACConfig.log_interval == 0:
            self._write_summary()

        return self.episode_counter

    def _calculate_success_rate(self):
        """성공률 계산"""
        if not self.statuses:
            return 0
        return sum(1 for s in self.statuses if s == "goal_reached") / len(self.statuses)

    def _calculate_collision_rate(self):
        """충돌률 계산"""
        if not self.statuses:
            return 0
        return sum(1 for s in self.statuses if s == "collision") / len(self.statuses)

    def _write_summary(self):
        """요약 정보 파일 작성"""
        with open(self.summary_file, 'w') as f:
            f.write(f"총 에피소드: {self.episode_counter}\n")
            f.write(f"최근 {len(self.rewards)}개 에피소드 통계:\n")
            f.write(f"  평균 보상: {np.mean(self.rewards):.2f}\n")
            f.write(f"  평균 스텝: {np.mean(self.steps):.2f}\n")
            f.write(f"  성공률: {self._calculate_success_rate()*100:.2f}%\n")
            f.write(f"  충돌률: {self._calculate_collision_rate()*100:.2f}%\n")


class BackgroundLearner(threading.Thread):
    """백그라운드 학습 스레드"""

    def __init__(self, agent, stop_event):
        threading.Thread.__init__(self)
        self.agent = agent
        self.stop_event = stop_event
        self.daemon = True  # 메인 스레드 종료 시 자동 종료
        self.error = None  # 에러 저장용

        # 학습 통계
        self.training_steps = 0
        self.last_losses = {'critic': 0, 'actor': 0, 'alpha': 0}

    def run(self):
        """스레드 실행 메서드"""
        try:
            while not self.stop_event.is_set():
                # 버퍼에 충분한 데이터가 있으면 학습
                if len(self.agent.replay_buffer) >= SACConfig.batch_size:
                    critic_loss, actor_loss, alpha_loss = self.agent.train()

                    # 학습 스텝 증가
                    if critic_loss + actor_loss + alpha_loss > 0:
                        self.training_steps += 1
                        self.last_losses = {
                            'critic': critic_loss,
                            'actor': actor_loss,
                            'alpha': alpha_loss
                        }

                    # CPU 사용률 조절
                    if self.training_steps % 10 == 0:
                        time.sleep(0.001)
                else:
                    # 데이터 부족 시 대기
                    time.sleep(0.1)

        except Exception as e:
            # 예외 발생 시 기록
            self.error = traceback.format_exc()
            print(f"\n백그라운드 학습 스레드 오류:\n{self.error}")

            # 오류 발생 후에도 스레드 유지 (중단 알림 위해)
            while not self.stop_event.is_set():
                time.sleep(1.0)


def train_sequential(env, agent, num_episodes=SACConfig.num_episodes,
                    log_dir="logs", save_dir="models", render=False):
    """순차적 학습 함수"""
    # 로거 초기화
    logger = Logger(log_dir=log_dir)

    # 백그라운드 학습기 시작
    stop_event = threading.Event()
    learner = BackgroundLearner(agent, stop_event)
    learner.start()

    try:
        # 에피소드 기반 학습
        start_time = time.time()

        for episode in range(1, num_episodes + 1):
            # 백그라운드 학습기 오류 확인
            if learner.error is not None:
                print("\n백그라운드 학습기에 오류가 발생했습니다. 학습을 중단합니다.")
                print(f"오류 내용: {learner.error}")
                break

            # 환경 및 에이전트 초기화
            depth_image, state = env.reset()
            agent.reset_hidden_states()

            episode_reward = 0
            steps = 0
            done = False
            status = "timeout"  # 기본값

            # 에피소드 실행
            while not done:
                # 액션 선택
                action = agent.select_action(depth_image, state)

                # 환경에서 스텝 실행
                next_depth_image, next_state, reward, done, info = env.step(action)

                # 렌더링 (요청된 경우)
                if render:
                    env.visualize_3d_lidar(depth_image, frame_count=steps, show=True)

                # 경험 저장
                agent.store_transition(depth_image, state, action, reward, next_depth_image, next_state, done)

                # 상태 및 리워드 업데이트
                depth_image = next_depth_image
                state = next_state
                episode_reward += reward
                steps += 1

                # 상태 확인
                if done and 'status' in info:
                    status = info['status']

            # 에피소드 종료 로깅
            episode_num = logger.log_episode(episode_reward, steps, status)

            # 학습 스텝 정보
            train_steps = learner.training_steps
            losses = learner.last_losses

            # 진행 상황 출력
            elapsed = int(time.time() - start_time)
            print(f"\r에피소드 {episode}/{num_episodes} " +
                 f"({episode/num_episodes*100:.1f}%) " +
                 f"보상: {episode_reward:.2f}, 스텝: {steps}, 상태: {status}, " +
                 f"학습 스텝: {train_steps}, " +
                 f"경과: {elapsed//60:02d}:{elapsed%60:02d}", end="")

            # 주기적으로 모델 저장
            if episode % SACConfig.save_interval == 0:
                save_path = agent.save(save_dir, f"rsac_episode_{episode}")
                print(f"\n체크포인트 저장됨: {save_path}")

    finally:
        # 백그라운드 학습기 종료
        stop_event.set()
        learner.join(timeout=2)  # 최대 2초간 종료 대기

        # 최종 모델 저장
        save_path = agent.save(save_dir, "rsac_final")
        print(f"\n최종 모델 저장됨: {save_path}")

    print(f"\n학습 완료! 로그는 {log_dir} 디렉토리에 저장되었습니다.")
    return agent


def evaluate_agent(env, agent, num_episodes=5, render=False):
    """에이전트 평가 함수"""
    total_rewards = []
    success_count = 0
    collision_count = 0
    step_counts = []

    for episode in range(num_episodes):
        # 환경 및 에이전트 초기화
        depth_image, state = env.reset()
        agent.reset_hidden_states()

        episode_reward = 0
        steps = 0
        done = False
        status = "timeout"

        while not done:
            # 결정적 액션 선택 (평가 모드)
            action = agent.select_action(depth_image, state, evaluate=True)

            # 환경에서 스텝 실행
            next_depth_image, next_state, reward, done, info = env.step(action)

            # 렌더링 (요청된 경우)
            if render:
                env.visualize_3d_lidar(depth_image, frame_count=steps, show=True)

            # 상태 및 리워드 업데이트
            depth_image = next_depth_image
            state = next_state
            episode_reward += reward
            steps += 1

            # 종료 상태 확인
            if done and 'status' in info:
                status = info['status']

        # 결과 통계
        total_rewards.append(episode_reward)
        step_counts.append(steps)
        if status == "goal_reached":
            success_count += 1
        elif status == "collision":
            collision_count += 1

        print(f"평가 에피소드 {episode+1}/{num_episodes}: " +
             f"보상 = {episode_reward:.2f}, 스텝 = {steps}, 결과 = {status}")

    # 평균 통계 계산
    avg_reward = sum(total_rewards) / num_episodes
    avg_steps = sum(step_counts) / num_episodes
    success_rate = success_count / num_episodes * 100
    collision_rate = collision_count / num_episodes * 100

    print(f"\n--- 평가 결과 (에피소드 {num_episodes}개) ---")
    print(f"평균 보상: {avg_reward:.2f}")
    print(f"평균 스텝: {avg_steps:.2f}")
    print(f"성공률: {success_rate:.1f}%")
    print(f"충돌률: {collision_rate:.1f}%")

    return {
        'avg_reward': avg_reward,
        'avg_steps': avg_steps,
        'success_rate': success_rate,
        'collision_rate': collision_rate
    }


def plot_training_results(log_file, save_path="results/training_plots.png"):
    """학습 로그 시각화"""
    # 로그 데이터 읽기
    episodes = []
    rewards = []
    steps = []
    success_rates = []
    collision_rates = []

    with open(log_file, 'r') as f:
        header = f.readline().strip().split(',')
        for line in f:
            data = line.strip().split(',')
            if len(data) >= 7:
                episodes.append(int(data[0]))
                rewards.append(float(data[1]))
                steps.append(int(data[2]))
                success_rates.append(float(data[4]))
                collision_rates.append(float(data[5]))

    # 저장 디렉토리 생성
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 그래프 그리기
    plt.figure(figsize=(15, 10))

    # 보상 그래프
    plt.subplot(2, 2, 1)
    plt.plot(episodes, rewards)
    plt.title('보상')
    plt.xlabel('에피소드')
    plt.ylabel('에피소드 보상')
    plt.grid(True)

    # 스텝 그래프
    plt.subplot(2, 2, 2)
    plt.plot(episodes, steps)
    plt.title('스텝 수')
    plt.xlabel('에피소드')
    plt.ylabel('에피소드 스텝')
    plt.grid(True)

    # 성공률/충돌률 그래프
    plt.subplot(2, 2, 3)
    plt.plot(episodes, success_rates, 'g-', label='성공률')
    plt.plot(episodes, collision_rates, 'r-', label='충돌률')
    plt.title('성공률 및 충돌률')
    plt.xlabel('에피소드')
    plt.ylabel('비율 (%)')
    plt.legend()
    plt.grid(True)

    # 이동 평균 보상 그래프
    window_size = min(20, len(rewards))
    if window_size > 0:
        moving_avg = []
        for i in range(len(rewards) - window_size + 1):
            moving_avg.append(sum(rewards[i:i+window_size]) / window_size)

        plt.subplot(2, 2, 4)
        plt.plot(episodes[window_size-1:], moving_avg)
        plt.title(f'이동 평균 보상 (윈도우={window_size})')
        plt.xlabel('에피소드')
        plt.ylabel('평균 보상')
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"학습 그래프가 저장됨: {save_path}")


#############################################################################
# 메인 함수
#############################################################################

def main():
    """메인 함수"""
    # 명령줄 인자 파싱
    parser = argparse.ArgumentParser(description="Recurrent SAC 학습")

    # 학습 설정
    parser.add_argument("--episodes", type=int, default=SACConfig.num_episodes, help="학습 에피소드 수")
    parser.add_argument("--log_dir", type=str, default="logs", help="로그 디렉토리")
    parser.add_argument("--save_dir", type=str, default="models", help="모델 저장 디렉토리")
    parser.add_argument("--debug", action="store_true", help="디버깅 출력 활성화")

    # 모델 로드 및 평가
    parser.add_argument("--load_model", type=str, default="./models/rsac_episode_650.pt", help="로드할 모델 경로")
    parser.add_argument("--eval_only", action="store_true", help="평가만 수행 (학습 안 함)")
    parser.add_argument("--eval_episodes", type=int, default=10, help="평가 에피소드 수")
    parser.add_argument("--render", action="store_true", help="평가 중 시각화 활성화")

    args = parser.parse_args()

    # 디버깅 설정
    SACConfig.debug_prints = args.debug

    # 디렉토리 생성
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)

    # 환경 생성
    try:
        from Environment.Env import DroneEnv
        env = DroneEnv()
    except ImportError:
        print("Error: Cannot import DroneEnv. Make sure Env.py is in the current directory.")
        return

    # 상태 차원 확인
    _, state = env.reset()
    state_dim = len(state)
    print(f"환경 초기화 완료: 상태 차원 = {state_dim}")

    # 에이전트 생성
    agent = RecurrentSAC(state_dim=state_dim)

    # 기존 모델 로드 (있는 경우)
    if args.load_model:
        print(f"모델 로드 중: {args.load_model}")
        try:
            agent.load(args.load_model)
        except Exception as e:
            print(f"모델 로드 실패: {e}")
            if args.eval_only:
                return

    if not args.eval_only:
        # 학습 실행
        print(f"학습 시작: {args.episodes}개의 에피소드")
        agent = train_sequential(
            env=env,
            agent=agent,
            num_episodes=args.episodes,
            log_dir=args.log_dir,
            save_dir=args.save_dir,
            render=args.render
        )

        # 학습 결과 그래프 생성
        log_file = os.path.join(args.log_dir, "training_log.csv")
        plot_training_results(log_file, os.path.join(args.log_dir, "training_plots.png"))

    # 모델 평가 (항상 실행)
    print("\n모델 평가 시작...")
    evaluate_agent(env, agent, num_episodes=args.eval_episodes, render=args.render)


if __name__ == "__main__":
    main()