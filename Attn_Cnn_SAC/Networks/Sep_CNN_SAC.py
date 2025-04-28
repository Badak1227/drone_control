import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import random
from collections import deque


class ConvAttentionBlock(nn.Module):
    """
    컨볼루션 셀프-어텐션 블록:
      - 입력 X를 1x1 컨볼루션을 통해 Q, K, V로 투영
      - 어텐션 맵 A = softmax(Q * K^T / sqrt(d_k))을 계산
      - A를 V에 적용하여 O를 얻은 다음 입력 X에 잔차 연결
    """

    def __init__(self, in_channels, inter_channels=None):
        super(ConvAttentionBlock, self).__init__()
        # 기본적으로 Q/K/V 채널 수 감소
        self.inter_channels = inter_channels or in_channels // 4

        # Query, Key, Value 투영을 위한 1x1 컨볼루션
        self.conv_q = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.conv_k = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.conv_v = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)

        # 잔차 연결을 위한 학습 가능한 스케일링 요소
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()

        # 입력을 Q, K, V로 투영
        q = self.conv_q(x)
        k = self.conv_k(x)
        v = self.conv_v(x)

        # 공간 차원 평탄화: (B, C', H*W)
        q = q.view(B, self.inter_channels, -1).permute(0, 2, 1)  # (B, N, C')
        k = k.view(B, self.inter_channels, -1)  # (B, C', N)
        v = v.view(B, self.inter_channels, -1)  # (B, C', N)

        # 어텐션 맵 계산 (스케일링 추가됨)
        attn = torch.bmm(q, k)  # (B, N, N)
        attn = attn / torch.sqrt(torch.tensor(self.inter_channels, dtype=torch.float32))  # 스케일링 요소 추가
        attn = F.softmax(attn, dim=-1)

        # 어텐션에 의한 V의 가중합
        out = torch.bmm(v, attn.permute(0, 2, 1))  # (B, C', N)
        out = out.view(B, self.inter_channels, H, W)

        # 잔차 연결
        return self.gamma * out + x


class DualReplayBuffer:
    """
    긍정적인 전이와 부정적인 전이를 별도로 저장하는 버퍼.
    """

    def __init__(self, capacity_pos, capacity_neg):
        self.pos = []
        self.neg = []
        self.cap_pos = capacity_pos
        self.cap_neg = capacity_neg

    def push(self, state, action, reward, next_state, done):
        buf = self.pos if reward >= 0 else self.neg
        buf.append((state, action, reward, next_state, done))

        # 용량 유지
        if reward >= 0 and len(self.pos) > self.cap_pos:
            self.pos.pop(0)
        elif reward < 0 and len(self.neg) > self.cap_neg:
            self.neg.pop(0)

    def sample(self, batch_size, pos_frac=0.6):
        # 각 버퍼의 크기를 확인하고 가능한 배치 크기 조정
        available_pos = min(int(batch_size * pos_frac), len(self.pos))
        available_neg = min(batch_size - available_pos, len(self.neg))

        # 사용 가능한 샘플 수가 충분한지 확인
        if available_pos == 0 or available_neg == 0:
            if len(self.pos) > 0 and len(self.neg) == 0:
                # 부정적 샘플이 없으면 모두 긍정적 샘플 사용
                available_pos = min(batch_size, len(self.pos))
                available_neg = 0
            elif len(self.pos) == 0 and len(self.neg) > 0:
                # 긍정적 샘플이 없으면 모두 부정적 샘플 사용
                available_pos = 0
                available_neg = min(batch_size, len(self.neg))
            else:
                # 둘 다 비어 있으면 샘플링할 수 없음
                raise ValueError("Cannot sample from empty buffer")

        pos_samples = random.sample(self.pos, available_pos) if available_pos > 0 else []
        neg_samples = random.sample(self.neg, available_neg) if available_neg > 0 else []

        batch = pos_samples + neg_samples
        random.shuffle(batch)

        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.stack(states),
            torch.stack(actions),
            torch.tensor(rewards).float(),
            torch.stack(next_states),
            torch.tensor(dones).float()
        )

    def __len__(self):
        return len(self.pos) + len(self.neg)


# Actor 네트워크
class DACActor(nn.Module):
    """
    Actor 네트워크: 정책을 위한 평균과 로그 표준편차를 출력.
    독립적인 백본 구조를 사용함.
    """

    def __init__(self, in_channels=1, base_channels=32,
                 feature_dim=256, action_dim=3):
        super(DACActor, self).__init__()

        # 액터 전용 백본
        # 스테이지 1: Conv -> ReLU -> MaxPool
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )

        # 스테이지 1의 셀프-어텐션
        self.attn1 = ConvAttentionBlock(base_channels)

        # 스테이지 2: Conv -> ReLU -> MaxPool
        self.stage2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )

        # 스테이지 2의 셀프-어텐션
        self.attn2 = ConvAttentionBlock(base_channels * 2)

        # 스테이지 3: Conv -> ReLU
        self.stage3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten()
        )

        # 특징에서 액션 분포로 변환하는 헤드
        self.fc = nn.Sequential(
            nn.Linear(base_channels * 4, feature_dim),
            nn.ReLU(inplace=True)
        )
        self.mu_head = nn.Linear(feature_dim, action_dim)
        self.logstd_head = nn.Linear(feature_dim, action_dim)

    def forward(self, x):
        # 백본을 통한 특징 추출
        x = self.stage1(x)
        x = self.attn1(x)  # 첫 번째 셀프-어텐션
        x = self.stage2(x)
        x = self.attn2(x)  # 두 번째 셀프-어텐션
        feat = self.stage3(x)

        # 특징에서 액션 분포 계산
        feat = self.fc(feat)
        mu = self.mu_head(feat)
        logstd = self.logstd_head(feat)
        logstd = torch.clamp(logstd, min=-20, max=2)  # 안정성을 위한 클램핑
        return mu, logstd

# Critic 네트워크
class DACCritic(nn.Module):
    """
    Critic 네트워크: 상태와 행동으로부터 두 개의 Q-값을 추정.
    독립적인 백본 구조를 사용함.
    """

    def __init__(self, in_channels=1, base_channels=32,
                 feature_dim=256, action_dim=3):
        super(DACCritic, self).__init__()

        # 크리틱 전용 백본
        # 스테이지 1: Conv -> ReLU -> MaxPool -> Attention
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            ConvAttentionBlock(base_channels)
        )

        # 스테이지 2: Conv -> ReLU -> MaxPool -> Attention
        self.stage2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            ConvAttentionBlock(base_channels * 2)
        )

        # 스테이지 3: Conv -> ReLU -> Attention
        self.stage3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            ConvAttentionBlock(base_channels * 4),
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten()
        )

        # 특징과 액션을 결합하여 Q값 추정
        in_size = base_channels * 4 + action_dim

        self.q1_net = nn.Sequential(
            nn.Linear(in_size, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, 1)
        )

        self.q2_net = nn.Sequential(
            nn.Linear(in_size, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, 1)
        )

    def forward(self, x, action):
        # 백본을 통한 특징 추출
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        feat = self.stage4(x)

        # 특징과 액션 결합
        xq = torch.cat([feat, action], dim=1)

        # 두 개의 Q값 계산
        q1 = self.q1_net(xq)
        q2 = self.q2_net(xq)
        return q1, q2


class DACSACAgent:
    """
    이중 경험 버퍼와 컨볼루션 셀프-어텐션을 갖춘 Soft Actor-Critic 에이전트.
    """

    def __init__(self, state_ch, action_dim, device,
                 alpha=0.2, auto_alpha=True, target_entropy=None,
                 buffer_pos_size=100000, buffer_neg_size=100000,
                 lr=1e-4, gamma=0.99, tau=0.005):
        self.device = device
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau

        # 알파 (온도 매개변수) 초기화
        self.log_alpha = torch.zeros(1, requires_grad=auto_alpha, device=device)
        self.alpha = alpha if not auto_alpha else self.log_alpha.exp().item()
        self.auto_alpha = auto_alpha
        self.target_entropy = -action_dim if target_entropy is None else target_entropy

        # 네트워크 - 각각 별도의 백본 사용
        self.actor = DACActor(state_ch, 32, 256, action_dim).to(device)
        self.critic = DACCritic(state_ch, 32, 256, action_dim).to(device)
        self.target_critic = copy.deepcopy(self.critic)

        # Replay 버퍼
        self.buffer = DualReplayBuffer(buffer_pos_size, buffer_neg_size)

        # 옵티마이저
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=lr)

        if auto_alpha:
            self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=lr)

        # 지연 학습을 위한 변수
        self.delayed_learning = True
        self.episode_transitions = []

    def select_action(self, state, evaluate=False):
        with torch.no_grad():
            mu, logstd = self.actor(state)
            if evaluate:
                return mu
            std = logstd.exp()
            dist = torch.distributions.Normal(mu, std)
            action = dist.rsample()
            action = torch.tanh(action)  # 액션 범위를 [-1, 1]로 제한
            return action

    def calc_action_with_log_prob(self, state):
        mu, logstd = self.actor(state)
        std = logstd.exp()
        dist = torch.distributions.Normal(mu, std)
        x = dist.rsample()  # 재매개변수화 트릭

        # tanh 변환 이후의 로그 확률 계산
        action = torch.tanh(x)
        # tanh 변환에 대한 로그 결정자
        log_det = torch.log(1 - action.pow(2) + 1e-6)
        log_prob = dist.log_prob(x) - log_det
        log_prob = log_prob.sum(-1, keepdim=True)

        return action, log_prob

    def store_transition(self, state, action, reward, next_state, done):
        """트랜지션을 이중 경험 버퍼에 저장"""
        self.buffer.push(state, action, reward, next_state, done)

        # 지연 학습을 위해 에피소드 전이 저장
        if self.delayed_learning:
            self.episode_transitions.append((state, action, reward, next_state, done))

    def end_episode(self):
        """에피소드 종료 시 학습 수행 (지연 학습)"""
        if self.delayed_learning and len(self.episode_transitions) > 0:
            # 에피소드 데이터 배치로 학습
            self.update_from_batch(self.episode_transitions)
            self.episode_transitions = []  # 에피소드 전이 초기화

    def update_from_batch(self, transitions, batch_size=32):
        """주어진 전이 배치로부터 네트워크 업데이트"""
        # 배치 크기가 전이 수보다 크면 모든 전이 사용
        actual_batch_size = min(batch_size, len(transitions))

        for i in range(0, len(transitions), actual_batch_size):
            batch = transitions[i:i + actual_batch_size]
            if len(batch) < actual_batch_size:
                continue

            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.stack(states).to(self.device)
            actions = torch.stack(actions).to(self.device)
            rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
            next_states = torch.stack(next_states).to(self.device)
            dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

            self.update(states, actions, rewards.unsqueeze(1), next_states, dones.unsqueeze(1))

    def update(self, states, actions, rewards, next_states, dones, pos_frac=0.6):
        """네트워크 업데이트 논리"""
        # 표준 배치로부터 업데이트
        if not isinstance(states, torch.Tensor):
            states, actions, rewards, next_states, dones = self.buffer.sample(batch_size, pos_frac)
            states, actions, rewards, next_states, dones = [x.to(self.device) for x in
                                                            (states, actions, rewards, next_states, dones)]

        with torch.no_grad():
            next_actions, next_log_probs = self.calc_action_with_log_prob(next_states)
            q1_next, q2_next = self.target_critic(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            q_target = rewards + (1 - dones) * self.gamma * q_next

        # Critic 업데이트
        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.opt_c.zero_grad()
        critic_loss.backward()
        # 그라디언트 클리핑 추가
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.opt_c.step()

        # Actor 업데이트
        actions_pi, log_probs = self.calc_action_with_log_prob(states)
        q1_pi, q2_pi = self.critic(states, actions_pi)
        q_pi = torch.min(q1_pi, q2_pi)

        actor_loss = (self.alpha * log_probs - q_pi).mean()

        self.opt_a.zero_grad()
        actor_loss.backward()
        # 그라디언트 클리핑 추가
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.opt_a.step()

        # 알파 업데이트 (자동 엔트로피 조정)
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()

            self.opt_alpha.zero_grad()
            alpha_loss.backward()
            self.opt_alpha.step()

            self.alpha = self.log_alpha.exp().item()

        # Target critic 소프트 업데이트
        for p, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
            tp.data.mul_(1 - self.tau)
            tp.data.add_(self.tau * p.data)

        return critic_loss.item(), actor_loss.item()




