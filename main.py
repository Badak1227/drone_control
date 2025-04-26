import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import random


# Configuration parameters
class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr_actor = 3e-4
    lr_critic = 3e-4
    lr_alpha = 3e-4
    gamma = 0.99
    tau = 0.005
    alpha = 0.2
    batch_size = 64
    burn_in = 16
    seq_length = 32
    action_dim = 2  # [angular_velocity, linear_velocity]
    max_action = 1.0
    target_entropy = -2  # -action_dim


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
        probs = prios / prios.sum()

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
            # errors: List of TD-errors of sequence length
            max_err = max(abs(e) for e in errors)
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
        dist = Normal(mean, std)
        z = dist.rsample()
        action = torch.tanh(z) * self.max_action
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(-1, keepdim=True)

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


# --- Layer 1: Obstacle Avoidance Module ---
class ObstacleAvoidanceModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = CNNFeatureExtractor()
        self.lstm = nn.LSTM(512, 256, batch_first=True)  # Changed from GRU to LSTM as per paper
        self.actor = PolicyNetwork(256, Config.action_dim, Config.max_action)
        self.q1 = QNetwork(256, Config.action_dim)
        self.q2 = QNetwork(256, Config.action_dim)

    def forward(self, obs_seq, h):
        """
        obs_seq: [B, L, C, H, W] batch of observation sequences
        h: hidden state tuple (h, c) for LSTM
        """
        B, L, C, H, W = obs_seq.shape
        x = obs_seq.view(B * L, C, H, W)
        feats = self.cnn(x).view(B, L, 512)
        out_seq, h_new = self.lstm(feats, h)
        last = out_seq[:, -1]
        a, logp = self.actor.sample(last)
        q1 = self.q1(last, a)
        q2 = self.q2(last, a)
        return out_seq, h_new, a, logp, q1, q2


# --- Layer 2: Navigation Module ---
class NavigationModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = CNNFeatureExtractor()
        self.lstm = nn.LSTM(512 + 256, 256, batch_first=True)  # Changed from GRU to LSTM
        self.actor = PolicyNetwork(256, Config.action_dim, Config.max_action)
        self.q1 = QNetwork(256, Config.action_dim)
        self.q2 = QNetwork(256, Config.action_dim)

    def forward(self, obs_seq, evade_out, h):
        """
        obs_seq: [B, L, C, H, W] batch of observation sequences
        evade_out: output from evade network [B, L, 256]
        h: hidden state tuple (h, c) for LSTM
        """
        B, L, C, H, W = obs_seq.shape
        x = obs_seq.view(B * L, C, H, W)
        feats = self.cnn(x).view(B, L, 512)
        combined = torch.cat([feats, evade_out], dim=-1)
        out_seq, h_new = self.lstm(combined, h)
        last = out_seq[:, -1]
        a, logp = self.actor.sample(last)
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
class LayeredRSACAgent:
    def __init__(self):
        self.layer1 = ObstacleAvoidanceModule().to(Config.device)
        self.layer2 = NavigationModule().to(Config.device)
        self.integrated = IntegratedNetwork().to(Config.device)

        # Target networks
        self.target_q1_1 = QNetwork(256, Config.action_dim).to(Config.device)
        self.target_q1_2 = QNetwork(256, Config.action_dim).to(Config.device)
        self.target_q2_1 = QNetwork(256, Config.action_dim).to(Config.device)
        self.target_q2_2 = QNetwork(256, Config.action_dim).to(Config.device)

        # Copy weights
        for t, p in zip(self.target_q1_1.parameters(), self.layer1.q1.parameters()): t.data.copy_(p.data)
        for t, p in zip(self.target_q1_2.parameters(), self.layer1.q2.parameters()): t.data.copy_(p.data)
        for t, p in zip(self.target_q2_1.parameters(), self.layer2.q1.parameters()): t.data.copy_(p.data)
        for t, p in zip(self.target_q2_2.parameters(), self.layer2.q2.parameters()): t.data.copy_(p.data)

        # Optimizers
        self.opt1 = optim.Adam(self.layer1.parameters(), lr=Config.lr_critic)
        self.opt2 = optim.Adam(self.layer2.parameters(), lr=Config.lr_critic)
        self.opt_actor1 = optim.Adam(self.layer1.actor.parameters(), lr=Config.lr_actor)
        self.opt_actor2 = optim.Adam(self.layer2.actor.parameters(), lr=Config.lr_actor)
        self.opt_integrated = optim.Adam(self.integrated.parameters(), lr=Config.lr_critic)

        # Entropy
        self.log_alpha1 = torch.zeros(1, requires_grad=True, device=Config.device)
        self.log_alpha2 = torch.zeros(1, requires_grad=True, device=Config.device)
        self.alpha_opt1 = optim.Adam([self.log_alpha1], lr=Config.lr_alpha)
        self.alpha_opt2 = optim.Adam([self.log_alpha2], lr=Config.lr_alpha)

        # Prior policy parameters
        self.prior_sigma = 0.45  # From paper's optimal value
        self.prior_decay = 0.00005  # Experience attenuation rate

    def select_action(self, obs_seq, h1=None, h2=None, evaluate=False):
        """
        Select action based on current observation sequence
        obs_seq: [B, L, C, H, W] batch of observation sequences
        h1, h2: hidden states for the two modules
        evaluate: if True, use deterministic actions
        """
        B = obs_seq.size(0)
        if h1 is None:
            h1 = (torch.zeros(1, B, 256, device=Config.device),
                  torch.zeros(1, B, 256, device=Config.device))  # (h, c) for LSTM
        if h2 is None:
            h2 = (torch.zeros(1, B, 256, device=Config.device),
                  torch.zeros(1, B, 256, device=Config.device))  # (h, c) for LSTM

        with torch.no_grad():
            # Layer 1 - Obstacle Avoidance
            out1, h1_new, a1, _, _, _ = self.layer1(obs_seq, h1)

            # Layer 2 - Navigation
            out2, h2_new, a2, _, _, _ = self.layer2(obs_seq, out1, h2)

            if evaluate:
                a1 = self.layer1.actor.get_action(out1[:, -1])
                a2 = self.layer2.actor.get_action(out2[:, -1])

            # Integrated selection (mix between obstacle avoidance and navigation)
            mix = self.integrated(out1[:, -1], out2[:, -1])
            # a = mix * a1 + (1 - mix) * a2  # Linear interpolation

            # Instead of interpolation, choose one based on situation
            # In reality, we'd use mix as a threshold, but for simplicity:
            a = a1 if mix.item() > 0.5 else a2

        return a, h1_new, h2_new

    def compute_rewards(self, state, action, next_state):
        """
        Compute separate rewards for obstacle avoidance and navigation
        Returns:
            r_evade: obstacle avoidance reward
            r_approach: navigation reward
        """
        # This is a simplification - actual implementation would depend on environment details

        # For obstacle avoidance: reward for maintaining distance from obstacles
        closest_obstacle_dist = state.get('obstacle_distance', 1.0)  # Example
        r_evade = -1.0 if closest_obstacle_dist < 0.5 else 0.1  # Penalty for being close to obstacles

        # For navigation: reward for approaching goal
        goal_dist = state.get('goal_distance', 10.0)  # Example
        prev_goal_dist = state.get('prev_goal_distance', 10.0)  # Example
        r_approach = 0.1 * (prev_goal_dist - goal_dist)  # Reward for getting closer to goal

        # Additional reward for reaching goal
        if goal_dist < 0.5:
            r_approach += 10.0

        return r_evade, r_approach

    def soft_update(self, net, target_net):
        """Soft update target network parameters"""
        for p, tp in zip(net.parameters(), target_net.parameters()):
            tp.data.copy_(tp.data * (1 - Config.tau) + p.data * Config.tau)

    def update(self, replay_buffer_evade, replay_buffer_approach):
        """
        Update the agent using samples from both replay buffers
        """
        # Sample from obstacle avoidance buffer
        seqs_evade, idxs_evade, is_weights_evade = replay_buffer_evade.sample(
            Config.batch_size, Config.burn_in + (Config.seq_length - Config.burn_in)
        )
        if not seqs_evade:
            return

        # Sample from navigation buffer
        seqs_approach, idxs_approach, is_weights_approach = replay_buffer_approach.sample(
            Config.batch_size, Config.burn_in + (Config.seq_length - Config.burn_in)
        )
        if not seqs_approach:
            return

        # Process obstacle avoidance sequences
        obs_evade = torch.stack([torch.stack([t.depth_image for t in s], dim=0) for s in seqs_evade], dim=0).unsqueeze(
            2).to(Config.device)
        actions_evade = torch.stack([torch.stack([torch.tensor(t.action) for t in s], dim=0) for s in seqs_evade],
                                    dim=0).to(Config.device)
        rewards_evade = torch.stack([torch.tensor([t.reward_evade for t in s]) for s in seqs_evade], dim=0).to(
            Config.device)
        dones_evade = torch.stack([torch.tensor([t.done for t in s], dtype=torch.float32) for s in seqs_evade],
                                  dim=0).to(Config.device)

        # Process navigation sequences
        obs_approach = torch.stack([torch.stack([t.depth_image for t in s], dim=0) for s in seqs_approach],
                                   dim=0).unsqueeze(2).to(Config.device)
        actions_approach = torch.stack([torch.stack([torch.tensor(t.action) for t in s], dim=0) for s in seqs_approach],
                                       dim=0).to(Config.device)
        rewards_approach = torch.stack([torch.tensor([t.reward_approach for t in s]) for s in seqs_approach], dim=0).to(
            Config.device)
        dones_approach = torch.stack([torch.tensor([t.done for t in s], dtype=torch.float32) for s in seqs_approach],
                                     dim=0).to(Config.device)

        # Split burn-in and training sequences
        bi = Config.burn_in
        obs_evade_bi, obs_evade_train = obs_evade[:, :bi], obs_evade[:, bi:]
        actions_evade_bi, actions_evade_train = actions_evade[:, :bi], actions_evade[:, bi:]
        rewards_evade_bi, rewards_evade_train = rewards_evade[:, :bi], rewards_evade[:, bi:]
        dones_evade_bi, dones_evade_train = dones_evade[:, :bi], dones_evade[:, bi:]

        obs_approach_bi, obs_approach_train = obs_approach[:, :bi], obs_approach[:, bi:]
        actions_approach_bi, actions_approach_train = actions_approach[:, :bi], actions_approach[:, bi:]
        rewards_approach_bi, rewards_approach_train = rewards_approach[:, :bi], rewards_approach[:, bi:]
        dones_approach_bi, dones_approach_train = dones_approach[:, :bi], dones_approach[:, bi:]

        # Initial hidden states
        h1 = (torch.zeros(1, Config.batch_size, 256, device=Config.device),
              torch.zeros(1, Config.batch_size, 256, device=Config.device))
        h2 = (torch.zeros(1, Config.batch_size, 256, device=Config.device),
              torch.zeros(1, Config.batch_size, 256, device=Config.device))

        # ---- Update Layer 1 (Obstacle Avoidance) ----
        out1, h1_new, a1, logp1, q11, q12 = self.layer1(obs_evade, h1)

        # Compute targets for layer 1
        with torch.no_grad():
            _, _, a1_next, logp1_next, _, _ = self.layer1(obs_evade, h1_new)
            min_tq1 = torch.min(
                self.target_q1_1(out1[:, -1], a1_next),
                self.target_q1_2(out1[:, -1], a1_next)
            )
            target1 = rewards_evade_train[:, -1].unsqueeze(-1) + \
                      (1 - dones_evade_train[:, -1].unsqueeze(-1)) * \
                      Config.gamma * (min_tq1 - self.log_alpha1.exp() * logp1_next)

        # Critic 1 loss
        c1 = self.layer1.q1(out1[:, -1], a1)
        c2 = self.layer1.q2(out1[:, -1], a1)
        critic1_loss = F.mse_loss(c1, target1) + F.mse_loss(c2, target1)

        # Actor 1 loss
        actor1_loss = (self.log_alpha1.exp() * logp1 - torch.min(c1, c2)).mean()

        # Entropy 1 loss
        alpha1_loss = -(self.log_alpha1 * (logp1 + Config.target_entropy).detach()).mean()

        # Update layer 1
        self.opt1.zero_grad()
        critic1_loss.backward(retain_graph=True)
        self.opt1.step()

        self.opt_actor1.zero_grad()
        actor1_loss.backward()
        self.opt_actor1.step()

        self.alpha_opt1.zero_grad()
        alpha1_loss.backward()
        self.alpha_opt1.step()

        # Soft update targets 1
        self.soft_update(self.layer1.q1, self.target_q1_1)
        self.soft_update(self.layer1.q2, self.target_q1_2)

        # ---- Update Layer 2 (Navigation) ----
        out2, h2_new, a2, logp2, q21, q22 = self.layer2(obs_approach, out1.detach(), h2)

        # Compute targets for layer 2
        with torch.no_grad():
            _, _, a2_next, logp2_next, _, _ = self.layer2(obs_approach, out1.detach(), h2_new)
            min_tq2 = torch.min(
                self.target_q2_1(out2[:, -1], a2_next),
                self.target_q2_2(out2[:, -1], a2_next)
            )
            target2 = rewards_approach_train[:, -1].unsqueeze(-1) + \
                      (1 - dones_approach_train[:, -1].unsqueeze(-1)) * \
                      Config.gamma * (min_tq2 - self.log_alpha2.exp() * logp2_next)

        # Critic 2 loss
        c21 = self.layer2.q1(out2[:, -1], a2)
        c22 = self.layer2.q2(out2[:, -1], a2)
        critic2_loss = F.mse_loss(c21, target2) + F.mse_loss(c22, target2)

        # Actor 2 loss
        actor2_loss = (self.log_alpha2.exp() * logp2 - torch.min(c21, c22)).mean()

        # Entropy 2 loss
        alpha2_loss = -(self.log_alpha2 * (logp2 + Config.target_entropy).detach()).mean()

        # Update layer 2
        self.opt2.zero_grad()
        critic2_loss.backward(retain_graph=True)
        self.opt2.step()

        self.opt_actor2.zero_grad()
        actor2_loss.backward()
        self.opt_actor2.step()

        self.alpha_opt2.zero_grad()
        alpha2_loss.backward()
        self.alpha_opt2.step()

        # Soft update targets 2
        self.soft_update(self.layer2.q1, self.target_q2_1)
        self.soft_update(self.layer2.q2, self.target_q2_2)

        # Update integrated network (simplified loss)
        integrated_out = self.integrated(out1[:, -1].detach(), out2[:, -1].detach())

        # Integrated loss based on which module is more accurate
        # This is simplified - in practice you'd use a more sophisticated loss
        obstacle_near = rewards_evade_train[:, -1] < 0  # Negative reward means obstacle is near
        integrated_target = obstacle_near.float().unsqueeze(1)  # 1 if obstacle is near (use layer1)
        integrated_loss = F.binary_cross_entropy(integrated_out, integrated_target)

        self.opt_integrated.zero_grad()
        integrated_loss.backward()
        self.opt_integrated.step()

        # Update buffer priorities
        td_errs1 = torch.abs(c1 - target1).detach().squeeze().tolist()
        td_errs2 = torch.abs(c21 - target2).detach().squeeze().tolist()

        replay_buffer_evade.update_priorities(idxs_evade, td_errs1)
        replay_buffer_approach.update_priorities(idxs_approach, td_errs2)

        return {
            'loss_evade': critic1_loss.item(),
            'policy_evade': actor1_loss.item(),
            'alpha_evade': self.log_alpha1.exp().item(),
            'loss_approach': critic2_loss.item(),
            'policy_approach': actor2_loss.item(),
            'alpha_approach': self.log_alpha2.exp().item(),
            'loss_integrated': integrated_loss.item()
        }


# Example training loop and environment would be implemented separately
# This would include:
# 1. Environment setup with depth image observations
# 2. Replay buffer creation for each module
# 3. Training loop with action selection and reward collection
# 4. Evaluation of trained policy

# Example transition class for storing experiences
class Transition:
    def __init__(self, depth_image, action, reward_evade, reward_approach, next_depth_image, done):
        self.depth_image = depth_image  # [1, 84, 84] tensor
        self.action = action  # [2] tensor [angular_velocity, linear_velocity]
        self.reward_evade = reward_evade  # scalar for obstacle avoidance
        self.reward_approach = reward_approach  # scalar for navigation
        self.next_depth_image = next_depth_image  # [1, 84, 84] tensor
        self.done = done  # boolean

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
            if self.goal_distance >= 15:
                print(self.goal_position)
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
        reward, done, info = self._compute_reward(depth_image, state, position)

        # 스텝 카운터 증가
        self.steps += 1

        # 최대 스텝 수 초과 시 종료
        if self.steps >= Config.max_episode_steps:
            done = True
            info['timeout'] = True

        return depth_image, state, reward, done, info

    def _get_lidar_data(self):
        """라이다 데이터를 정규화된 2D depth 이미지로 변환"""
        lidar_data = self.client.getLidarData("LidarSensor1", "HelloDrone").point_cloud
        # 포인트 수 확인
        points = []
        if len(lidar_data) >= 3:
            # 라이다 데이터는 [x1, y1, z1, x2, y2, z2, ...] 형태로 제공됩니다
            for i in range(0, len(lidar_data), 3):
                if i + 2 < len(lidar_data):  # 안전 검사
                    x, y, z = lidar_data[i], lidar_data[i + 1], lidar_data[i + 2]
                    points.append((x, y, z))

        return lidar_to_depth_image(points)

    def _get_state(self):
        # 드론 상태와 방향 가져오기
        kinematics = self.client.simGetGroundTruthKinematics()

        depth_image = self._get_lidar_data()
        position = np.array([kinematics.position.x_val, kinematics.position.y_val, kinematics.position.z_val])
        orientation = kinematics.orientation

        # 속도 가져오기 (NED 좌표계)
        velocity_ned = np.array([
            kinematics.linear_velocity.x_val,
            kinematics.linear_velocity.y_val,
            kinematics.linear_velocity.z_val
        ])

        # NED 좌표계에서 목표까지의 상대 벡터
        relative_goal_ned = self.goal_position - position

        # 쿼터니언을 회전 행렬로 변환
        euler = self._quaternion_to_euler(
            np.array([orientation.w_val, orientation.x_val, orientation.y_val, orientation.z_val])
        )
        rotation_matrix = self._quaternion_to_rotation_matrix(
            orientation.w_val, orientation.x_val, orientation.y_val, orientation.z_val
        )

        # NED 속도와 상대 목표 위치를 바디 프레임으로 변환
        velocity_body = np.dot(rotation_matrix.T, velocity_ned)
        relative_goal_body = np.dot(rotation_matrix.T, relative_goal_ned)

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
        goal_distance = np.linalg.norm(drone_state[3:6])

        # 충돌 확인
        collision_info = self.client.simGetCollisionInfo()
        has_collided = collision_info.has_collided

        # 보상 계산
        reward = 0
        info = {}
        done = False

        info["distance_to_goal"] = goal_distance

        # 목표 도달 확인
        if goal_distance < Config.goal_threshold:
            print(position)
            reward += 100.0  # 목표 도달 큰 보상
            done = True
            info["status"] = "goal_reached"

        # 충돌 확인
        elif has_collided:
            print(position)
            reward -= 100.0  # 충돌 큰 페널티
            done = True
            info["status"] = "collision"

        # 목표 방향 진행 보상
        else:
            # 이전 위치 (히스토리에서)
            prev_goal_distance = self.goal_distance

            # 목표를 향해 가까워지면 보상, 멀어지면 페널티
            distance_reward = np.clip(prev_goal_distance - goal_distance, -1.0, 1.0)
            reward += distance_reward * 3.0

            # 방향 정렬 보상 추가
            if goal_distance > 0:
                goal_direction = drone_state[3:6]
                goal_direction_magnitude = np.linalg.norm(goal_direction)
                if goal_direction_magnitude > 0:  # 0으로 나누기 방지
                    normalized_goal_direction = goal_direction / goal_direction_magnitude
                else:
                    normalized_goal_direction = np.zeros_like(goal_direction)

                velocity = drone_state[:3]
                velocity_magnitude = np.linalg.norm(velocity)

                if velocity_magnitude > 0.5:  # 드론이 충분히 움직이고 있을 때만
                    velocity_direction = velocity / velocity_magnitude
                    alignment = np.dot(normalized_goal_direction, velocity_direction)
                    # 이제 정확히 -1(반대) ~ +1(정렬) 범위의 값
                    alignment_reward = alignment * 4.0
                    reward += alignment_reward

            # 시간 패널티 (빠른 목표 도달 장려)
            reward -= 0.1

            # 에너지 효율성 (급격한 움직임 패널티)
            velocity = drone_state[:3]
            speed = np.linalg.norm(velocity)

            # 너무 빠른 속도에 페널티
            if speed > Config.max_drone_speed:
                reward -= 0.2 * (speed - Config.max_drone_speed)

            # 장애물 회피 보상 추가
            min_depth = np.min(depth_image) * Config.max_lidar_distance

            # 안전 거리 임계값
            safety_threshold = 3.0  # 미터
            danger_threshold = 1.5  # 미터

            # 장애물이 가까울수록 페널티
            if min_depth < safety_threshold:
                # 선형 보간: 안전 거리에서는 작은 페널티, 위험 거리에서는 큰 페널티
                obstacle_penalty = -3.0 * (1.0 - (min_depth - danger_threshold) /
                                           (safety_threshold - danger_threshold))
                obstacle_penalty = max(obstacle_penalty, -3.0)  # 최대 페널티 제한
                reward += obstacle_penalty

                # 디버깅 정보 추가
                info["obstacle_distance"] = min_depth
                info["obstacle_penalty"] = obstacle_penalty

            info["status"] = "moving"

        self.goal_distance = goal_distance

        return reward, done, info


#로깅 기능
def setup_logger():
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # 파일 핸들러 설정
    file_handler = logging.FileHandler(f"{log_dir}/training_{timestamp}.log")
    file_handler.setLevel(logging.INFO)

    # 콘솔 핸들러 설정
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 포맷 설정
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 로거 설정
    logger = logging.getLogger("drone_sac")
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

# 학습 함수
def train():
    logger = setup_logger()

    # 에이전트 및 환경 초기화
    agent = SACAgent()
    env = DroneEnv()

    # 리플레이 버퍼 로드 시도
    buffer_path = os.path.join(Config.checkpoint_dir, f"{Config.model_name}_buffer.pkl")
    try:
        agent.replay_buffer.load(buffer_path)
        logger.info(f"리플레이 버퍼 로드 완료: {len(agent.replay_buffer)} 에피소드")
    except:
        logger.info("새 리플레이 버퍼 생성")

    # 이전 체크포인트 로드 시도
    latest_model_path = os.path.join(Config.checkpoint_dir, f"{Config.model_name}_latest.pt")
    start_episode = 0

    if os.path.exists(latest_model_path):
        episode = agent.load_checkpoint(latest_model_path)
        if episode is not None:
            start_episode = episode + 1
            logger.info(f"체크포인트 로드 완료: 에피소드 {start_episode}부터 계속")

    # 학습 통계


    logger.info("학습 시작")

    for episode in range(start_episode, Config.num_episodes):
        # 환경 초기화
        depth_image, state = env.reset()

        # GRU 히든 스테이트 초기화
        hidden = agent.init_hidden(1)

        episode_reward = 0
        done = False

        steps = 0
        info = {}

        stats = None

        for step in range(Config.max_episode_steps):
            # 액션 선택
            action, hidden = agent.select_action(depth_image, state, hidden)

            # early_phase = episode < 200
            # if early_phase:
            #     # 목표 방향 추출 및 정규화
            #     goal_direction = state[3:6]
            #     goal_distance = np.linalg.norm(goal_direction)
            #
            #     if goal_distance > 0:
            #         goal_direction = goal_direction / goal_distance
            #
            #     # 블렌딩 비율 계산 (시간에 따라 감소)
            #     blend_ratio = max(0.2, 0.8 - 0.003 * episode)
            #
            #     # 목표 방향 기반 액션
            #     goal_action = goal_direction * Config.max_drone_speed
            #
            #     # 액션 블렌딩
            #     action = blend_ratio * goal_action + (1 - blend_ratio) * action
            #
            #     # 약간의 노이즈 추가
            #     noise_scale = 0.3
            #     noise = np.random.normal(0, noise_scale, 3)
            #     action = action + noise
            #
            #     # 액션 클리핑
            #     action = np.clip(action, -Config.max_drone_speed, Config.max_drone_speed)

            # 환경 스텝
            next_depth_image, next_state, reward, done, info = env.step(action)

            # 리플레이 버퍼에 저장
            agent.replay_buffer.push(
                depth_image, state, action, reward, next_depth_image, next_state, done
            )

            # 상태 업데이트
            depth_image = next_depth_image
            state = next_state

            episode_reward += reward

            steps += 1



            if done:
                break

        # 에피소드 종료 후 학습
        if len(agent.replay_buffer) > Config.batch_size:
            # 소규모 업데이트 (1~3회)
            for _ in range(3):
                agent.update(
                    batch_size=Config.batch_size,
                    sequence_length=Config.sequence_length
                )

        if len(agent.replay_buffer) > Config.batch_size and (episode + 1) % 5 == 0:
            print(f"마일스톤 학습 시작 (에피소드 {episode})")

            for _ in range(50):
                agent.update(
                    batch_size= Config.batch_size,
                    sequence_length=Config.sequence_length
                )

        # 목표까지 거리
        distance = info.get('distance_to_goal', float('inf'))

        # 로깅
        log_msg = (f"에피소드 {episode} - "
                   f"보상: {episode_reward:.2f}, "
                   f"스텝: {steps}, "
                   f"최종거리: {distance:.2f}m, "
                   f"충돌: {info['status']}")

        print(f"에피소드 {episode} - "
              f"보상: {episode_reward:.2f}, "
              f"스텝: {steps}, "
              f"최종거리: {distance:.2f}m, "
              f"충돌: {info['status']}")
        print()

        logger.info(log_msg)


        # 체크포인트 저장 (20 에피소드마다)
        if episode % 20 == 0:
            agent.save_checkpoint(episode)

            # 리플레이 버퍼 저장
            agent.replay_buffer.save(buffer_path)
            logger.info(f"리플레이 버퍼 저장 완료: {len(agent.replay_buffer)} 에피소드")

    # 최종 모델 저장
    agent.save_checkpoint(Config.num_episodes - 1)
    logger.info("학습 완료")


# 평가 함수
def evaluate(model_path, num_episodes=10):
    # 에이전트 및 환경 초기화
    agent = SACAgent()
    env = DroneEnv()

    # 모델 로드
    if agent.load_checkpoint(model_path) is None:
        print(f"모델 로드 실패: {model_path}")
        return

    success_count = 0
    collision_count = 0
    timeout_count = 0
    episode_rewards = []
    episode_distances = []

    for episode in range(num_episodes):
        depth_image, state = env.reset()
        hidden = agent.init_hidden(1)

        episode_reward = 0
        done = False

        for step in range(Config.max_episode_steps):
            # 평가 모드에서는 결정론적 액션 선택
            action, hidden = agent.select_action(depth_image, state, hidden, evaluate=True)

            next_depth_image, next_state, reward, done, info = env.step(action)

            depth_image = next_depth_image
            state = next_state
            episode_reward += reward

            if done:
                break

        # 에피소드 결과
        episode_rewards.append(episode_reward)
        distance = info.get('distance_to_goal', float('inf'))
        episode_distances.append(distance)

        # 결과 카운팅
        if info.get('success', False):
            success_count += 1
            result = "성공"
        elif info.get('collision', False):
            collision_count += 1
            result = "충돌"
        else:
            timeout_count += 1
            result = "타임아웃"

        print(f"에피소드 {episode} - 결과: {result}, 보상: {episode_reward:.2f}, 스텝: {step + 1}, 최종거리: {distance:.2f}m")

    # 종합 결과
    success_rate = success_count / num_episodes
    avg_reward = sum(episode_rewards) / num_episodes
    avg_distance = sum(episode_distances) / num_episodes

    print("\n===== 평가 결과 =====")
    print(f"성공률: {success_rate:.2f} ({success_count}/{num_episodes})")
    print(f"충돌률: {collision_count / num_episodes:.2f} ({collision_count}/{num_episodes})")
    print(f"타임아웃률: {timeout_count / num_episodes:.2f} ({timeout_count}/{num_episodes})")
    print(f"평균 보상: {avg_reward:.2f}")
    print(f"평균 최종거리: {avg_distance:.2f}m")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"],
                        help="학습 또는 평가 모드 선택")
    parser.add_argument("--model", type=str, default=None,
                        help="평가 모드에서 사용할 모델 경로")
    args = parser.parse_args()

    if args.mode == "train":
        train()
    elif args.mode == "eval":
        if args.model is None:
            # 최신 모델 사용
            model_path = os.path.join(Config.checkpoint_dir, f"{Config.model_name}_latest.pt")
        else:
            model_path = args.model

        evaluate(model_path)