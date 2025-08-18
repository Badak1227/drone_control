import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Normal
from collections import deque, namedtuple
from copy import deepcopy
import random
import matplotlib.pyplot as plt


# Configuration for SAC with GRU
class SACConfig:
    # Environment
    max_episode_steps = 1000
    goal_threshold = 2.0
    seq_length = 20  # 0.5초 커버
    burn_in_length = 5  # 0.2초 context

    # Neural Network - 차원 일관성을 위한 정리
    hidden_dim = 256
    cnn_features = 64
    gru_hidden_dim = 128
    gru_num_layers = 2

    # SAC Parameters
    batch_size = 16
    buffer_capacity = 500  # 일관된 이름과 값 사용
    gamma = 0.99
    tau = 0.005
    lr_actor = 3e-4
    lr_critic = 3e-4
    lr_alpha = 3e-4
    alpha_init = 0.2
    target_entropy = -3

    # Training
    start_steps = 1000
    update_interval = 1
    max_grad_norm = 10.0  # gradient clipping 명시화

    # Learning rate scheduler 옵션 (선택적 사용)
    use_lr_scheduler = False
    lr_scheduler_step_size = 10000
    lr_scheduler_gamma = 0.5


# CNN Feature Extractor (no self-attention for efficiency)
class CNNFeatureExtractor(nn.Module):
    def __init__(self, input_channels=1, output_dim=SACConfig.cnn_features):
        super(CNNFeatureExtractor, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.fc = nn.Linear(128 * 5 * 5, output_dim)

    def forward(self, x):
        # 입력 형태 검증
        if len(x.shape) != 4:
            raise ValueError(f"Expected input with shape [B, C, H, W], got {x.shape}")
        if x.shape[1] != 1:
            raise ValueError(f"Expected 1 channel, got {x.shape[1]}")

        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return x


# Visual-State Summarizer Module
class VisualStateSummarizer(nn.Module):
    def __init__(self, state_dim):
        super().__init__()

        # CNN for visual features
        self.cnn = CNNFeatureExtractor()

        # State processing
        self.state_fc = nn.Linear(state_dim, SACConfig.cnn_features)

        # Combined feature processing
        self.feature_combine = nn.Linear(SACConfig.cnn_features * 2, SACConfig.hidden_dim)

        # GRU for temporal processing
        self.gru = nn.GRU(
            input_size=SACConfig.hidden_dim,
            hidden_size=SACConfig.gru_hidden_dim,
            num_layers=SACConfig.gru_num_layers,
            batch_first=True
        )

        self.hidden_dim = SACConfig.gru_hidden_dim

        # 차원 체크를 위한 출력 projection (필요시)
        if SACConfig.hidden_dim != SACConfig.gru_hidden_dim:
            self.output_projection = nn.Linear(SACConfig.gru_hidden_dim, SACConfig.hidden_dim)
        else:
            self.output_projection = nn.Identity()

    def forward_single(self, depth_image, state, hidden=None):
        """
        Single frame processing (for inference)
        Args:
            depth_image: [batch_size, H, W]
            state: [batch_size, state_dim]
            hidden: hidden state from previous step
        """
        batch_size = depth_image.shape[0]

        # CNN processing (add channel dimension)
        visual_features = self.cnn(depth_image.unsqueeze(1))

        # State processing
        state_features = F.relu(self.state_fc(state))

        # Combine features
        combined = torch.cat([visual_features, state_features], dim=-1)
        combined = F.relu(self.feature_combine(combined))

        # Add sequence dimension for GRU (single step sequence)
        combined = combined.unsqueeze(1)  # [batch_size, 1, hidden_dim]

        # GRU processing
        gru_output, new_hidden = self.gru(combined, hidden)

        self.gru.flatten_parameters()
        # Remove sequence dimension
        gru_features = gru_output.squeeze(1)

        # Project if needed
        features = self.output_projection(gru_features)

        return features, new_hidden

    def forward_sequence(self, depth_images, states, hidden=None):
        """
        Sequence processing (for training)
        Args:
            depth_images: [batch_size, seq_len, H, W]
            states: [batch_size, seq_len, state_dim]
            hidden: initial hidden state
        """
        batch_size, seq_len = depth_images.shape[:2]

        # Reshape for CNN processing: [batch*seq, H, W]
        depth_reshaped = depth_images.reshape(batch_size * seq_len, *depth_images.shape[2:])

        # CNN processing  [batch*seq, cnn_features]
        visual_features = self.cnn(depth_reshaped.unsqueeze(1))

        # Reshape back to sequence: [batch, seq, cnn_features]
        visual_features = visual_features.reshape(batch_size, seq_len, -1)

        # State processing (FC layer handles batch and sequence automatically)
        state_features = F.relu(self.state_fc(states))

        # Combine features
        combined = torch.cat([visual_features, state_features], dim=-1)
        combined = F.relu(self.feature_combine(combined))

        self.gru.flatten_parameters()
        # GRU processing with full sequence
        gru_output, new_hidden = self.gru(combined, hidden)

        # Project if needed
        features = self.output_projection(gru_output)

        return features, new_hidden

# Actor Network - 입력 차원 명확화
class Actor(nn.Module):
    def __init__(self, input_dim=SACConfig.gru_hidden_dim, action_dim=None, max_action=1.0):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, SACConfig.hidden_dim)
        self.fc2 = nn.Linear(SACConfig.hidden_dim, SACConfig.hidden_dim)

        self.mean = nn.Linear(SACConfig.hidden_dim, action_dim)
        self.log_std = nn.Linear(SACConfig.hidden_dim, action_dim)

        self.max_action = max_action
        self.action_dim = action_dim

    def forward(self, features):
        x = F.relu(self.fc1(features))
        x = F.relu(self.fc2(x))

        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), min=-20, max=2)

        return mean, log_std.exp()

    def sample(self, features):
        mean, std = self.forward(features)

        normal = Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)

        # Compute log probability
        log_prob = normal.log_prob(x_t).sum(dim=-1, keepdim=True)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1, keepdim=True)

        # Scale to action space
        action = action * self.max_action
        mean_action = torch.tanh(mean) * self.max_action

        return action, log_prob, mean_action


# Critic Network - 입력 차원 명확화
class Critic(nn.Module):
    def __init__(self, input_dim=SACConfig.gru_hidden_dim, action_dim=None):
        super().__init__()

        # Q1 network
        self.q1_fc1 = nn.Linear(input_dim + action_dim, SACConfig.hidden_dim)
        self.q1_fc2 = nn.Linear(SACConfig.hidden_dim, SACConfig.hidden_dim)
        self.q1_out = nn.Linear(SACConfig.hidden_dim, 1)

        # Q2 network
        self.q2_fc1 = nn.Linear(input_dim + action_dim, SACConfig.hidden_dim)
        self.q2_fc2 = nn.Linear(SACConfig.hidden_dim, SACConfig.hidden_dim)
        self.q2_out = nn.Linear(SACConfig.hidden_dim, 1)

    def forward(self, features, action):
        combined = torch.cat([features, action], dim=-1)

        # Q1
        q1 = F.relu(self.q1_fc1(combined))
        q1 = F.relu(self.q1_fc2(q1))
        q1 = self.q1_out(q1)

        # Q2
        q2 = F.relu(self.q2_fc1(combined))
        q2 = F.relu(self.q2_fc2(q2))
        q2 = self.q2_out(q2)

        return q1, q2

# RecurrentReplayBuffer with improved memory management
BurnInRecurrentBatch = namedtuple('BurnInRecurrentBatch', [
    'burn_in_depth', 'burn_in_states',
    'depth_sequences', 'states', 'actions', 'rewards',
    'next_depth_sequences', 'next_states', 'dones'])


class EpisodeReplayBuffer:
    """
    Stores entire episodes and allows sampling of fixed-length subsequences (segments) from within stored episodes.

    Attributes:
        capacity (int): max number of episodes to store
        max_episode_len (int): maximum expected length of an episode (timesteps)
        segment_len (int): length of subsequence to sample
        device (torch.device): device for sampled tensors
        ptr (int): next episode index to overwrite
        size (int): current number of stored episodes
        episodes (dict of lists): temporary buffers for current episode
        storage (dict of np.ndarray): arrays storing full episodes
        ep_lens (np.ndarray): actual lengths of stored episodes
    """
    def __init__(self, o_dim, state_dim, action_dim,
                 capacity, max_episode_len, segment_len,
                 device=None):
        self.capacity = capacity
        self.max_episode_len = max_episode_len
        self.segment_len = segment_len
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # Ring buffer pointers
        self.ptr = 0
        self.size = 0

        # Temporary buffers for current episode
        self.episodes = {
            'depth': [],
            'state': [],
            'action': [],
            'reward': [],
            'next_depth': [],
            'next_state': [],
            'done': []
        }

        # Storage arrays: [capacity, max_len(+1) , ...]
        self.storage = {
            'depth': np.zeros((capacity, max_episode_len+1, *o_dim), dtype=np.float32),
            'state': np.zeros((capacity, max_episode_len+1, state_dim), dtype=np.float32),
            'action': np.zeros((capacity, max_episode_len, action_dim), dtype=np.float32),
            'reward': np.zeros((capacity, max_episode_len, 1), dtype=np.float32),
            'next_depth': np.zeros((capacity, max_episode_len+1, *o_dim), dtype=np.float32),
            'next_state': np.zeros((capacity, max_episode_len+1, state_dim), dtype=np.float32),
            'done': np.zeros((capacity, max_episode_len, 1), dtype=np.float32)
        }
        # Actual episode lengths
        self.ep_lens = np.zeros(capacity, dtype=np.int32)

    def add(self, depth, state, action, reward, next_depth, next_state, done):
        """
        Add a timestep to current episode. When done=True, finalize and store the episode.
        """
        # Append timestep
        self.episodes['depth'].append(depth)
        self.episodes['state'].append(state)
        self.episodes['action'].append(action)
        self.episodes['reward'].append([reward])
        self.episodes['next_depth'].append(next_depth)
        self.episodes['next_state'].append(next_state)
        self.episodes['done'].append([done])

        if done:
            ep_len = len(self.episodes['action'])
            idx = self.ptr
            # Clip if longer than max_episode_len
            L = min(ep_len, self.max_episode_len)
            # Store arrays
            for key, buf in self.episodes.items():
                arr = np.array(buf)
                if key in ('depth', 'state'):
                    # size max_episode_len+1
                    self.storage[key][idx, :L+1] = arr[:L+1]
                else:
                    # size max_episode_len
                    self.storage[key][idx, :L] = arr[:L]
            self.ep_lens[idx] = L
            # Update pointers
            self.ptr = (self.ptr + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)
            # Reset temp buffers
            for buf in self.episodes.values():
                buf.clear()

    def sample(self, batch_size):
        """
        Sample batch of fixed-length segments: choose random episodes,
        then random start idx within each episode (so segment fits),
        and return tensors of shape [B, segment_len, ...].
        """
        assert self.size > 0, "No episodes to sample from."
        # Randomly select episodes
        idxs = np.random.randint(0, self.size, size=batch_size)
        depth_batch = []
        state_batch = []
        action_batch = []
        reward_batch = []
        next_depth_batch = []
        next_state_batch = []
        done_batch = []

        for i, ep in enumerate(idxs):
            L = self.ep_lens[ep]
            # If episode shorter than segment_len, sample from start
            if L < self.segment_len:
                start = 0
            else:
                start = np.random.randint(0, L - self.segment_len + 1)
            end = start + self.segment_len
            # Append slices
            depth_batch.append(self.storage['depth'][ep, start:end])
            state_batch.append(self.storage['state'][ep, start:end])
            action_batch.append(self.storage['action'][ep, start:end])
            reward_batch.append(self.storage['reward'][ep, start:end])
            next_depth_batch.append(self.storage['next_depth'][ep, start+1:end+1])
            next_state_batch.append(self.storage['next_state'][ep, start+1:end+1])
            done_batch.append(self.storage['done'][ep, start:end])

        def to_tensor(arr_list):
            arr = np.stack(arr_list, axis=0)
            return torch.tensor(arr, dtype=torch.float32, device=self.device)

        return (
            to_tensor(depth_batch),
            to_tensor(state_batch),
            to_tensor(action_batch),
            to_tensor(reward_batch),
            to_tensor(next_depth_batch),
            to_tensor(next_state_batch),
            to_tensor(done_batch)
        )

    def __len__(self):
        return self.size


# Main GRU-SAC Agent with optimized processing
class GRUSACAgent:
    def __init__(self, state_dim, action_dim, o_dim, max_action=1.0, device="cpu"):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.o_dim = o_dim
        self.max_action = max_action
        self.device = device

        # Initialize separate networks for Actor and Critic
        # Actor networks (for inference and learning)
        self.actor_summarizer = VisualStateSummarizer(state_dim).to(device)
        self.actor = Actor(
            input_dim=SACConfig.hidden_dim,
            action_dim=action_dim,
            max_action=max_action
        ).to(device)

        # Critic networks
        self.critic_summarizer = VisualStateSummarizer(state_dim).to(device)
        self.critic = Critic(
            input_dim=SACConfig.hidden_dim,
            action_dim=action_dim
        ).to(device)

        # Target networks for both summarizer and critic
        self.critic_summarizer_target = deepcopy(self.critic_summarizer)
        self.critic_target = deepcopy(self.critic)

        # Separate optimizers with appropriate learning rates
        self.actor_summarizer_optimizer = optim.Adam(self.actor_summarizer.parameters(), lr=SACConfig.lr_actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=SACConfig.lr_actor)

        self.critic_summarizer_optimizer = optim.Adam(self.critic_summarizer.parameters(), lr=SACConfig.lr_critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=SACConfig.lr_critic)

        # Alpha temperature parameter
        self.log_alpha = torch.tensor(np.log(SACConfig.alpha_init)).to(device).requires_grad_(True)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=SACConfig.lr_alpha)
        self.target_entropy = SACConfig.target_entropy

        # Learning rate schedulers (optional)
        if SACConfig.use_lr_scheduler:
            self.actor_scheduler = optim.lr_scheduler.StepLR(
                self.actor_optimizer,
                step_size=SACConfig.lr_scheduler_step_size,
                gamma=SACConfig.lr_scheduler_gamma
            )
            self.critic_scheduler = optim.lr_scheduler.StepLR(
                self.critic_optimizer,
                step_size=SACConfig.lr_scheduler_step_size,
                gamma=SACConfig.lr_scheduler_gamma
            )

        # Experience buffer
        self.memory = RecurrentReplayBuffer(
            o_dim=o_dim,
            state_dim=state_dim,
            action_dim=action_dim,
            max_episode_len=SACConfig.max_episode_steps,
            segment_len=SACConfig.seq_length,
            burn_in_len=SACConfig.burn_in_length,
            capacity=SACConfig.buffer_capacity,
            batch_size=SACConfig.batch_size
        )

        # Separate hidden states for actor and critic (only for inference)
        self.actor_hidden = None  # for real-time inference
        self.critic_hidden = None  # could be used for real-time evaluation if needed

        # Training step counter
        self.training_steps = 0

    def reset_hidden(self):
        """Reset hidden states between episodes"""
        self.actor_hidden = None
        self.critic_hidden = None

    def select_action(self, depth_image, state, evaluate=False):
        """
        Single frame inference (matches real-time usage)
        """
        with torch.no_grad():
            # Convert to tensors
            depth_tensor = torch.FloatTensor(depth_image).to(self.device)
            state_tensor = torch.FloatTensor(state).to(self.device)

            # Add batch dimension
            depth_tensor = depth_tensor.unsqueeze(0)  # [1, H, W]
            state_tensor = state_tensor.unsqueeze(0)  # [1, state_dim]

            # Process single frame with actor summarizer
            features, self.actor_hidden = self.actor_summarizer.forward_single(
                depth_tensor, state_tensor, self.actor_hidden
            )

            # Get action
            if evaluate:
                _, _, mean_action = self.actor.sample(features)
                action = mean_action
            else:
                action, _, _ = self.actor.sample(features)

            return action.squeeze().detach().cpu().numpy()

    def update_parameters(self, delayed_update=True):
        """
        Efficient training with sequence processing
        """
        if len(self.memory) < SACConfig.batch_size:
            return None

        self.training_steps += 1

        if delayed_update and self.training_steps % SACConfig.update_interval != 0:
            return None

        # Sample batch with burn-in
        batch = self.memory.sample_with_burnin()

        # Current alpha
        alpha = self.log_alpha.exp().detach()  # Detach alpha for stability

        # Concatenate burn-in and main sequence for continuous processing
        full_depth_seq = torch.cat([batch.burn_in_depth, batch.depth_sequences], dim=1)
        full_state_seq = torch.cat([batch.burn_in_states, batch.states], dim=1)

        # Next state sequences (shifted)
        full_next_depth_seq = torch.cat([batch.burn_in_depth, batch.next_depth_sequences], dim=1)
        full_next_state_seq = torch.cat([batch.burn_in_states, batch.next_states], dim=1)

        # Critic Update
        # Process full sequences efficiently
        critic_features, _ = self.critic_summarizer.forward_sequence(full_depth_seq, full_state_seq)

        # Extract only the main sequence features (skip burn-in)
        burn_in_len = batch.burn_in_depth.shape[1]
        main_critic_features = critic_features[:, burn_in_len:]

        # Current Q-values
        current_q1, current_q2 = self.critic(main_critic_features, batch.actions)

        # Next state features (for target calculation)
        with torch.no_grad():
            # Process next states with actor and critic networks
            next_actor_features, _ = self.actor_summarizer.forward_sequence(
                full_next_depth_seq, full_next_state_seq
            )
            next_critic_target_features, _ = self.critic_summarizer_target.forward_sequence(
                full_next_depth_seq, full_next_state_seq
            )

            # Extract main sequence
            main_next_actor_features = next_actor_features[:, burn_in_len:]
            main_next_critic_target_features = next_critic_target_features[:, burn_in_len:]

            # Next action from actor
            next_action, next_log_prob, _ = self.actor.sample(main_next_actor_features)

            # Target Q-values
            target_q1, target_q2 = self.critic_target(main_next_critic_target_features, next_action)
            target_q = torch.min(target_q1, target_q2) - alpha * next_log_prob
            target_value = batch.rewards + (1 - batch.dones) * SACConfig.gamma * target_q

        # Critic loss
        critic_loss = F.mse_loss(current_q1, target_value) + F.mse_loss(current_q2, target_value)

        # Update critic
        self.critic_optimizer.zero_grad()
        self.critic_summarizer_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), SACConfig.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.critic_summarizer.parameters(), SACConfig.max_grad_norm)
        self.critic_optimizer.step()
        self.critic_summarizer_optimizer.step()

        # Actor Update
        # Process full sequence for actor (separate from critic to avoid interference)
        actor_features, _ = self.actor_summarizer.forward_sequence(full_depth_seq, full_state_seq)
        main_actor_features = actor_features[:, burn_in_len:]

        # New action from actor
        new_action, log_prob, _ = self.actor.sample(main_actor_features)

        # Get Q-value for actor loss using detached critic features
        with torch.no_grad():
            # Recompute critic features for actor (avoiding in-place operations)
            q_critic_features, _ = self.critic_summarizer.forward_sequence(full_depth_seq, full_state_seq)
            q_main_critic_features = q_critic_features[:, burn_in_len:].detach()

        q1_new, q2_new = self.critic(q_main_critic_features, new_action)
        q_new = torch.min(q1_new, q2_new)

        # Actor loss
        actor_loss = (alpha * log_prob - q_new).mean()

        # Update actor
        self.actor_optimizer.zero_grad()
        self.actor_summarizer_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), SACConfig.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.actor_summarizer.parameters(), SACConfig.max_grad_norm)
        self.actor_optimizer.step()
        self.actor_summarizer_optimizer.step()

        # Alpha Update
        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Update learning rates if scheduler is enabled
        if SACConfig.use_lr_scheduler:
            self.actor_scheduler.step()
            self.critic_scheduler.step()

        # Soft update target networks
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - SACConfig.tau) + param.data * SACConfig.tau)

        for target_param, param in zip(self.critic_summarizer_target.parameters(), self.critic_summarizer.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - SACConfig.tau) + param.data * SACConfig.tau)

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha': alpha.item()
        }

    def add_experience(self, depth_image, state, action, reward, next_depth_image, next_state, done, cutoff=False):
        """Add experience to memory buffer"""
        self.memory.add(depth_image, state, action, reward, next_depth_image, next_state, done, cutoff)

    def save(self, filename):
        """Save agent state"""
        try:
            torch.save({
                'actor_summarizer_state_dict': self.actor_summarizer.state_dict(),
                'actor_state_dict': self.actor.state_dict(),
                'critic_summarizer_state_dict': self.critic_summarizer.state_dict(),
                'critic_state_dict': self.critic.state_dict(),
                'critic_summarizer_target_state_dict': self.critic_summarizer_target.state_dict(),
                'critic_target_state_dict': self.critic_target.state_dict(),
                'actor_summarizer_optimizer_state_dict': self.actor_summarizer_optimizer.state_dict(),
                'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
                'critic_summarizer_optimizer_state_dict': self.critic_summarizer_optimizer.state_dict(),
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
            checkpoint = torch.load(filename, map_location=self.device)

            self.actor_summarizer.load_state_dict(checkpoint['actor_summarizer_state_dict'])
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic_summarizer.load_state_dict(checkpoint['critic_summarizer_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_summarizer_target.load_state_dict(checkpoint['critic_summarizer_target_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])

            self.actor_summarizer_optimizer.load_state_dict(checkpoint['actor_summarizer_optimizer_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.critic_summarizer_optimizer.load_state_dict(checkpoint['critic_summarizer_optimizer_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

            self.log_alpha = checkpoint['log_alpha']
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
            self.training_steps = checkpoint['training_steps']

            self.reset_hidden()

            print(f"Model loaded from {filename}")
            print(f"Training steps: {self.training_steps}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False