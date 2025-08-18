import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from multiprocessing import Process, Queue, Event
import copy
import matplotlib.pyplot as plt

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Configuration class for SAC hyperparameters
class SACConfig:
    # Network architecture
    cnn_features = 256
    hidden_dim = 256
    gru_hidden_dim = 256
    gru_layers = 1

    # SAC hyperparameters
    gamma = 0.99
    tau = 0.005
    alpha = 0.2
    learning_rate = 3e-4

    # Experience replay
    buffer_size = 100000
    batch_size = 64
    sequence_length = 8

    # Training
    update_every = 1

    # Action space
    action_dim = 3
    action_scale = 5.0

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# CNN Feature Extractor (as provided)
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
        # Input shape validation
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


# Recurrent Actor Network
class RecurrentActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, gru_hidden_dim, gru_layers):
        super().__init__()
        self.cnn = CNNFeatureExtractor()
        self.state_fc = nn.Linear(state_dim, hidden_dim)
        self.gru = nn.GRU(
            input_size=SACConfig.cnn_features + hidden_dim,
            hidden_size=gru_hidden_dim,
            num_layers=gru_layers,
            batch_first=True
        )
        self.mean_fc = nn.Linear(gru_hidden_dim, action_dim)
        self.log_std_fc = nn.Linear(gru_hidden_dim, action_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, depth, state, hidden=None):
        # ensure contiguity for RNN
        self.gru.flatten_parameters()
        img_feat = self.cnn(depth)
        state_feat = F.relu(self.state_fc(state))
        combo = torch.cat([img_feat, state_feat], dim=-1)
        if combo.dim() == 2:
            combo = combo.unsqueeze(1)
        output, hidden = self.gru(combo, hidden)
        mean = self.mean_fc(output)
        log_std = torch.clamp(self.log_std_fc(output), -20, 2)
        return mean, log_std, hidden

    def sample(self, depth, state, hidden=None):
        mean, log_std, hidden = self.forward(depth, state, hidden)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        scaled = action * SACConfig.action_scale
        # detach hidden to prevent long backprop graph
        return scaled, log_prob, hidden.detach()


# Recurrent Critic Network (Q-function)
class RecurrentCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, gru_hidden_dim, gru_layers):
        super().__init__()
        self.cnn = CNNFeatureExtractor()
        self.state_fc = nn.Linear(state_dim, hidden_dim)
        self.action_fc = nn.Linear(action_dim, hidden_dim)
        self.gru = nn.GRU(
            input_size=SACConfig.cnn_features + hidden_dim*2,
            hidden_size=gru_hidden_dim,
            num_layers=gru_layers,
            batch_first=True
        )
        self.q_fc = nn.Linear(gru_hidden_dim, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, depth, state, action, hidden=None):
        self.gru.flatten_parameters()
        img_feat = self.cnn(depth)
        s_feat = F.relu(self.state_fc(state))
        a_feat = F.relu(self.action_fc(action))
        combo = torch.cat([img_feat, s_feat, a_feat], dim=-1)
        if combo.dim() == 2:
            combo = combo.unsqueeze(1)
        out, hidden = self.gru(combo, hidden)
        q = self.q_fc(out)
        return (q.squeeze(1), hidden.detach()) if q.dim()==3 else (q, hidden.detach())


# Recurrent Replay Buffer for sequential data
class RecurrentReplayBuffer:
    def __init__(self, buffer_size, batch_size, sequence_length, device):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.device = device

        # Storage for transitions
        self.depth_images = []
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_depth_images = []
        self.next_states = []
        self.dones = []

        # Episode boundaries (end indices of episodes)
        self.episode_boundaries = []

        # Current episode buffer
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
        """Add experience to current episode buffer."""
        self.current_episode['depth_images'].append(depth_image)
        self.current_episode['states'].append(state)
        self.current_episode['actions'].append(action)
        self.current_episode['rewards'].append(reward)
        self.current_episode['next_depth_images'].append(next_depth_image)
        self.current_episode['next_states'].append(next_state)
        self.current_episode['dones'].append(done)

        # If episode ends, store the episode in the main buffer
        if done:
            self._store_episode()

    def _store_episode(self):
        """Store the current episode in the main buffer."""
        episode_length = len(self.current_episode['states'])

        if episode_length > 0:
            # Add episode to buffer
            self.depth_images.extend(self.current_episode['depth_images'])
            self.states.extend(self.current_episode['states'])
            self.actions.extend(self.current_episode['actions'])
            self.rewards.extend(self.current_episode['rewards'])
            self.next_depth_images.extend(self.current_episode['next_depth_images'])
            self.next_states.extend(self.current_episode['next_states'])
            self.dones.extend(self.current_episode['dones'])

            # Record the end index of this episode
            if not self.episode_boundaries:
                self.episode_boundaries.append(episode_length - 1)
            else:
                self.episode_boundaries.append(self.episode_boundaries[-1] + episode_length)

            # Limit buffer size by removing oldest experiences if needed
            if len(self.states) > self.buffer_size:
                overflow = len(self.states) - self.buffer_size

                # Remove oldest data
                self.depth_images = self.depth_images[overflow:]
                self.states = self.states[overflow:]
                self.actions = self.actions[overflow:]
                self.rewards = self.rewards[overflow:]
                self.next_depth_images = self.next_depth_images[overflow:]
                self.next_states = self.next_states[overflow:]
                self.dones = self.dones[overflow:]

                # Update episode boundaries
                while self.episode_boundaries and self.episode_boundaries[0] < overflow:
                    self.episode_boundaries.pop(0)

                # Adjust remaining episode boundaries
                self.episode_boundaries = [b - overflow for b in self.episode_boundaries]

        # Reset current episode buffer
        self.current_episode = {
            'depth_images': [],
            'states': [],
            'actions': [],
            'rewards': [],
            'next_depth_images': [],
            'next_states': [],
            'dones': []
        }

    def sample(self):
        """Sample a batch of sequences."""
        batch_tensors = {
            'depth_images': [],
            'states': [],
            'actions': [],
            'rewards': [],
            'next_depth_images': [],
            'next_states': [],
            'dones': [],
            'masks': []  # For handling variable-length sequences
        }

        # Get episode start indices
        episode_start_indices = [0] + [b + 1 for b in self.episode_boundaries[:-1]] if self.episode_boundaries else [0]

        # Randomly sample episodes
        sampled_episodes = random.choices(range(len(episode_start_indices)), k=self.batch_size)

        for episode_idx in sampled_episodes:
            # Determine episode boundaries
            start_idx = episode_start_indices[episode_idx]
            end_idx = self.episode_boundaries[episode_idx]
            episode_length = end_idx - start_idx + 1

            # Select sequence from episode
            if episode_length <= self.sequence_length:
                # Episode is shorter than sequence length
                seq_start_idx = start_idx
                seq_length = episode_length
            else:
                # Randomly select a starting point
                max_start_idx = start_idx + episode_length - self.sequence_length
                seq_start_idx = random.randint(start_idx, max_start_idx)
                seq_length = self.sequence_length

            # Extract sequence
            seq_depth_images = self.depth_images[seq_start_idx:seq_start_idx + seq_length]
            seq_states = self.states[seq_start_idx:seq_start_idx + seq_length]
            seq_actions = self.actions[seq_start_idx:seq_start_idx + seq_length]
            seq_rewards = self.rewards[seq_start_idx:seq_start_idx + seq_length]
            seq_next_depth_images = self.next_depth_images[seq_start_idx:seq_start_idx + seq_length]
            seq_next_states = self.next_states[seq_start_idx:seq_start_idx + seq_length]
            seq_dones = self.dones[seq_start_idx:seq_start_idx + seq_length]

            # Create mask (1 for actual data, 0 for padding)
            seq_mask = [1.0] * seq_length

            # Pad sequences if needed
            if seq_length < self.sequence_length:
                # Calculate padding needed
                pad_length = self.sequence_length - seq_length

                # Create padding values
                depth_img_shape = seq_depth_images[0].shape
                zero_depth_img = np.zeros(depth_img_shape)

                state_shape = seq_states[0].shape
                zero_state = np.zeros(state_shape)

                action_shape = seq_actions[0].shape
                zero_action = np.zeros(action_shape)

                # Apply padding
                seq_depth_images.extend([zero_depth_img] * pad_length)
                seq_states.extend([zero_state] * pad_length)
                seq_actions.extend([zero_action] * pad_length)
                seq_rewards.extend([0.0] * pad_length)
                seq_next_depth_images.extend([zero_depth_img] * pad_length)
                seq_next_states.extend([zero_state] * pad_length)
                seq_dones.extend([1.0] * pad_length)
                seq_mask.extend([0.0] * pad_length)

            # Add to batch
            batch_tensors['depth_images'].append(seq_depth_images)
            batch_tensors['states'].append(seq_states)
            batch_tensors['actions'].append(seq_actions)
            batch_tensors['rewards'].append(seq_rewards)
            batch_tensors['next_depth_images'].append(seq_next_depth_images)
            batch_tensors['next_states'].append(seq_next_states)
            batch_tensors['dones'].append(seq_dones)
            batch_tensors['masks'].append(seq_mask)

        # Convert to tensors
        depth_tensors = self._process_depth_images(batch_tensors['depth_images'])
        next_depth_tensors = self._process_depth_images(batch_tensors['next_depth_images'])

        # Process other tensors
        processed_tensors = {
            'depth_images': depth_tensors,
            'states': torch.FloatTensor(batch_tensors['states']).to(self.device),
            'actions': torch.FloatTensor(batch_tensors['actions']).to(self.device),
            'rewards': torch.FloatTensor(batch_tensors['rewards']).unsqueeze(-1).to(self.device),
            'next_depth_images': next_depth_tensors,
            'next_states': torch.FloatTensor(batch_tensors['next_states']).to(self.device),
            'dones': torch.FloatTensor(batch_tensors['dones']).unsqueeze(-1).to(self.device),
            'masks': torch.FloatTensor(batch_tensors['masks']).to(self.device)
        }

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

    def _process_depth_images(self, depth_images):
        """Convert depth images list to proper tensor format."""
        batch_size = len(depth_images)
        seq_len = len(depth_images[0])

        # Get depth image dimensions
        height, width = depth_images[0][0].shape

        # Create tensor [batch, seq_len, channels=1, height, width]
        tensor = torch.zeros(batch_size, seq_len, 1, height, width, device=self.device)

        # Fill tensor with depth image data
        for b in range(batch_size):
            for s in range(seq_len):
                tensor[b, s, 0] = torch.FloatTensor(depth_images[b][s])

        return tensor

    def __len__(self):
        """Return the current size of the buffer."""
        return len(self.states)


# Main Recurrent SAC Agent
class RecurrentSAC:
    def __init__(self, state_dim, action_dim=SACConfig.action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = SACConfig.device

        # Initialize actor network
        self.actor = RecurrentActor(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=SACConfig.hidden_dim,
            gru_hidden_dim=SACConfig.gru_hidden_dim,
            gru_layers=SACConfig.gru_layers
        ).to(self.device)

        # Initialize two critic networks (twin critics)
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

        # Initialize target networks
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)

        # Freeze target networks with respect to optimizers
        for param in self.critic1_target.parameters():
            param.requires_grad = False

        for param in self.critic2_target.parameters():
            param.requires_grad = False

        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=SACConfig.learning_rate)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=SACConfig.learning_rate)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=SACConfig.learning_rate)

        # Initialize entropy coefficient (alpha)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = SACConfig.alpha
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=SACConfig.learning_rate)

        # Initialize target entropy
        self.target_entropy = -torch.prod(torch.Tensor([action_dim]).to(self.device)).item()

        # Initialize hidden states
        self.actor_hidden = None
        self.critic1_hidden = None
        self.critic2_hidden = None

        # Initialize replay buffer
        self.replay_buffer = RecurrentReplayBuffer(
            buffer_size=SACConfig.buffer_size,
            batch_size=SACConfig.batch_size,
            sequence_length=SACConfig.sequence_length,
            device=self.device
        )

        # Training step counter
        self.steps = 0

    def reset_hidden_states(self, batch_size=1):
        """Reset hidden states at the beginning of an episode."""
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
        """Select action using the policy."""
        with torch.no_grad():
            # Pre-process inputs
            depth_tensor = torch.FloatTensor(depth_image).unsqueeze(0).unsqueeze(0).to(self.device)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            if evaluate:
                # Use deterministic policy for evaluation
                mean, _, self.actor_hidden = self.actor(depth_tensor, state_tensor, self.actor_hidden)
                action = torch.tanh(mean) * SACConfig.action_scale
                return action.cpu().data.numpy().flatten()
            else:
                # Sample from policy for training
                action, _, self.actor_hidden = self.actor.sample(depth_tensor, state_tensor, self.actor_hidden)
                return action.cpu().data.numpy().flatten()

    def store_transition(self, depth_image, state, action, reward, next_depth_image, next_state, done):
        """Store transition in replay buffer."""
        self.replay_buffer.add(depth_image, state, action, reward, next_depth_image, next_state, done)

    def train(self):
        """Update the policy and value parameters."""
        self.steps += 1

        # Only update at specified frequency
        if self.steps % SACConfig.update_every != 0:
            return 0, 0, 0

        # Check if enough samples in buffer
        if len(self.replay_buffer) < SACConfig.batch_size:
            return 0, 0, 0

        # Sample batch from replay buffer
        depth_imgs, states, actions, rewards, next_depth_imgs, next_states, dones, masks = self.replay_buffer.sample()

        # Update critic networks
        critic_loss = self._update_critics(depth_imgs, states, actions, rewards, next_depth_imgs, next_states, dones,
                                           masks)

        # Update actor network and entropy coefficient
        actor_loss, alpha_loss = self._update_actor_and_alpha(depth_imgs, states, masks)

        # Soft update target networks
        self._soft_update_targets()

        return critic_loss, actor_loss, alpha_loss

    def _update_critics(self, depth_imgs, states, actions, rewards, next_depth_imgs, next_states, dones, masks):
        """Update critic networks."""
        batch_size, seq_len = states.shape[0], states.shape[1]

        # Initialize hidden states
        critic1_hidden = torch.zeros(SACConfig.gru_layers, batch_size, SACConfig.gru_hidden_dim, device=self.device)
        critic2_hidden = torch.zeros(SACConfig.gru_layers, batch_size, SACConfig.gru_hidden_dim, device=self.device)
        actor_hidden = torch.zeros(SACConfig.gru_layers, batch_size, SACConfig.gru_hidden_dim, device=self.device)

        with torch.no_grad():
            # Process each timestep in the sequence
            next_actions = []
            next_log_probs = []

            for t in range(seq_len):
                # Get data for current timestep
                next_depth_t = next_depth_imgs[:, t:t + 1]  # Keep sequence dimension
                next_state_t = next_states[:, t]
                mask_t = masks[:, t].unsqueeze(-1)

                # Sample actions and log probs for next state
                next_action_t, next_log_prob_t, actor_hidden = self.actor.sample(
                    next_depth_t, next_state_t, actor_hidden
                )

                # Apply mask to zero out padding
                next_action_t = next_action_t * mask_t
                next_log_prob_t = next_log_prob_t * mask_t

                next_actions.append(next_action_t)
                next_log_probs.append(next_log_prob_t)

            # Stack all timesteps
            next_actions = torch.stack(next_actions, dim=1)
            next_log_probs = torch.stack(next_log_probs, dim=1)

            # Calculate target Q-values
            target_q1_values = []
            target_q2_values = []

            critic1_target_hidden = torch.zeros(SACConfig.gru_layers, batch_size, SACConfig.gru_hidden_dim,
                                                device=self.device)
            critic2_target_hidden = torch.zeros(SACConfig.gru_layers, batch_size, SACConfig.gru_hidden_dim,
                                                device=self.device)

            for t in range(seq_len):
                next_depth_t = next_depth_imgs[:, t:t + 1]
                next_state_t = next_states[:, t]
                next_action_t = next_actions[:, t]
                mask_t = masks[:, t].unsqueeze(-1)

                # Get Q-values from target critics
                target_q1_t, critic1_target_hidden = self.critic1_target(
                    next_depth_t, next_state_t, next_action_t, critic1_target_hidden
                )

                target_q2_t, critic2_target_hidden = self.critic2_target(
                    next_depth_t, next_state_t, next_action_t, critic2_target_hidden
                )

                # Apply mask
                target_q1_t = target_q1_t * mask_t
                target_q2_t = target_q2_t * mask_t

                target_q1_values.append(target_q1_t)
                target_q2_values.append(target_q2_t)

            # Stack timesteps
            target_q1_values = torch.stack(target_q1_values, dim=1)
            target_q2_values = torch.stack(target_q2_values, dim=1)

            # Take minimum of two Q-values (clipped double Q-learning)
            target_q_values = torch.min(target_q1_values, target_q2_values)

            # Subtract entropy term
            target_q_values = target_q_values - self.alpha * next_log_probs

            # Calculate TD targets
            targets = rewards + (1 - dones) * SACConfig.gamma * target_q_values

        # Calculate current Q-values
        current_q1_values = []
        current_q2_values = []

        for t in range(seq_len):
            depth_t = depth_imgs[:, t:t + 1]
            state_t = states[:, t]
            action_t = actions[:, t]
            mask_t = masks[:, t].unsqueeze(-1)

            # Get Q-values from critics
            q1_t, critic1_hidden = self.critic1(depth_t, state_t, action_t, critic1_hidden)
            q2_t, critic2_hidden = self.critic2(depth_t, state_t, action_t, critic2_hidden)

            # Apply mask
            q1_t = q1_t * mask_t
            q2_t = q2_t * mask_t

            current_q1_values.append(q1_t)
            current_q2_values.append(q2_t)

        # Stack timesteps
        current_q1_values = torch.stack(current_q1_values, dim=1)
        current_q2_values = torch.stack(current_q2_values, dim=1)

        # Calculate MSE loss for critics
        critic1_loss = F.mse_loss(current_q1_values, targets)
        critic2_loss = F.mse_loss(current_q2_values, targets)
        critic_loss = critic1_loss + critic2_loss

        # Update critics
        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()

        return critic_loss.item()

    def _update_actor_and_alpha(self, depth_imgs, states, masks):
        """Update actor network and entropy coefficient."""
        batch_size, seq_len = states.shape[0], states.shape[1]

        # Initialize hidden states
        actor_hidden = torch.zeros(SACConfig.gru_layers, batch_size, SACConfig.gru_hidden_dim, device=self.device)
        critic1_hidden = torch.zeros(SACConfig.gru_layers, batch_size, SACConfig.gru_hidden_dim, device=self.device)
        critic2_hidden = torch.zeros(SACConfig.gru_layers, batch_size, SACConfig.gru_hidden_dim, device=self.device)

        actor_losses = []
        log_probs_list = []

        for t in range(seq_len):
            depth_t = depth_imgs[:, t:t + 1]
            state_t = states[:, t]
            mask_t = masks[:, t].unsqueeze(-1)

            # Sample actions from policy
            action_t, log_prob_t, actor_hidden = self.actor.sample(depth_t, state_t, actor_hidden)

            # Get Q-values for sampled actions
            q1_t, critic1_hidden = self.critic1(depth_t, state_t, action_t, critic1_hidden)
            q2_t, critic2_hidden = self.critic2(depth_t, state_t, action_t, critic2_hidden)

            # Take minimum Q-value
            min_q_t = torch.min(q1_t, q2_t)

            # Calculate actor loss (Policy gradient with entropy regularization)
            actor_loss_t = (self.alpha * log_prob_t - min_q_t) * mask_t

            actor_losses.append(actor_loss_t)
            log_probs_list.append(log_prob_t * mask_t)

        # Average across sequence
        actor_loss = torch.stack(actor_losses, dim=1).mean()

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update alpha (temperature parameter)
        log_probs = torch.stack(log_probs_list, dim=1)
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Update alpha value
        self.alpha = self.log_alpha.exp().item()

        return actor_loss.item(), alpha_loss.item()

    def _soft_update_targets(self):
        """Soft update target networks."""
        tau = SACConfig.tau

        # Update critic1 target
        for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        # Update critic2 target
        for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save(self, directory, filename="rsac"):
        """Save model weights."""
        if not os.path.exists(directory):
            os.makedirs(directory)

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
        }, os.path.join(directory, f"{filename}.pt"))

    def load(self, filepath):
        ckpt = torch.load(filepath)
        self.actor.load_state_dict(ckpt['actor'])
        self.critic1.load_state_dict(ckpt['critic1'])
        self.critic2.load_state_dict(ckpt['critic2'])
        self.critic1_target.load_state_dict(ckpt['critic1_target'])
        self.critic2_target.load_state_dict(ckpt['critic2_target'])
        # Properly restore log_alpha without losing grad
        self.log_alpha.data.copy_(ckpt['log_alpha'].data)
        self.alpha = self.log_alpha.exp().item()
        self.actor_optimizer.load_state_dict(ckpt['actor_optimizer'])
        self.critic1_optimizer.load_state_dict(ckpt['critic1_optimizer'])
        self.critic2_optimizer.load_state_dict(ckpt['critic2_optimizer'])
        self.alpha_optimizer.load_state_dict(ckpt['alpha_optimizer'])


# Training function
import torch.multiprocessing as mp
from multiprocessing.managers import BaseManager
import time


# 모델 관리자 정의
class ModelManager:
    def __init__(self, agent):
        self.agent = agent
        self.update_counter = 0
        self.lock = mp.Lock()

    def get_weights(self):
        with self.lock:
            return {
                'actor': self.agent.actor.state_dict(),
                'critic1': self.agent.critic1.state_dict(),
                'critic2': self.agent.critic2.state_dict(),
                'log_alpha': self.agent.log_alpha,
                'counter': self.update_counter
            }

    def update_weights(self, weights_dict):
        with self.lock:
            self.agent.actor.load_state_dict(weights_dict['actor'])
            self.agent.critic1.load_state_dict(weights_dict['critic1'])
            self.agent.critic2.load_state_dict(weights_dict['critic2'])
            self.agent.log_alpha.data.copy_(weights_dict['log_alpha'].data)
            self.agent.alpha = self.agent.log_alpha.exp().item()
            self.update_counter += 1


# 매니저 등록
BaseManager.register('ModelManager', ModelManager)


def data_collector(model_manager, queue, stop_event):
    """수집 에이전트 - 최신 모델로 환경과 상호작용하며 데이터 수집"""
    # 로컬 에이전트 초기화
    local_agent = RecurrentSAC(state_dim=model_manager.agent.state_dim)
    last_update = -1

    env = DroneEnv()  # 환경 생성

    while not stop_event.is_set():
        # 주기적으로 모델 가중치 업데이트 확인 (예: 5 step마다)
        weights = model_manager.get_weights()
        if weights['counter'] > last_update:
            del weights['counter']
            local_agent.actor.load_state_dict(weights['actor'])
            local_agent.critic1.load_state_dict(weights['critic1'])
            local_agent.critic2.load_state_dict(weights['critic2'])
            local_agent.log_alpha.data.copy_(weights['log_alpha'].data)
            local_agent.alpha = local_agent.log_alpha.exp().item()
            last_update = model_manager.get_weights()['counter']

        # 환경과 상호작용
        depth, state = env.reset()
        local_agent.reset_hidden_states()
        done = False

        while not done and not stop_event.is_set():
            action = local_agent.select_action(depth, state)
            next_depth, next_state, reward, done, _ = env.step(action)
            queue.put((depth, state, action, reward, next_depth, next_state, done))
            depth, state = next_depth, next_state


def learner(model_manager, queue, stop_event):
    """학습 에이전트 - 데이터로 모델 학습 및 중앙 모델 가중치 업데이트"""
    agent = model_manager.agent
    update_freq = 100  # 가중치 업데이트 빈도
    steps = 0

    while not stop_event.is_set():
        # 큐에서 데이터 가져오기
        transitions = []
        for _ in range(min(SACConfig.batch_size, queue.qsize())):
            if not queue.empty():
                transitions.append(queue.get())
            else:
                time.sleep(0.01)  # 데이터 기다리기

        if not transitions:
            continue

        # 리플레이 버퍼에 저장
        for transition in transitions:
            agent.store_transition(*transition)

        # 모델 학습
        if len(agent.replay_buffer) >= SACConfig.batch_size:
            agent.train()
            steps += 1

            # 주기적으로 모델 가중치 업데이트
            if steps % update_freq == 0:
                weights = {
                    'actor': agent.actor.state_dict(),
                    'critic1': agent.critic1.state_dict(),
                    'critic2': agent.critic2.state_dict(),
                    'log_alpha': agent.log_alpha
                }
                model_manager.update_weights(weights)


def train_async(env, agent, run_seconds=600, n_collectors=2):
    """비동기 학습 실행"""
    mp.set_start_method('spawn', force=True)  # 윈도우 환경 호환성

    # 모델 매니저 설정
    manager = BaseManager()
    manager.start()
    model_manager = manager.ModelManager(agent)

    # 큐와 이벤트 설정
    queue = mp.Queue(maxsize=10000)
    stop_event = mp.Event()

    # 프로세스 생성
    collectors = []
    for _ in range(n_collectors):
        p = mp.Process(target=data_collector, args=(model_manager, queue, stop_event))
        collectors.append(p)
        p.start()

    learner_process = mp.Process(target=learner, args=(model_manager, queue, stop_event))
    learner_process.start()

    # 지정된 시간 동안 실행
    try:
        time.sleep(run_seconds)
    finally:
        stop_event.set()
        for p in collectors:
            p.join()
        learner_process.join()

    # 최종 모델 가중치 가져오기
    final_weights = model_manager.get_weights()
    del final_weights['counter']
    agent.actor.load_state_dict(final_weights['actor'])
    agent.critic1.load_state_dict(final_weights['critic1'])
    agent.critic2.load_state_dict(final_weights['critic2'])
    agent.log_alpha.data.copy_(final_weights['log_alpha'].data)
    agent.alpha = agent.log_alpha.exp().item()

    return agent

# ---------------------
# Usage Example
# ---------------------
if __name__ == "__main__":
    from Environment.Env import DroneEnv, Config
    env    = DroneEnv()
    _,state=env.reset()
    agent  = RecurrentSAC(state_dim=len(state))
    # Asynchronous train for 10 minutes
    train_async(env, agent, run_seconds=600)
