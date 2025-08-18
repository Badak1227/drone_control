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


# Configuration for SAC
class SACConfig:
    # Environment
    max_episode_steps = 1000
    goal_threshold = 2.0  # Distance to goal considered as reached
    seq_length = 10  # Minimum steps before allowing termination on collision

    # Neural Network
    hidden_dim = 256
    cnn_features = 64

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
    def __init__(self, input_channels=4, output_dim=SACConfig.cnn_features):
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

        # Calculate feature dimensions after convolutions for a 84x84 input
        # After conv1 + maxpool1: 21x21x32
        # After conv2 + maxpool2: 10x10x64
        # After conv3: 10x10x128
        # After conv4: 10x10x128

        # Fully connected layer
        self.fc = nn.Linear(5 * 5 * 128, output_dim)

    def forward(self, x):
        # Make sure input has the right shape for a single image (add batch and channel dims if needed)
        if len(x.shape) == 3:
            x = x.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        elif len(x.shape) == 3:
            x = x.unsqueeze(1)  # Add channel dimension

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


# Actor Network
class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, max_action=1.0):
        super(ActorNetwork, self).__init__()

        # Process depth image with CNN
        self.cnn = CNNFeatureExtractor(input_channels=4)

        # Process state vector with FC layers
        self.fc1 = nn.Linear(state_dim + SACConfig.cnn_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Output mean and log_std for each action
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

        self.max_action = max_action
        self.action_dim = action_dim

    def forward(self, depth_image, state):
        # Extract features from depth image
        image_features = self.cnn(depth_image)

        # Combine image features with state
        combined = torch.cat([image_features, state], dim=1)

        # Process through FC layers
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))

        # Calculate mean and log_std
        mean = self.mean(x)
        log_std = self.log_std(x)

        # Constrain log_std to prevent extremely large/small values
        log_std = torch.clamp(log_std, min=-20, max=2)

        return mean, log_std

    def sample(self, depth_image, state):
        mean, log_std = self.forward(depth_image, state)
        std = log_std.exp()

        # Create normal distribution
        normal = Normal(mean, std)

        # Sample action from distribution
        x_t = normal.rsample()  # Reparameterization trick

        # Calculate log probabilities
        log_prob = normal.log_prob(x_t).sum(dim=1, keepdim=True)

        # Apply tanh squashing to constrain actions
        action = torch.tanh(x_t)

        # Correct log probabilities for the squashing transformation
        log_prob -= torch.sum(torch.log(1 - action.pow(2) + 1e-6), dim=1, keepdim=True)

        # Scale actions to our desired range
        action = action * self.max_action

        return action, log_prob, torch.tanh(mean) * self.max_action


# Critic Network
class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(CriticNetwork, self).__init__()

        # Process depth image with CNN (shared between both Q networks)
        self.cnn = CNNFeatureExtractor(input_channels=4)

        # Q1 network
        self.q1_fc1 = nn.Linear(state_dim + action_dim + SACConfig.cnn_features, hidden_dim)
        self.q1_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_out = nn.Linear(hidden_dim, 1)

        # Q2 network
        self.q2_fc1 = nn.Linear(state_dim + action_dim + SACConfig.cnn_features, hidden_dim)
        self.q2_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_out = nn.Linear(hidden_dim, 1)

    def forward(self, depth_image, state, action):
        # Extract features from depth image
        image_features = self.cnn(depth_image)

        # Combine image features with state and action
        combined = torch.cat([image_features, state, action], dim=1)

        # Q1 network
        q1 = F.relu(self.q1_fc1(combined))
        q1 = F.relu(self.q1_fc2(q1))
        q1 = self.q1_out(q1)

        # Q2 network
        q2 = F.relu(self.q2_fc1(combined))
        q2 = F.relu(self.q2_fc2(q2))
        q2 = self.q2_out(q2)

        return q1, q2

    # For use when only need one Q value (e.g., during action selection)
    def q1(self, depth_image, state, action):
        image_features = self.cnn(depth_image)
        combined = torch.cat([image_features, state, action], dim=1)

        q1 = F.relu(self.q1_fc1(combined))
        q1 = F.relu(self.q1_fc2(q1))
        q1 = self.q1_out(q1)

        return q1


# Dual Experience Replay Buffer (Success and Failure buffers)
class DualExperienceBuffer:
    def __init__(self, success_buffer_size, fail_buffer_size, state_dim, action_dim):
        self.success_buffer = deque(maxlen=success_buffer_size)
        self.fail_buffer = deque(maxlen=fail_buffer_size)

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.experience = namedtuple("Experience",
                                     field_names=["depth_image", "state", "action", "reward", "next_depth_image",
                                                  "next_state", "done", "is_success"])

    def add(self, depth_image, state, action, reward, next_depth_image, next_state, done, is_success):
        # Create experience tuple
        e = self.experience(depth_image, state, action, reward, next_depth_image, next_state, done, is_success)

        # Add to appropriate buffer
        if is_success:
            self.success_buffer.append(e)
        else:
            self.fail_buffer.append(e)

    def sample(self, batch_size, success_ratio=0.4):
        # Determine number of samples from each buffer
        success_samples = int(batch_size * success_ratio)
        fail_samples = batch_size - success_samples

        # Adjust if either buffer has insufficient samples
        if len(self.success_buffer) < success_samples:
            success_samples = len(self.success_buffer)
            fail_samples = batch_size - success_samples

        if len(self.fail_buffer) < fail_samples:
            fail_samples = len(self.fail_buffer)
            success_samples = batch_size - fail_samples

        # Sample from each buffer
        success_experiences = random.sample(self.success_buffer, min(success_samples, len(self.success_buffer))) if len(
            self.success_buffer) > 0 else []
        fail_experiences = random.sample(self.fail_buffer, min(fail_samples, len(self.fail_buffer))) if len(
            self.fail_buffer) > 0 else []

        # Combine samples
        experiences = success_experiences + fail_experiences
        random.shuffle(experiences)

        # Check if we have any experiences
        if len(experiences) == 0:
            return None, None, None, None, None, None, None

        # Convert to tensors
        depth_images = torch.from_numpy(np.array([e.depth_image for e in experiences if e is not None])).float()
        states = torch.from_numpy(np.array([e.state for e in experiences if e is not None])).float()
        actions = torch.from_numpy(np.array([e.action for e in experiences if e is not None])).float()
        rewards = torch.from_numpy(np.array([e.reward for e in experiences if e is not None])).float().unsqueeze(1)
        next_depth_images = torch.from_numpy(
            np.array([e.next_depth_image for e in experiences if e is not None])).float()
        next_states = torch.from_numpy(np.array([e.next_state for e in experiences if e is not None])).float()
        dones = torch.from_numpy(
            np.array([e.done for e in experiences if e is not None]).astype(np.uint8)).float().unsqueeze(1)

        return depth_images, states, actions, rewards, next_depth_images, next_states, dones

    def __len__(self):
        return len(self.success_buffer) + len(self.fail_buffer)

    def success_buffer_len(self):
        return len(self.success_buffer)

    def fail_buffer_len(self):
        return len(self.fail_buffer)


# SAC Agent
class SACAgent:
    def __init__(self, state_dim, action_dim, max_action=1.0, device="cpu"):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.device = device

        # Initialize actor network
        self.actor = ActorNetwork(state_dim, action_dim, SACConfig.hidden_dim, max_action).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=SACConfig.lr_actor)

        # Initialize critic networks
        self.critic = CriticNetwork(state_dim, action_dim, SACConfig.hidden_dim).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=SACConfig.lr_critic)

        # Initialize target critic network
        self.critic_target = CriticNetwork(state_dim, action_dim, SACConfig.hidden_dim).to(device)
        # Copy weights from critic to target
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        # Initialize temperature parameter alpha (for entropy)
        self.log_alpha = torch.tensor(np.log(SACConfig.alpha_init)).to(device)
        self.log_alpha.requires_grad = True
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=SACConfig.lr_alpha)

        # Initialize target entropy
        self.target_entropy = SACConfig.target_entropy

        # Initialize dual experience buffer
        self.memory = DualExperienceBuffer(
            SACConfig.success_buffer_size,
            SACConfig.fail_buffer_size,
            state_dim,
            action_dim
        )

        # Training step counter
        self.training_steps = 0

    def select_action(self, depth_image, state, evaluate=False):
        # Convert to tensor and add batch dimension if needed
        depth_image = torch.FloatTensor(depth_image).to(self.device)
        state = torch.FloatTensor(state).to(self.device)

        if len(depth_image.shape) == 3:
            depth_image = depth_image.unsqueeze(0)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)

        # During evaluation, we use the mean action
        if evaluate:
            _, _, action = self.actor.sample(depth_image, state)
        else:
            action, _, _ = self.actor.sample(depth_image, state)

        return action.detach().cpu().numpy()[0]

    def update_parameters(self, delayed_update=True):
        # Only update if we have enough samples
        if len(self.memory) < SACConfig.batch_size:
            return None

        # Increment counter
        self.training_steps += 1

        # If using delayed update and it's not time yet, skip
        if delayed_update and self.training_steps % SACConfig.update_interval != 0:
            return None

        # Sample from replay buffer
        sampled_data = self.memory.sample(SACConfig.batch_size, SACConfig.sample_success_ratio)

        # Check if we have data (might happen if buffer is empty)
        if sampled_data[0] is None:
            return None

        depth_images, states, actions, rewards, next_depth_images, next_states, dones = sampled_data

        # Move to device
        depth_images = depth_images.to(self.device)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_depth_images = next_depth_images.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Current temperature parameter
        alpha = self.log_alpha.exp()

        # ---------- Update Critic ----------
        with torch.no_grad():
            # Sample actions from target policy
            next_actions, next_log_probs, _ = self.actor.sample(next_depth_images, next_states)

            # Compute target Q values
            target_q1, target_q2 = self.critic_target(next_depth_images, next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - alpha * next_log_probs
            target_q = rewards + (1 - dones) * SACConfig.gamma * target_q

        # Compute current Q values
        current_q1, current_q2 = self.critic(depth_images, states, actions)

        # Compute critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------- Update Actor ----------
        # Sample actions from current policy
        actions_new, log_probs_new, _ = self.actor.sample(depth_images, states)

        # Compute actor loss
        q1_new, q2_new = self.critic(depth_images, states, actions_new)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (alpha * log_probs_new - q_new).mean()

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ---------- Update Temperature ----------
        alpha_loss = -(self.log_alpha * (log_probs_new.detach() + self.target_entropy)).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # ---------- Soft Update Target Networks ----------
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

    def load(self, filename):
        checkpoint = torch.load(filename, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.log_alpha = checkpoint['log_alpha']
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
        self.training_steps = checkpoint['training_steps']