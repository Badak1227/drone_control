import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import time
from datetime import datetime

# Import your environment
from Environment.Env import DroneEnv, Config

# Import the SAC agent
from Networks.sac import SACAgent, SACConfig


def train_agent(env, agent, num_episodes, max_steps=SACConfig.max_episode_steps,
                render=False, start_steps=SACConfig.start_steps, log_dir="logs"):
    """
    Train the SAC agent in the drone environment

    Args:
        env: The drone environment
        agent: The SAC agent
        num_episodes: Number of episodes to train
        max_steps: Maximum steps per episode
        render: Whether to render the environment during training
        start_steps: Number of steps to take random actions for exploration
        log_dir: Directory to save logs

    Returns:
        List of episode rewards
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Metrics for tracking progress
    episode_rewards = []
    success_rate_window = []
    collision_rate_window = []
    step_counts = []
    losses = []

    # Log file
    log_file = os.path.join(log_dir, f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(log_file, "w") as f:
        f.write("Episode,Reward,Steps,Status,SuccessRate,CollisionRate,AvgSteps\n")

    for episode in range(num_episodes):
        episode_reward = 0
        episode_loss = []
        step_count = 0

        # Reset environment
        depth_image, drone_state = env.reset()

        for step in range(max_steps):
            # In early steps, take random actions for exploration
            if len(agent.memory) < start_steps:
                action = np.random.uniform(-agent.max_action, agent.max_action, size=agent.action_dim)
            else:
                action = agent.select_action(depth_image, drone_state)

            # Take action in environment
            next_depth_image, next_drone_state, reward, done, info = env.step(action)

            # Determine if this is a success or collision
            is_success = info.get('status', '') == 'goal_reached'
            is_collision = info.get('status', '') == 'collision'

            # Store transition in replay buffer
            agent.memory.add(
                depth_image, drone_state, action, reward,
                next_depth_image, next_drone_state, float(done), is_success
            )

            # Update agent if enough samples are available
            if len(agent.memory) >= SACConfig.batch_size:
                update_info = agent.update_parameters(delayed_update=True)
                if update_info:
                    episode_loss.append(update_info['critic_loss'])

            # Update state and counters
            depth_image = next_depth_image
            drone_state = next_drone_state
            episode_reward += reward
            step_count += 1

            # Render if requested
            if render:
                env.visualize_3d_lidar(depth_image, frame_count=step, show=True)

            # If episode is done, break
            if done:
                break

        # Log metrics
        episode_rewards.append(episode_reward)
        step_counts.append(step_count)

        # Calculate average loss for this episode
        avg_loss = np.mean(episode_loss) if episode_loss else 0
        losses.append(avg_loss)

        # Track success/collision rates over a window
        success = 1 if is_success else 0
        collision = 1 if is_collision else 0

        success_rate_window.append(success)
        collision_rate_window.append(collision)

        if len(success_rate_window) > 100:
            success_rate_window.pop(0)
            collision_rate_window.pop(0)

        success_rate = sum(success_rate_window) / len(success_rate_window)
        collision_rate = sum(collision_rate_window) / len(collision_rate_window)

        # Calculate average steps
        recent_steps = step_counts[-min(100, len(step_counts)):]
        avg_steps = sum(recent_steps) / len(recent_steps)

        # Print progress
        print(f"Episode {episode}: Reward={episode_reward:.2f}, Steps={step_count}, " +
              f"Status={info.get('status', '')}, " +
              f"Success Rate={success_rate:.2f}, Collision Rate={collision_rate:.2f}, " +
              f"Buffer: Success={agent.memory.success_buffer_len()}, Fail={agent.memory.fail_buffer_len()}")

        # Log to file
        with open(log_file, "a") as f:
            f.write(f"{episode},{episode_reward:.2f},{step_count},{info.get('status', '')}," +
                    f"{success_rate:.4f},{collision_rate:.4f},{avg_steps:.2f}\n")

        # Save model periodically
        if episode % 100 == 0 and episode > 0:
            model_path = os.path.join(log_dir, f"sac_model_episode_{episode}.pt")
            agent.save(model_path)

            # Plot training progress
            plt.figure(figsize=(15, 10))

            plt.subplot(2, 2, 1)
            plt.plot(episode_rewards)
            plt.title('Episode Rewards')
            plt.xlabel('Episode')
            plt.ylabel('Reward')

            plt.subplot(2, 2, 2)
            success_rates = [sum(success_rate_window[:i + 1]) / (i + 1) for i in range(len(success_rate_window))]
            plt.plot(range(episode - len(success_rate_window) + 1, episode + 1), success_rates)
            plt.title('Success Rate')
            plt.xlabel('Episode')
            plt.ylabel('Rate')

            plt.subplot(2, 2, 3)
            collision_rates = [sum(collision_rate_window[:i + 1]) / (i + 1) for i in range(len(collision_rate_window))]
            plt.plot(range(episode - len(collision_rate_window) + 1, episode + 1), collision_rates)
            plt.title('Collision Rate')
            plt.xlabel('Episode')
            plt.ylabel('Rate')

            plt.subplot(2, 2, 4)
            if losses:
                plt.plot(losses)
                plt.title('Critic Loss')
                plt.xlabel('Episode')
                plt.ylabel('Loss')

            plt.tight_layout()
            plt.savefig(os.path.join(log_dir, f"training_progress_episode_{episode}.png"))
            plt.close()

    # Final save
    model_path = os.path.join(log_dir, "sac_model_final.pt")
    agent.save(model_path)

    return episode_rewards


def test_agent(env, agent, num_episodes, max_steps=SACConfig.max_episode_steps,
               render=False, log_dir="logs", noise_level=0.0):
    """
    Test the trained SAC agent in the drone environment

    Args:
        env: The drone environment
        agent: The SAC agent
        num_episodes: Number of episodes to test
        max_steps: Maximum steps per episode
        render: Whether to render the environment during testing
        log_dir: Directory to save logs
        noise_level: Amount of noise to add to actions (0.0 = deterministic)

    Returns:
        Success rate, episode rewards, and step counts
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Metrics for tracking performance
    episode_rewards = []
    success_count = 0
    collision_count = 0
    timeout_count = 0
    step_counts = []

    # Log file
    log_file = os.path.join(log_dir, f"test_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(log_file, "w") as f:
        f.write("Episode,Reward,Steps,Status,DistanceToGoal\n")

    for episode in range(num_episodes):
        episode_reward = 0
        step_count = 0

        # Reset environment
        depth_image, drone_state = env.reset()

        for step in range(max_steps):
            # Select action deterministically (with optional noise)
            action = agent.select_action(depth_image, drone_state, evaluate=True)

            # Add exploration noise if requested
            if noise_level > 0:
                noise = np.random.normal(0, noise_level, size=action.shape)
                action += noise
                # Clip to valid range
                action = np.clip(action, -agent.max_action, agent.max_action)

            # Take action in environment
            next_depth_image, next_drone_state, reward, done, info = env.step(action)

            # Update state and counters
            depth_image = next_depth_image
            drone_state = next_drone_state
            episode_reward += reward
            step_count += 1

            # Render if requested
            if render:
                env.visualize_3d_lidar(depth_image, frame_count=step, show=True)

            # If episode is done, break
            if done:
                break

        # Log metrics
        episode_rewards.append(episode_reward)
        step_counts.append(step_count)

        # Check outcome
        status = info.get('status', 'unknown')
        if status == 'goal_reached':
            success_count += 1
        elif status == 'collision':
            collision_count += 1
        elif info.get('timeout', False):
            timeout_count += 1

        # Log to file
        with open(log_file, "a") as f:
            f.write(f"{episode},{episode_reward:.2f},{step_count},{status}," +
                    f"{info.get('distance_to_goal', -1):.4f}\n")

        # Print progress
        print(f"Test Episode {episode}: Reward={episode_reward:.2f}, Steps={step_count}, " +
              f"Status={status}, Distance to Goal={info.get('distance_to_goal', -1):.2f}")

    # Calculate overall success rate
    success_rate = success_count / num_episodes
    collision_rate = collision_count / num_episodes
    timeout_rate = timeout_count / num_episodes
    avg_steps = sum(step_counts) / len(step_counts)
    avg_reward = sum(episode_rewards) / len(episode_rewards)

    print(f"\nTest Results:")
    print(f"Success Rate: {success_rate:.4f} ({success_count}/{num_episodes})")
    print(f"Collision Rate: {collision_rate:.4f} ({collision_count}/{num_episodes})")
    print(f"Timeout Rate: {timeout_rate:.4f} ({timeout_count}/{num_episodes})")
    print(f"Average Steps: {avg_steps:.2f}")
    print(f"Average Reward: {avg_reward:.2f}")

    # Save results to summary file
    summary_file = os.path.join(log_dir, f"test_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(summary_file, "w") as f:
        f.write(f"Test Episodes: {num_episodes}\n")
        f.write(f"Success Rate: {success_rate:.4f} ({success_count}/{num_episodes})\n")
        f.write(f"Collision Rate: {collision_rate:.4f} ({collision_count}/{num_episodes})\n")
        f.write(f"Timeout Rate: {timeout_rate:.4f} ({timeout_count}/{num_episodes})\n")
        f.write(f"Average Steps: {avg_steps:.2f}\n")
        f.write(f"Average Reward: {avg_reward:.2f}\n")

    return success_rate, episode_rewards, step_counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or test SAC agent for drone obstacle avoidance")
    parser.add_argument("--mode", choices=["train", "test"], default="train", help="Train or test mode")
    parser.add_argument("--episodes", type=int, default=5000, help="Number of episodes")
    parser.add_argument("--render", action="store_true", help="Render the environment")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory to save logs")
    parser.add_argument("--model_path", type=str, help="Path to model file (for testing)")
    parser.add_argument("--max_steps", type=int, default=SACConfig.max_episode_steps, help="Maximum steps per episode")
    parser.add_argument("--noise", type=float, default=0.0, help="Noise level for testing (0.0 = deterministic)")
    parser.add_argument("--continue_train", action="store_true", help="Continue training from a saved model")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")
    args = parser.parse_args()

    # Set device
    if args.gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Create environment
    env = DroneEnv()

    # Get state and action dimensions
    depth_image, drone_state = env.reset()
    state_dim = len(drone_state)
    action_dim = 3  # vx, vy, vz

    # Create agent
    agent = SACAgent(state_dim, action_dim, max_action=Config.max_drone_speed, device=device)

    # Create log directory
    os.makedirs(args.log_dir, exist_ok=True)

    if args.mode == "train":
        # Load model if continuing training
        if args.continue_train and args.model_path:
            print(f"Loading model from {args.model_path} and continuing training")
            agent.load(args.model_path)

        # Train agent
        try:
            print(f"Starting training for {args.episodes} episodes")
            start_time = time.time()

            rewards = train_agent(env, agent, num_episodes=args.episodes,
                                  max_steps=args.max_steps, render=args.render,
                                  log_dir=args.log_dir)

            training_time = time.time() - start_time
            print(f"Training completed in {training_time:.2f} seconds")

            # Plot final training rewards
            plt.figure(figsize=(10, 5))
            plt.plot(rewards)
            plt.title('Episode Rewards')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.savefig(os.path.join(args.log_dir, 'final_training_rewards.png'))
            plt.close()

        except KeyboardInterrupt:
            print("Training interrupted.")
            # Save the model on interrupt
            interrupted_path = os.path.join(args.log_dir, "sac_model_interrupted.pt")
            agent.save(interrupted_path)
            print(f"Model saved to {interrupted_path}")

    elif args.mode == "test":
        if not args.model_path:
            print("Error: Model path is required for testing")
            exit(1)

        # Load model
        print(f"Loading model from {args.model_path}")
        agent.load(args.model_path)

        # Test the trained agent
        print(f"Starting testing for {args.episodes} episodes")
        start_time = time.time()

        success_rate, test_rewards, step_counts = test_agent(
            env, agent, num_episodes=args.episodes, max_steps=args.max_steps,
            render=args.render, log_dir=args.log_dir, noise_level=args.noise
        )

        testing_time = time.time() - start_time
        print(f"Testing completed in {testing_time:.2f} seconds")