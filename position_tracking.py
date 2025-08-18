import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import time
from datetime import datetime

# Import environment
from Environment.Env import DroneEnv, Config

# Import the GRU-SAC agent
from Networks.sac import GRUSACAgent, SACConfig


def train_agent(env, agent, num_episodes, max_steps=SACConfig.max_episode_steps,
                render=False, start_steps=SACConfig.start_steps,
                updates_per_episode=10, log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(log_file, "w") as f:
        f.write("Episode,Reward,Steps,Status,SuccessRate,CollisionRate,AvgSteps,Losses\n")

    episode_rewards = []
    success_window = []
    collision_window = []
    step_counts = []

    # 학습 진행 추적
    total_steps = 0
    best_success_rate = 0.0

    for episode in range(num_episodes):
        ep_reward = 0.0
        steps = 0
        depth, state = env.reset()
        agent.reset_hidden()

        done = False
        episode_losses = []

        while not done and steps < max_steps:
            if total_steps < start_steps:
                # 초기 exploration을 위한 랜덤 액션
                action = np.random.uniform(-agent.max_action, agent.max_action, size=agent.action_dim)
            else:
                # 정책에서 액션 선택
                action = agent.select_action(depth, state)

            next_depth, next_state, reward, done, info = env.step(action)

            # Reward scaling 적용
            scaled_reward = reward * SACConfig.reward_scale

            agent.add_experience(depth, state, action, scaled_reward,
                                 next_depth, next_state, float(done), False)

            depth, state = next_depth, next_state
            ep_reward += reward  # 원본 보상으로 기록
            steps += 1
            total_steps += 1

            # 온라인 학습 수행
            if total_steps > start_steps and total_steps % SACConfig.online_update_freq == 0:
                for _ in range(SACConfig.batch_updates_per_step):
                    update_info = agent.update_parameters(delayed_update=False)
                    if update_info is not None:
                        episode_losses.append({
                            'critic': update_info['critic_loss'],
                            'actor': update_info['actor_loss'],
                            'alpha': update_info['alpha']
                        })

            if render:
                env.visualize_3d_lidar(depth, frame_count=total_steps, show=True)

        # 에피소드 종료 후 추가 배치 업데이트
        last_info = info.get('status', '')
        if total_steps > start_steps:
            for _ in range(updates_per_episode):
                update_info = agent.update_parameters(delayed_update=False)
                if update_info is not None:
                    episode_losses.append({
                        'critic': update_info['critic_loss'],
                        'actor': update_info['actor_loss'],
                        'alpha': update_info['alpha']
                    })

        # 통계 업데이트
        episode_rewards.append(ep_reward)
        step_counts.append(steps)
        succ = 1 if last_info == 'goal_reached' else 0
        coll = 1 if last_info == 'collision' else 0
        success_window.append(succ)
        collision_window.append(coll)

        # 최근 100 에피소드 통계
        if len(success_window) > 100:
            success_window.pop(0)
            collision_window.pop(0)

        success_rate = sum(success_window) / len(success_window)
        collision_rate = sum(collision_window) / len(collision_window)
        recent_steps = step_counts[-min(100, len(step_counts)):]
        avg_steps = sum(recent_steps) / len(recent_steps)

        # Loss 평균 계산
        if episode_losses:
            avg_critic_loss = sum(l['critic'] for l in episode_losses) / len(episode_losses)
            avg_actor_loss = sum(l['actor'] for l in episode_losses) / len(episode_losses)
            avg_alpha = sum(l['alpha'] for l in episode_losses) / len(episode_losses)
        else:
            avg_critic_loss = avg_actor_loss = avg_alpha = 0.0

        # 진행 상황 출력
        print(f"Ep {episode:4d}: R={ep_reward:8.2f}, Steps={steps:4d}, "
              f"Status={last_info:12s}, SuccRate={success_rate:.2f}, "
              f"CollRate={collision_rate:.2f}, Buffer={len(agent.memory):6d}")

        if total_steps > start_steps:
            print(f"         Losses - Critic: {avg_critic_loss:.4f}, "
                  f"Actor: {avg_actor_loss:.4f}, Alpha: {avg_alpha:.4f}")

        # 로그 파일에 기록
        with open(log_file, "a") as f:
            f.write(f"{episode},{ep_reward:.2f},{steps},{last_info},"
                    f"{success_rate:.4f},{collision_rate:.4f},{avg_steps:.2f},"
                    f"{avg_critic_loss:.4f},{avg_actor_loss:.4f},{avg_alpha:.4f}\n")

        # 모델 저장 (개선)
        if success_rate > best_success_rate and total_steps > start_steps:
            best_success_rate = success_rate
            best_path = os.path.join(log_dir, f"best_model_sr{success_rate:.3f}.pt")
            agent.save(best_path)
            print(f"Saved best model with success rate: {success_rate:.3f}")

        # 주기적 저장
        if episode and episode % 500 == 0:
            path = os.path.join(log_dir, f"gru_sac_ep{episode}.pt")
            agent.save(path)

    # 최종 저장
    final_path = os.path.join(log_dir, "gru_sac_final.pt")
    agent.save(final_path)

    # 학습 그래프 생성
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')

    plt.subplot(2, 2, 2)
    window_size = 100
    if len(episode_rewards) > window_size:
        smoothed_rewards = np.convolve(episode_rewards,
                                       np.ones(window_size) / window_size,
                                       mode='valid')
        plt.plot(smoothed_rewards)
        plt.title(f'Smoothed Rewards (window={window_size})')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')

    plt.subplot(2, 2, 3)
    success_rates = []
    collision_rates = []
    for i in range(len(success_window)):
        if i >= 99:
            success_rates.append(sum(success_window[i - 99:i + 1]) / 100)
            collision_rates.append(sum(collision_window[i - 99:i + 1]) / 100)

    plt.plot(success_rates, label='Success Rate')
    plt.plot(collision_rates, label='Collision Rate')
    plt.title('Success/Collision Rates (100-ep window)')
    plt.xlabel('Episode')
    plt.ylabel('Rate')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.hist(episode_rewards, bins=50)
    plt.title('Reward Distribution')
    plt.xlabel('Reward')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'training_summary.png'), dpi=300)
    plt.close()

    return episode_rewards


def test_agent(env, agent, num_episodes, max_steps=SACConfig.max_episode_steps,
               render=False, log_dir="logs", noise_level=0.0):
    os.makedirs(log_dir, exist_ok=True)
    episode_rewards, step_counts = [], []
    success_count = collision_count = timeout_count = 0

    # 상세 통계
    detailed_stats = []

    for episode in range(num_episodes):
        ep_reward, steps = 0.0, 0
        depth, state = env.reset()
        agent.reset_hidden()

        # 에피소드별 상세 추적
        episode_info = {
            'actions': [],
            'states': [],
            'rewards': []
        }

        for step_count in range(max_steps):
            action = agent.select_action(depth, state, evaluate=True)

            # 추가 노이즈 (테스트 목적)
            if noise_level > 0:
                action += np.random.normal(0, noise_level, size=action.shape)
                action = np.clip(action, -agent.max_action, agent.max_action)

            depth, state, reward, done, info = env.step(action)

            # 상세 정보 저장
            episode_info['actions'].append(action.copy())
            episode_info['states'].append(state.copy())
            episode_info['rewards'].append(reward)

            ep_reward += reward
            steps += 1

            if render:
                env.visualize_3d_lidar(depth, frame_count=step_count, show=True)

            if done:
                break

        status = info.get('status', 'unknown')
        if status == 'goal_reached':
            success_count += 1
        elif status == 'collision':
            collision_count += 1
        elif info.get('timeout', False):
            timeout_count += 1

        episode_rewards.append(ep_reward)
        step_counts.append(steps)
        detailed_stats.append(episode_info)

        print(f"Test {episode}: Reward={ep_reward:.2f}, Steps={steps}, Status={status}")

    # 결과 통계
    success_rate = success_count / num_episodes
    collision_rate = collision_count / num_episodes
    timeout_rate = timeout_count / num_episodes

    print("\n=== Test Results ===")
    print(f"Success Rate: {success_rate:.3f}")
    print(f"Collision Rate: {collision_rate:.3f}")
    print(f"Timeout Rate: {timeout_rate:.3f}")
    print(f"Average Reward: {np.mean(episode_rewards):.2f}")
    print(f"Average Steps: {np.mean(step_counts):.2f}")

    # 결과 저장
    results_path = os.path.join(log_dir, "test_results.npz")
    np.savez(results_path,
             rewards=episode_rewards,
             steps=step_counts,
             success_rate=success_rate,
             collision_rate=collision_rate,
             timeout_rate=timeout_rate)

    return success_rate, episode_rewards, step_counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--max_steps", type=int, default=SACConfig.max_episode_steps)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--continue", dest="continue_train", action="store_true")
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    env = DroneEnv()
    depth, state = env.reset()
    state_dim = len(state)
    action_dim = 3
    agent = GRUSACAgent(state_dim, action_dim,
                        o_dim=(Config.depth_image_height, Config.depth_image_width),
                        max_action=Config.max_drone_speed, device=device)

    if args.mode == "train":
        if args.continue_train and args.model_path:
            agent.load(args.model_path)
            print(f"Continuing training from {args.model_path}")

        rewards = train_agent(env, agent, args.episodes,
                              max_steps=args.max_steps, render=args.render,
                              log_dir=args.log_dir)
    else:
        if not args.model_path:
            raise ValueError("Model path required for test mode")
        agent.load(args.model_path)
        test_agent(env, agent, args.episodes,
                   max_steps=args.max_steps, render=args.render,
                   log_dir=args.log_dir, noise_level=args.noise)