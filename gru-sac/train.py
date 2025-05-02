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

import threading
import queue
import time


# AsyncGRUSACAgent 클래스 수정
class AsyncGRUSACAgent(GRUSACAgent):
    def __init__(self, state_dim, action_dim, max_action=1.0, device="cpu"):
        super().__init__(state_dim, action_dim, max_action, device)

        # 비동기 학습을 위한 추가 속성
        self.experience_queue = queue.Queue(maxsize=10000)
        self.stop_training = threading.Event()
        self.training_thread = None
        self.training_active = False

        # 학습 설정
        self.updates_per_iteration = 10
        self.min_experiences_for_update = SACConfig.batch_size * 2

    # 기존 add_experience 메서드 수정 불필요 (계속 is_success 사용)

    def _training_loop(self):
        print("학습 스레드 시작됨")

        last_update_time = time.time()
        update_interval = 0.1

        while not self.stop_training.is_set():
            current_time = time.time()

            # 큐에서 대기 중인 모든 경험 처리
            experiences_count = 0
            while not self.experience_queue.empty():
                experience = self.experience_queue.get()
                # SequenceExperienceBuffer는 is_success 인자를 사용하므로 그대로 유지
                self.memory.add(*experience)
                experiences_count += 1

            # 일정 시간이 지나고 충분한 데이터가 있으면 업데이트 수행
            if (current_time - last_update_time >= update_interval and
                    len(self.memory) >= self.min_experiences_for_update):

                # 여러 번 업데이트 수행
                update_info = None
                for _ in range(self.updates_per_iteration):
                    update_result = self.update_parameters(delayed_update=False)
                    if update_result is not None:
                        update_info = update_result

                # 업데이트 정보 로깅
                if update_info:
                    critic_loss = update_info['critic_loss']
                    actor_loss = update_info['actor_loss']
                    print(f"Updates: critic_loss={critic_loss:.4f}, actor_loss={actor_loss:.4f}, " +
                          f"buffer_size={len(self.memory)}, " +
                          f"Success={self.memory.success_buffer_len()}, Fail={self.memory.fail_buffer_len()}")

                last_update_time = current_time

            # CPU 과부하 방지를 위한 짧은 대기
            time.sleep(0.001)

        print("학습 스레드 종료됨")

def train_agent_async(env, agent, num_episodes, max_steps=SACConfig.max_episode_steps,
                      render=False, start_steps=SACConfig.start_steps, log_dir="logs"):
    """
    비동기 학습을 사용한 에이전트 학습

    Args:
        env: 드론 환경
        agent: 비동기 학습 가능한 SAC 에이전트
        num_episodes: 학습할 에피소드 수
        max_steps: 에피소드당 최대 스텝 수
        render: 환경 시각화 여부
        start_steps: 초기 랜덤 액션 수행 스텝 수
        log_dir: 로그 저장 디렉토리

    Returns:
        에피소드 보상 목록
    """
    # 로그 디렉토리 생성
    os.makedirs(log_dir, exist_ok=True)

    # 학습 진행 추적을 위한 메트릭
    episode_rewards = []
    success_rate_window = []
    collision_rate_window = []
    step_counts = []

    # 로그 파일
    log_file = os.path.join(log_dir, f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(log_file, "w") as f:
        f.write("Episode,Reward,Steps,Status,SuccessRate,CollisionRate,AvgSteps\n")

    # 비동기 학습 시작
    agent.start_training()

    try:
        for episode in range(num_episodes):
            episode_reward = 0
            step_count = 0

            # 환경 및 에이전트 히든 스테이트 초기화
            depth_sequence, drone_state = env.reset()
            agent.reset_hidden()

            for step in range(max_steps):
                # 초기 스텝에서는 랜덤 액션 (탐색용)
                if len(agent.memory) < start_steps:
                    action = np.random.uniform(-agent.max_action, agent.max_action, size=agent.action_dim)
                else:
                    action = agent.select_action(depth_sequence[-1], drone_state)

                # 환경에서 액션 수행
                next_depth_sequence, next_drone_state, reward, done, info = env.step(action)

                # 성공/충돌 여부 확인
                is_success = info.get('status', '') == 'goal_reached'
                is_collision = info.get('status', '') == 'collision'

                # 경험 큐에 추가 (비동기 학습을 위해)
                agent.add_experience(
                    depth_sequence, drone_state, action, reward,
                    next_depth_sequence, next_drone_state, float(done), is_success
                )

                # 상태 및 카운터 업데이트
                depth_sequence = next_depth_sequence
                drone_state = next_drone_state
                episode_reward += reward
                step_count += 1

                # 환경 렌더링 (요청 시)
                if render:
                    env.visualize_3d_lidar(depth_sequence[-1], frame_count=step, show=True)

                # 에피소드 종료 시 루프 탈출
                if done:
                    break

            # 메트릭 로깅
            episode_rewards.append(episode_reward)
            step_counts.append(step_count)

            # 성공/충돌률 추적
            success = 1 if is_success else 0
            collision = 1 if is_collision else 0

            success_rate_window.append(success)
            collision_rate_window.append(collision)

            if len(success_rate_window) > 100:
                success_rate_window.pop(0)
                collision_rate_window.pop(0)

            success_rate = sum(success_rate_window) / len(success_rate_window)
            collision_rate = sum(collision_rate_window) / len(collision_rate_window)

            # 평균 스텝 계산
            recent_steps = step_counts[-min(100, len(step_counts)):]
            avg_steps = sum(recent_steps) / len(recent_steps)

            # 진행 상황 출력
            print(f"Episode {episode}: Reward={episode_reward:.2f}, Steps={step_count}, " +
                  f"Status={info.get('status', '')}, " +
                  f"Success Rate={success_rate:.2f}, Collision Rate={collision_rate:.2f}, " +
                  f"Buffer: Success={agent.memory.success_buffer_len()}, Fail={agent.memory.fail_buffer_len()}")

            # 로그 파일에 기록
            with open(log_file, "a") as f:
                f.write(f"{episode},{episode_reward:.2f},{step_count},{info.get('status', '')}," +
                        f"{success_rate:.4f},{collision_rate:.4f},{avg_steps:.2f}\n")

            # 주기적으로 모델 저장
            if episode % 100 == 0 and episode > 0:
                model_path = os.path.join(log_dir, f"gru_sac_model_episode_{episode}.pt")
                agent.save(model_path)

                # 학습 진행 상황 그래프 생성
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
                collision_rates = [sum(collision_rate_window[:i + 1]) / (i + 1) for i in
                                   range(len(collision_rate_window))]
                plt.plot(range(episode - len(collision_rate_window) + 1, episode + 1), collision_rates)
                plt.title('Collision Rate')
                plt.xlabel('Episode')
                plt.ylabel('Rate')

                plt.tight_layout()
                plt.savefig(os.path.join(log_dir, f"training_progress_episode_{episode}.png"))
                plt.close()

    except KeyboardInterrupt:
        print("학습 중단됨.")
    finally:
        # 비동기 학습 중지
        agent.stop_training_thread()

        # 최종 모델 저장
        model_path = os.path.join(log_dir, "gru_sac_model_final.pt")
        agent.save(model_path)

    return episode_rewards


def test_agent(env, agent, num_episodes, max_steps=SACConfig.max_episode_steps,
               render=False, log_dir="logs", noise_level=0.0):
    """
    Test the trained GRU-SAC agent in the drone environment

    Args:
        env: The drone environment
        agent: The GRU-SAC agent
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

        # Reset environment and agent's hidden states
        depth_sequence, drone_state = env.reset()
        agent.reset_hidden()

        for step in range(max_steps):
            # Select action deterministically (with optional noise)
            action = agent.select_action(depth_sequence[-1], drone_state, evaluate=True)

            # Add exploration noise if requested
            if noise_level > 0:
                noise = np.random.normal(0, noise_level, size=action.shape)
                action += noise
                # Clip to valid range
                action = np.clip(action, -agent.max_action, agent.max_action)

            # Take action in environment
            next_depth_sequence, next_drone_state, reward, done, info = env.step(action)

            # Update state and counters
            depth_sequence = next_depth_sequence
            drone_state = next_drone_state
            episode_reward += reward
            step_count += 1

            # Render if requested
            if render:
                env.visualize_3d_lidar(depth_sequence[-1], frame_count=step, show=True)

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
    parser = argparse.ArgumentParser(description="Train or test GRU-SAC agent for drone obstacle avoidance")
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

        # 환경 생성
    env = DroneEnv()

    # 상태 및 행동 차원 가져오기
    depth_sequence, drone_state = env.reset()
    state_dim = len(drone_state)
    action_dim = 3  # vx, vy, vz

    # 비동기 학습이 가능한 에이전트 생성
    agent = AsyncGRUSACAgent(state_dim, action_dim, max_action=Config.max_drone_speed, device=device)

    if args.mode == "train":
        # 모델 로드 (이어서 학습하는 경우)
        if args.continue_train and args.model_path:
            print(f"모델 로드: {args.model_path}")
            agent.load(args.model_path)

        # 비동기 학습 시작
        print(f"에피소드 {args.episodes}개 학습 시작")
        start_time = time.time()

        rewards = train_agent_async(env, agent, num_episodes=args.episodes,
                                    max_steps=args.max_steps, render=args.render,
                                    log_dir=args.log_dir)

        training_time = time.time() - start_time
        print(f"학습 완료: {training_time:.2f}초 소요")

        # 결과 그래프 생성
        plt.figure(figsize=(10, 5))
        plt.plot(rewards)
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.savefig(os.path.join(args.log_dir, 'final_training_rewards.png'))
        plt.close()

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