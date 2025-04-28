import torch

from Attn_Cnn_SAC.Environment.Env import DroneEnv
from Attn_Cnn_SAC.Networks.Sep_CNN_SAC import DACSACAgent


def train(env, agent, num_episodes=3500, max_steps=1000, evaluate_interval=100):
    """
    에이전트 훈련 함수
    """
    episode_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        state = state.to(agent.device).unsqueeze(0)  # 배치 차원 추가
        episode_reward = 0

        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done = env.step(action.cpu().numpy())
            next_state = torch.tensor(next_state, dtype=torch.float32).to(agent.device).unsqueeze(0)

            agent.store_transition(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

            if done:
                break

        # 에피소드 종료 후 지연 학습 수행
        agent.end_episode()

        episode_rewards.append(episode_reward)

        # 평가 및 로깅
        if (episode + 1) % evaluate_interval == 0:
            avg_reward = sum(episode_rewards[-evaluate_interval:]) / evaluate_interval
            print(f"Episode {episode + 1}/{num_episodes}, Avg Reward: {avg_reward:.2f}")

    return episode_rewards

# Hyper Parameter
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = (84, 84)
ACTION_DIM = 3
STATE_CHANNELS = 1
NUM_EPISODES = 3500


if __name__ == "__main__":
    # 환경 및 에이전트 생성
    env = DroneEnv(image_size=IMAGE_SIZE)
    agent = DACSACAgent(
        state_ch=STATE_CHANNELS,
        action_dim=ACTION_DIM,
        device=DEVICE,
        auto_alpha=True,
        buffer_pos_size=100000,
        buffer_neg_size=100000,
        lr=1e-4,
        gamma=0.99,
        tau=0.005
    )

    # 훈련 시작
    rewards = train(env, agent, num_episodes=NUM_EPISODES)

    # 모델 저장
    torch.save(agent.actor.state_dict(), "dac_sac_actor.pth")
    torch.save(agent.critic.state_dict(), "dac_sac_critic.pth")

    print("Training completed!")