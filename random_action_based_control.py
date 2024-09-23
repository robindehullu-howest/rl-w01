import gymnasium
import seaborn as sns
import matplotlib.pyplot as plt

env = gymnasium.make('CartPole-v1', render_mode=None)

successful_timesteps = []

for episode in range(1000):
    observation, info = env.reset()
    episode_timestamp = 0

    while True:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        episode_timestamp += 1
        if terminated or truncated:
            print(f"Episode {episode} finished after {episode_timestamp} timesteps")
            successful_timesteps.append(episode_timestamp)
            break

sns.histplot(successful_timesteps, kde=False)
plt.xlabel('Number of Successful Timesteps')
plt.ylabel('Frequency')
plt.title('Histogram of Successful Timesteps per Episode')
plt.show()

average_timesteps = sum(successful_timesteps) / len(successful_timesteps)
print(f"Average number of successful timesteps: {average_timesteps}")