import gymnasium
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def run_episode(env, weights):
    observation, info = env.reset()
    total_reward = 0
    while True:
        evaluation = np.dot(weights, observation)
        action = 0 if evaluation < 0 else 1
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    return total_reward

def random_search(env, num_iterations=1000, episodes_per_iteration=20):
    history = []
    print("\nIterations:")
    for iteration in range(num_iterations):
        weights = np.random.uniform(-1, 1, size=env.observation_space.shape[0])
        total_reward = sum(run_episode(env, weights) for _ in range(episodes_per_iteration))
        average_reward = total_reward / episodes_per_iteration
        history.append((average_reward, weights))
        if average_reward > 100:
            print(f"\tIteration {iteration}: Average Reward = {average_reward}")
    return history

def plot_histogram(average_rewards):
    sns.histplot(average_rewards, kde=False)
    plt.xlabel('Average Reward')
    plt.ylabel('Frequency')
    plt.title('Histogram of Average Rewards per Iteration')
    plt.show()

def plot_3d_scatter(history, important_indices):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = [entry[1][important_indices[0]] for entry in history]
    y = [entry[1][important_indices[1]] for entry in history]
    z = [entry[1][important_indices[2]] for entry in history]
    colors = ['red' if entry[0] >= 100 else 'black' for entry in history]

    ax.scatter(x, y, z, c=colors)
    ax.set_xlabel(f'Weight {important_indices[0] + 1}')
    ax.set_ylabel(f'Weight {important_indices[1] + 1}')
    ax.set_zlabel(f'Weight {important_indices[2] + 1}')
    plt.title('3D Scatter Plot of Important Weights')
    plt.show()

def main():
    env = gymnasium.make('CartPole-v1', render_mode=None)

    history = random_search(env)

    average_rewards = [entry[0] for entry in history]

    plot_histogram(average_rewards)

    weights_matrix = np.array([entry[1] for entry in history])
    contributions = weights_matrix * np.array([entry[0] for entry in history])[:, None]
    variances = np.var(contributions, axis=0)

    important_indices = np.argsort(variances)[::-1][:3]

    plot_3d_scatter(history, important_indices)

    print("\nBest weights (Max average reward):")
    for average_reward, weights in history:
        if average_reward > 499.9:
            print(f"\tWeights = {weights}")

if __name__ == "__main__":
    main()