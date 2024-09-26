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

def annealing(env, num_iterations=1000, episodes_per_iteration=20, sigma=0.1, starting_temperature=10000, cooling_rate=0.999):
    history = []
    best_reward = 0
    best_weights = np.random.uniform(-1, 1, size=env.observation_space.shape[0])
    temperature = starting_temperature

    print("\nIterations:")

    for iteration in range(num_iterations):
        noise = np.random.normal(0, sigma, size=best_weights.shape)
        current_weights = best_weights + noise
        
        total_reward = sum(run_episode(env, current_weights) for _ in range(episodes_per_iteration))
        current_reward = total_reward / episodes_per_iteration
        
        history.append((current_reward, current_weights))
        print(f"\tIteration {iteration}: Average Reward = {current_reward}\t{current_weights}\tTemperature = {temperature:.2f}")
        
        if current_reward > best_reward:
            best_weights = current_weights
            best_reward = current_reward
        else:
            probability = np.exp((current_reward - best_reward) / temperature)
            if np.random.rand() < probability:
                best_weights = current_weights
                best_reward = current_reward
        
        temperature *= cooling_rate
    
    return history

def plot_histogram(data, title, xlabel, ylabel):
    sns.histplot(data, kde=False)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

def main():
    env = gymnasium.make('CartPole-v1', render_mode=None)
    env.metadata["render_fps"] = 0

    history = annealing(env, starting_temperature=10000, cooling_rate=0.992)

    max_average_reward, best_weights = max(history, key=lambda x: x[0])

    print(f"\nMax average reward: {max_average_reward}")
    print(f"Best weights: {best_weights}")

    long_run = [run_episode(env, best_weights) for _ in range(1000)]
    long_run_average_reward = np.mean(long_run)
    print(f"\nLong run average reward: {long_run_average_reward}")

    plot_histogram(long_run, 'Histogram of Rewards over 1000 Episodes', 'Reward', 'Frequency')

if __name__ == "__main__":
    main()