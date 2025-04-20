import math
import numpy as np
import gym
from collections import deque
import csv
import os
import json

class QLearner:
    def __init__(self, params=None):
        self.environment = gym.make('CartPole-v1', render_mode=None)

        # Default parameters
        default_params = {
            'learning_rate': 0.5,
            'learning_rate_decay': 0.9995,
            'learning_rate_min': 0.05,
            'epsilon': 1.0,
            'epsilon_decay': 0.999,
            'epsilon_min': 0.01,
            'discount_factor': 0.99,
            'buckets': (1, 1, 6, 12),
            'stability_threshold': 50,
            'stability_delta': 0.1
        }

        # Update parameters if provided
        self.params = default_params.copy()
        if params:
            self.params.update(params)

        self.upper_bounds = [
            self.environment.observation_space.high[0],
            3.0,
            self.environment.observation_space.high[2],
            math.radians(50),
        ]
        self.lower_bounds = [
            self.environment.observation_space.low[0],
            -3.0,
            self.environment.observation_space.low[2],
            -math.radians(50),
        ]

        self.q_table = np.ones(self.params['buckets'] + (self.environment.action_space.n,))  # Optimistic init
        self.attempt_no = 1

        # Convergence detection
        self.reward_window = deque(maxlen=100)
        self.best_avg_reward = -np.inf
        self.stable_episodes = 0
        self.converged = False

    def discretise(self, observation):
        ratios = [
            (observation[i] - self.lower_bounds[i]) / (self.upper_bounds[i] - self.lower_bounds[i])
            for i in range(len(observation))
        ]
        ratios = [min(max(r, 0), 1) for r in ratios]
        return tuple(
            int(round((self.params['buckets'][i] - 1) * ratios[i])) for i in range(len(self.params['buckets']))
        )

    def pick_action(self, state):
        if np.random.random() < self.params['epsilon']:
            return self.environment.action_space.sample()
        else:
            return int(np.argmax(self.q_table[state]))

    def update_knowledge(self, state, action, reward, next_state):
        q_current = self.q_table[state + (action,)]
        q_future = np.max(self.q_table[next_state])
        td_target = reward + self.params['discount_factor'] * q_future
        self.q_table[state + (action,)] += self.params['learning_rate'] * (td_target - q_current)

    def attempt(self):
        obs, _ = self.environment.reset()
        state = self.discretise(obs)
        terminated, truncated = False, False
        reward_sum = 0

        while not terminated and not truncated:
            action = self.pick_action(state)
            new_obs, reward, terminated, truncated, _ = self.environment.step(action)
            next_state = self.discretise(new_obs)
            self.update_knowledge(state, action, reward, next_state)
            state = next_state
            reward_sum += reward

        return reward_sum

    def check_convergence(self, current_avg):
        if abs(current_avg - self.best_avg_reward) < self.params['stability_delta']:
            self.stable_episodes += 1
        else:
            self.stable_episodes = 0
            self.best_avg_reward = current_avg

        if self.stable_episodes >= self.params['stability_threshold']:
            print(f"✅ Q-Learning Converged: Avg(100): {current_avg:.2f}")
            self.converged = True
            self.params['epsilon'] = 0.0

    def learn(self, max_attempts, param_set_name="default"):
        results_dir = os.path.join(os.path.dirname(__file__), 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        csv_filename = os.path.join(results_dir, f'results_{param_set_name}.csv')
        
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Attempt', 'Reward', 'Average_Reward', 'Epsilon'])
            
            for _ in range(max_attempts):
                reward_sum = self.attempt()
                self.reward_window.append(reward_sum)
                avg_reward = np.mean(self.reward_window)
                
                writer.writerow([self.attempt_no, reward_sum, avg_reward, self.params['epsilon']])
                
                print(f"(Q-Learn) Ep: {self.attempt_no} | Reward: {reward_sum} | Avg: {avg_reward:.2f} | ε: {self.params['epsilon']:.4f}")
                
                if len(self.reward_window) == self.reward_window.maxlen and not self.converged:
                    self.check_convergence(avg_reward)

                if not self.converged:
                    self.params['epsilon'] = max(self.params['epsilon_min'], self.params['epsilon'] * self.params['epsilon_decay'])
                    self.params['learning_rate'] = max(self.params['learning_rate_min'], self.params['learning_rate'] * self.params['learning_rate_decay'])
                
                self.attempt_no += 1
        
        self.environment.close()
        return self.converged

def main():
    # Load parameter sets from JSON file
    json_path = os.path.join(os.path.dirname(__file__), 'parameter_sets.json')
    with open(json_path, 'r') as f:
        parameter_sets = json.load(f)
    
    # Convert bucket lists to tuples for each parameter set
    for param_set in parameter_sets.values():
        param_set['buckets'] = tuple(param_set['buckets'])

    # Test each parameter set
    for param_name, params in parameter_sets.items():
        print(f"\nTesting parameter set: {param_name}")
        learner = QLearner(params)
        converged = learner.learn(10000, param_name)
        if not converged:
            print(f"❌ Algorithm did not converge within the maximum number of attempts for {param_name}")

if __name__ == '__main__':
    main()
