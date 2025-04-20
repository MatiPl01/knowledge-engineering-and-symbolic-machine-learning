import math
import numpy as np
import gym
import os
import csv
import json
from collections import deque

class SARSALearner:
    def __init__(self, params=None):
        self.env = gym.make('CartPole-v1', render_mode=None)
        
        # Default parameters
        default_params = {
            'learning_rate': 0.5,
            'learning_rate_decay': 0.9995,
            'learning_rate_min': 0.05,
            'epsilon': 1.0,
            'epsilon_decay': 0.999,
            'epsilon_min': 0.01,
            'discount_factor': 0.99,
            'buckets': (1, 1, 6, 12)
        }

        # Update parameters if provided
        self.params = default_params.copy()
        if params:
            self.params.update(params)

        self.learning_rate = self.params['learning_rate']
        self.epsilon = self.params['epsilon']
        self.q_table = np.ones(self.params['buckets'] + (self.env.action_space.n,))  # Optimistic init

        self.upper_bounds = [
            self.env.observation_space.high[0],
            3.0,
            self.env.observation_space.high[2],
            math.radians(50),
        ]
        self.lower_bounds = [
            self.env.observation_space.low[0],
            -3.0,
            self.env.observation_space.low[2],
            -math.radians(50),
        ]

        self.attempt = 1
        self.reward_window = deque(maxlen=100)
        self.best_avg = -np.inf
        self.stable = 0
        self.stability_thresh = 50
        self.delta = 0.1
        self.converged = False

    def discretise(self, obs):
        ratios = [
            (obs[i] - self.lower_bounds[i]) / (self.upper_bounds[i] - self.lower_bounds[i])
            for i in range(len(obs))
        ]
        ratios = [min(max(r, 0), 1) for r in ratios]
        return tuple(
            int(round((self.params['buckets'][i] - 1) * ratios[i])) for i in range(len(self.params['buckets']))
        )

    def pick_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        return int(np.argmax(self.q_table[state]))

    def update_knowledge(self, s, a, r, s2, a2):
        q_sa = self.q_table[s + (a,)]
        q_s2a2 = self.q_table[s2 + (a2,)]
        td_target = r + self.params['discount_factor'] * q_s2a2
        self.q_table[s + (a,)] += self.learning_rate * (td_target - q_sa)

    def run_attempt(self):
        obs, _ = self.env.reset()
        s = self.discretise(obs)
        a = self.pick_action(s)
        done = False
        total_reward = 0

        while not done:
            obs2, reward, terminated, truncated, _ = self.env.step(a)
            s2 = self.discretise(obs2)
            a2 = self.pick_action(s2)

            self.update_knowledge(s, a, reward, s2, a2)

            s = s2
            a = a2
            done = terminated or truncated
            total_reward += reward

        return total_reward

    def check_convergence(self, avg):
        if abs(avg - self.best_avg) < self.delta:
            self.stable += 1
        else:
            self.stable = 0
            self.best_avg = avg

        if self.stable >= self.stability_thresh:
            print(f"✅ SARSA Converged: Avg(100): {avg:.2f}")
            self.converged = True
            self.epsilon = 0.0

    def learn(self, max_attempts=10000, param_set_name="default"):
        results_dir = os.path.join(os.path.dirname(__file__), 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        csv_filename = os.path.join(results_dir, f'results_{param_set_name}.csv')
        
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Attempt', 'Reward', 'Average_Reward', 'Epsilon'])
            
            for _ in range(max_attempts):
                r = self.run_attempt()
                self.reward_window.append(r)
                avg = np.mean(self.reward_window)
                
                writer.writerow([self.attempt, r, avg, self.epsilon])
                print(f"(SARSA) Ep: {self.attempt} | Reward: {r} | Avg: {avg:.2f} | ε: {self.epsilon:.4f}")
                
                if len(self.reward_window) == 100 and not self.converged:
                    self.check_convergence(avg)

                if not self.converged:
                    self.epsilon = max(self.params['epsilon_min'], self.epsilon * self.params['epsilon_decay'])
                    self.learning_rate = max(self.params['learning_rate_min'], 
                                          self.learning_rate * self.params['learning_rate_decay'])
                
                self.attempt += 1
        
        self.env.close()
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
        learner = SARSALearner(params)
        converged = learner.learn(10000, param_name)
        if not converged:
            print(f"❌ Algorithm did not converge within the maximum number of attempts for {param_name}")

if __name__ == '__main__':
    main()
