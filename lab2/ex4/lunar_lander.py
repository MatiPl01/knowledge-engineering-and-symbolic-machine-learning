import math
import numpy as np
import gym
import os
import csv
from collections import deque

class SARSALearner:
    def __init__(self):
        self.env = gym.make('LunarLander-v2')
        self.params = {
            'learning_rate': 0.1,
            'learning_rate_decay': 0.9999,
            'learning_rate_min': 0.01,
            'epsilon': 1.0,
            'epsilon_decay': 0.99995,
            'epsilon_min': 0.05,
            'discount_factor': 0.99,
            'buckets': (12, 12, 12, 12, 12, 12, 2, 2)
        }
        self.q_table = np.zeros(self.params['buckets'] + (self.env.action_space.n,))
        self.learning_rate = self.params['learning_rate']
        self.epsilon = self.params['epsilon']
        self.attempt = 1
        self.reward_window = deque(maxlen=100)

        self.upper_bounds = [1.5, 1.5, 2.0, 2.0, math.pi, 5.0, 1.0, 1.0]
        self.lower_bounds = [-1.5, -0.5, -2.0, -2.0, -math.pi, -5.0, 0.0, 0.0]

    def discretise(self, obs):
        ratios = [
            (obs[i] - self.lower_bounds[i]) / (self.upper_bounds[i] - self.lower_bounds[i])
            for i in range(len(obs))
        ]
        ratios = [min(max(r, 0), 1) for r in ratios]
        return tuple(
            int((self.params['buckets'][i] - 1) * ratios[i]) for i in range(len(self.params['buckets']))
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
            done = terminated or truncated
            s2 = self.discretise(obs2)
            a2 = self.pick_action(s2)
            self.update_knowledge(s, a, reward, s2, a2)
            s, a = s2, a2
            total_reward += reward
        return total_reward

    def learn(self, max_attempts):
        results_dir = os.path.join(os.path.dirname(__file__), 'results')
        os.makedirs(results_dir, exist_ok=True)
        csv_filename = os.path.join(results_dir, 'results_disc_sarsa1.2.csv')
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Attempt', 'Reward', 'Average_Reward', 'Epsilon'])
            for _ in range(max_attempts):
                r = self.run_attempt()
                self.reward_window.append(r)
                avg = np.mean(self.reward_window)
                writer.writerow([self.attempt, r, avg, self.epsilon])
                print(f"Ep: {self.attempt} | Reward: {r:.2f} | Avg(100): {avg:.2f} | ε: {self.epsilon:.3f}")
                self.epsilon = max(self.params['epsilon_min'], self.epsilon * self.params['epsilon_decay'])
                self.learning_rate = max(self.params['learning_rate_min'], self.learning_rate * self.params['learning_rate_decay'])
                self.attempt += 1
        self.env.close()

def main():
    learner = SARSALearner()
    learner.learn(250000)

if __name__ == '__main__':
    main()