import math
import numpy as np
import gym
import os
import csv
from collections import deque

LEARNING_RATE = 0.5
LEARNING_RATE_DECAY = 0.9995
LEARNING_RATE_MIN = 0.05

EPSILON = 1.0
EPSILON_DECAY = 0.999
EPSILON_MIN = 0.01

DISCOUNT = 0.99
BUCKETS = (1, 1, 6, 12)


class QLearner:
    def __init__(self):
        self.env = gym.make('CartPole-v1', render_mode=None)
        self.learning_rate = LEARNING_RATE
        self.epsilon = EPSILON
        self.q_table = np.ones(BUCKETS + (self.env.action_space.n,))  # Optimistic init

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
            int(round((BUCKETS[i] - 1) * ratios[i])) for i in range(len(BUCKETS))
        )

    def pick_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        return int(np.argmax(self.q_table[state]))

    def update_knowledge(self, s, a, r, s_next):
        q_current = self.q_table[s + (a,)]
        q_future = np.max(self.q_table[s_next])
        td_target = r + DISCOUNT * q_future
        self.q_table[s + (a,)] += self.learning_rate * (td_target - q_current)

    def run_attempt(self):
        state, _ = self.env.reset()
        s = self.discretise(state)
        total_reward = 0
        done = False

        while not done:
            a = self.pick_action(s)
            state2, reward, terminated, truncated, _ = self.env.step(a)
            s2 = self.discretise(state2)

            self.update_knowledge(s, a, reward, s2)

            s = s2
            total_reward += reward
            done = terminated or truncated

        return total_reward

    def check_convergence(self, avg):
        if abs(avg - self.best_avg) < self.delta:
            self.stable += 1
        else:
            self.stable = 0
            self.best_avg = avg

        if self.stable >= self.stability_thresh:
            print(f"✅ Q-Learning Converged: Avg(100): {avg:.2f}")
            self.converged = True
            self.epsilon = 0.0

    def learn(self, max_attempts=10000):
        os.makedirs("results", exist_ok=True)
        with open("results/results.csv", "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Attempt", "Reward", "Average_Reward", "Epsilon"])
            for _ in range(max_attempts):
                r = self.run_attempt()
                self.reward_window.append(r)
                avg = np.mean(self.reward_window)

                writer.writerow([self.attempt, r, avg, self.epsilon])
                print(f"(Q-Learn) Ep: {self.attempt} | Reward: {r} | Avg: {avg:.2f} | ε: {self.epsilon:.4f}")

                if len(self.reward_window) == 100 and not self.converged:
                    self.check_convergence(avg)

                if not self.converged:
                    self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
                    self.learning_rate = max(LEARNING_RATE_MIN, self.learning_rate * LEARNING_RATE_DECAY)

                self.attempt += 1

        self.env.close()
        return self.converged
      

if __name__ == '__main__':
    learner = QLearner()
    converged = learner.learn(10000)

    if not converged:
        print("❌ Q-Learning did not converge within the maximum number of attempts")
