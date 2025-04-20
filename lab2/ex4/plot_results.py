import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_sarsa_learning_curve(data_path='results/results.csv', window_size=100):
    # Read the CSV file
    data = pd.read_csv(data_path)
    
    # Find the point where epsilon becomes 0 (convergence point)
    convergence_point = data[data['Epsilon'] == 0].index[0] if len(data[data['Epsilon'] == 0]) > 0 else len(data)
    
    # Trim the data to show only until convergence
    data = data.iloc[:convergence_point + window_size]  # Include window_size more points after convergence
    
    # Calculate moving averages and standard deviation
    rewards_ma = data['Reward'].rolling(window=window_size).mean()
    rewards_std = data['Reward'].rolling(window=window_size).std()
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot rewards with moving average and standard deviation corridor
    plt.subplot(2, 1, 1)
    plt.plot(rewards_ma, label='Moving Average', linewidth=2, color='blue')
    plt.fill_between(range(len(rewards_ma)), 
                    rewards_ma - rewards_std,
                    rewards_ma + rewards_std,
                    alpha=0.2, color='blue')
    plt.title('SARSA Learning Progress', fontsize=14)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    # Plot epsilon
    plt.subplot(2, 1, 2)
    plt.plot(data['Epsilon'], label='Epsilon', color='green')
    plt.title('Exploration Rate (Epsilon)', fontsize=14)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Epsilon', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    
    # Ensure the results directory exists
    os.makedirs('results', exist_ok=True)
    
    # Save the figure
    plt.savefig('results/sarsa_learning_curve.png', dpi=300)
    plt.close()
    
    print("Plot saved to results/sarsa_learning_curve.png")
    if convergence_point < len(data):
        print(f"Training converged at episode {convergence_point}")
    else:
        print("Training did not converge within the recorded episodes")

if __name__ == '__main__':
    plot_sarsa_learning_curve() 