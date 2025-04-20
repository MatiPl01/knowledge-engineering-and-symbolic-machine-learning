import os
import pandas as pd
import matplotlib.pyplot as plt
import json

def plot_default_results(results_dir, output_dir):
    """Plot learning curve for the default results file."""
    default_file = os.path.join(results_dir, 'results_default.csv')
    
    if not os.path.exists(default_file):
        print(f"Default results file not found: {default_file}")
        return
    
    # Read the CSV file
    data = pd.read_csv(default_file)
    
    # Find the point where epsilon becomes 0 (convergence)
    convergence_point = data[data['Epsilon'] == 0].index[0] if 0 in data['Epsilon'].values else len(data)
    
    # Trim the data to show only until convergence
    data = data.iloc[:convergence_point + 100]  # Include 100 more points after convergence
    
    # Calculate moving averages and standard deviation
    window_size = 100
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
    plt.title('Learning Progress (Default Parameters)', fontsize=14)
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
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, 'default_learning_curve.png'), dpi=300)
    plt.close()
    
    print(f"Default plot saved to {os.path.join(output_dir, 'default_learning_curve.png')}")
    print(f"Training converged at episode {convergence_point}")

def load_results(results_dir):
    """Load all result files from the results directory."""
    results = {}
    
    for filename in os.listdir(results_dir):
        if filename.startswith('results_') and filename.endswith('.csv'):
            param_set_name = filename.replace('results_', '').replace('.csv', '')
            results[param_set_name] = pd.read_csv(os.path.join(results_dir, filename))
    
    return results

def load_parameters(param_file):
    """Load parameters from the JSON file."""
    with open(param_file, 'r') as f:
        return json.load(f)

def plot_all_results(results, params, output_dir):
    """Plot learning curve with confidence interval for all parameter sets."""
    plt.figure(figsize=(12, 8))
    
    # Create two subplots
    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[2, 1])
    
    # Plot learning progress
    for param_set_name, data in results.items():
        # Calculate moving average and standard deviation
        window = 100
        rewards = data['Reward'].values
        episodes = data['Attempt'].values
        
        # Calculate rolling statistics
        rolling_mean = pd.Series(rewards).rolling(window=window, min_periods=1).mean()
        rolling_std = pd.Series(rewards).rolling(window=window, min_periods=1).std()
        
        # Create a descriptive label with parameter information
        if param_set_name in params:
            param_info = params[param_set_name]
            label = f"{param_set_name} (lr={param_info['learning_rate']}, γ={param_info['discount_factor']})"
        else:
            label = param_set_name
        
        # Plot mean and confidence interval
        ax1.plot(episodes, rolling_mean, label=label)
        ax1.fill_between(episodes, 
                        rolling_mean - rolling_std, 
                        rolling_mean + rolling_std, 
                        alpha=0.2)
    
    ax1.set_title('Learning Progress (All Parameter Sets)')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=9)
    
    # Plot epsilon decay
    for param_set_name, data in results.items():
        # Use the same label as above for consistency
        if param_set_name in params:
            param_info = params[param_set_name]
            label = f"{param_set_name} (lr={param_info['learning_rate']}, γ={param_info['discount_factor']})"
        else:
            label = param_set_name
            
        ax2.plot(data['Attempt'], data['Epsilon'], label=label)
    
    ax2.set_title('Exploration Rate (Epsilon)')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Epsilon')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_learning_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"All results plot saved to {os.path.join(output_dir, 'all_learning_curves.png')}")

def main():
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, 'results')
    param_file = os.path.join(script_dir, 'parameter_sets.json')
    
    # Ensure the results directory exists
    os.makedirs(results_dir, exist_ok=True)
    
    # Plot default results
    plot_default_results(results_dir, results_dir)
    
    # Load results and parameters for all plots
    results = load_results(results_dir)
    params = load_parameters(param_file)
    
    if not results:
        print("No results found. Please run the SARSA script first.")
        return
    
    # Generate learning curve plot for all parameter sets
    plot_all_results(results, params, results_dir)
    
    print("Results analysis complete.")

if __name__ == '__main__':
    main() 