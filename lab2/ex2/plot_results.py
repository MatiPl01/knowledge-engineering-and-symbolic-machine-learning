import os
import pandas as pd
import matplotlib.pyplot as plt
import json

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

def plot_learning_curve(results, params, output_dir):
    """Plot learning curve with confidence interval."""
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
    
    ax1.set_title('Learning Progress')
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
    plt.savefig(os.path.join(output_dir, 'learning_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, 'results')
    param_file = os.path.join(script_dir, 'parameter_sets.json')
    
    # Load results and parameters
    results = load_results(results_dir)
    params = load_parameters(param_file)
    
    if not results:
        print("No results found. Please run the Q-learning script first.")
        return
    
    # Generate learning curve plot
    plot_learning_curve(results, params, results_dir)
    
    print(f"Results analysis complete. Plot saved to {results_dir}")

if __name__ == '__main__':
    main() 