"""
Evaluation Script: Test Set Performance
Computes IPS, WIS, and analyzes policy behavior
"""

import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Import from training script
import sys
sys.path.append('.')
from train_policy import PolicyNetwork, BanditDataset, collate_fn, device

def compute_bootstrap_ci(data, n_bootstrap=1000, confidence=0.95):
    """Compute bootstrap confidence interval"""
    bootstrap_means = []
    n = len(data)
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))
    
    alpha = (1 - confidence) / 2
    lower = np.percentile(bootstrap_means, alpha * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha) * 100)
    
    return lower, upper

def evaluate_test_set(model, test_loader, device):
    """
    Comprehensive evaluation on test set
    """
    model.eval()
    
    all_rewards = []
    all_ips_rewards = []
    all_weights = []
    all_actions = []
    all_predicted_actions = []
    
    with torch.no_grad():
        for x_dict, actions, rewards, propensities in test_loader:
            # Move to device
            for feat in x_dict:
                x_dict[feat] = x_dict[feat].to(device)
            actions = actions.to(device)
            rewards = rewards.to(device)
            propensities = propensities.to(device)
            
            # Get policy
            logits, _ = model(x_dict)
            policy_probs = F.softmax(logits, dim=-1)
            
            # Predicted actions (greedy)
            predicted_actions = torch.argmax(logits, dim=-1)
            
            # Probability of taken actions
            pi_a = policy_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
            
            # Importance weights
            weights = (pi_a / propensities).clamp(max=50.0)
            
            # Store results
            all_rewards.extend(rewards.cpu().numpy())
            all_ips_rewards.extend((weights * rewards).cpu().numpy())
            all_weights.extend(weights.cpu().numpy())
            all_actions.extend(actions.cpu().numpy())
            all_predicted_actions.extend(predicted_actions.cpu().numpy())
    
    all_rewards = np.array(all_rewards)
    all_ips_rewards = np.array(all_ips_rewards)
    all_weights = np.array(all_weights)
    all_actions = np.array(all_actions)
    all_predicted_actions = np.array(all_predicted_actions)
    
    # Compute metrics
    n_samples = len(all_rewards)
    
    # Behavior policy CTR (from logged data)
    behavior_ctr = np.mean(all_rewards)
    
    # IPS estimate
    ips = np.mean(all_ips_rewards)
    ips_lower, ips_upper = compute_bootstrap_ci(all_ips_rewards)
    
    # WIS estimate
    wis = np.sum(all_ips_rewards) / np.sum(all_weights)
    
    # On-policy estimate (if we actually deployed this policy)
    # This is just for reference - we can't actually know this from logged data
    
    results = {
        'n_samples': n_samples,
        'behavior_ctr': behavior_ctr,
        'ips': ips,
        'ips_ci': (ips_lower, ips_upper),
        'wis': wis,
        'effective_sample_size': np.sum(all_weights)**2 / np.sum(all_weights**2),
    }
    
    return results, all_actions, all_predicted_actions, all_rewards, all_weights

def analyze_policy_behavior(model, test_loader, config, device):
    """
    Analyze what the policy learned
    """
    model.eval()
    
    action_counts = np.zeros(config['n_actions'])
    action_rewards = np.zeros(config['n_actions'])
    action_sample_counts = np.zeros(config['n_actions'])
    
    with torch.no_grad():
        for x_dict, actions, rewards, propensities in test_loader:
            # Move to device
            for feat in x_dict:
                x_dict[feat] = x_dict[feat].to(device)
            actions = actions.cpu().numpy()
            rewards = rewards.cpu().numpy()
            
            # Get predicted actions
            logits, _ = model(x_dict)
            predicted_actions = torch.argmax(logits, dim=-1).cpu().numpy()
            
            # Count predictions
            for a in predicted_actions:
                action_counts[a] += 1
            
            # Track rewards per action (from logged data)
            for a, r in zip(actions, rewards):
                action_rewards[a] += r
                action_sample_counts[a] += 1
    
    # Normalize
    action_distribution = action_counts / action_counts.sum()
    action_ctr = action_rewards / (action_sample_counts + 1e-10)
    
    return action_distribution, action_ctr

def main():
    print("="*60)
    print("Test Set Evaluation")
    print("="*60)
    
    # Load configuration
    with open('feature_config.json', 'r') as f:
        config = json.load(f)
    
    # Load action encoder
    import pickle
    with open('action_encoder.pkl', 'rb') as f:
        action_encoder = pickle.load(f)
    
    # Load test dataset
    print("\nLoading test dataset...")
    test_dataset = BanditDataset('test_encoded.csv', config)
    test_loader = DataLoader(
        test_dataset,
        batch_size=512,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    print(f"  Test samples: {len(test_dataset):,}")
    
    # Load model
    print("\nLoading trained model...")
    model = PolicyNetwork(config)
    model.load_state_dict(torch.load('policy_model.pth', map_location=device))
    model = model.to(device)
    print("  Model loaded successfully")
    
    # Evaluate
    print("\nEvaluating on test set...")
    results, all_actions, all_predicted_actions, all_rewards, all_weights = evaluate_test_set(
        model, test_loader, device
    )
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Test samples: {results['n_samples']:,}")
    print(f"\nBehavior Policy CTR: {results['behavior_ctr']:.4f}")
    print(f"\nLearned Policy Estimates:")
    print(f"  IPS: {results['ips']:.4f} (95% CI: [{results['ips_ci'][0]:.4f}, {results['ips_ci'][1]:.4f}])")
    print(f"  WIS: {results['wis']:.4f}")
    print(f"  Effective Sample Size: {results['effective_sample_size']:.0f}")
    
    # Improvement
    improvement_ips = ((results['ips'] - results['behavior_ctr']) / results['behavior_ctr']) * 100
    improvement_wis = ((results['wis'] - results['behavior_ctr']) / results['behavior_ctr']) * 100
    
    print(f"\nEstimated Improvement:")
    print(f"  IPS: {improvement_ips:+.2f}%")
    print(f"  WIS: {improvement_wis:+.2f}%")
    
    # Analyze policy behavior
    print("\nAnalyzing policy behavior...")
    action_distribution, action_ctr = analyze_policy_behavior(model, test_loader, config, device)
    
    print("\nLearned Policy Action Distribution:")
    for action_id in range(config['n_actions']):
        banner_pos = action_encoder.classes_[action_id]
        print(f"  Banner Pos {banner_pos}: {action_distribution[action_id]:.4f} (CTR from logs: {action_ctr[action_id]:.4f})")
    
    # Save results
    with open('test_results.json', 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        json_results = {
            'n_samples': int(results['n_samples']),
            'behavior_ctr': float(results['behavior_ctr']),
            'ips': float(results['ips']),
            'ips_ci_lower': float(results['ips_ci'][0]),
            'ips_ci_upper': float(results['ips_ci'][1]),
            'wis': float(results['wis']),
            'effective_sample_size': float(results['effective_sample_size']),
            'improvement_ips_percent': float(improvement_ips),
            'improvement_wis_percent': float(improvement_wis),
            'action_distribution': action_distribution.tolist(),
            'action_ctr': action_ctr.tolist(),
        }
        json.dump(json_results, f, indent=2)
    print("\n✅ Saved test_results.json")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Action distribution comparison
    x = np.arange(config['n_actions'])
    width = 0.35
    
    # Behavior policy distribution (from test data)
    behavior_dist = np.bincount(all_actions, minlength=config['n_actions'])
    behavior_dist = behavior_dist / behavior_dist.sum()
    
    axes[0, 0].bar(x - width/2, behavior_dist, width, label='Behavior Policy', alpha=0.7)
    axes[0, 0].bar(x + width/2, action_distribution, width, label='Learned Policy', alpha=0.7)
    axes[0, 0].set_xlabel('Action (Banner Position)')
    axes[0, 0].set_ylabel('Probability')
    axes[0, 0].set_title('Action Distribution: Behavior vs Learned Policy')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels([action_encoder.classes_[i] for i in range(config['n_actions'])])
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. CTR by action
    axes[0, 1].bar(x, action_ctr, alpha=0.7, color='green')
    axes[0, 1].axhline(y=results['behavior_ctr'], color='r', linestyle='--', label='Overall CTR')
    axes[0, 1].set_xlabel('Action (Banner Position)')
    axes[0, 1].set_ylabel('CTR')
    axes[0, 1].set_title('CTR by Banner Position (from logged data)')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels([action_encoder.classes_[i] for i in range(config['n_actions'])])
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Importance weight distribution
    axes[1, 0].hist(all_weights, bins=50, alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(x=1.0, color='r', linestyle='--', label='Weight = 1')
    axes[1, 0].set_xlabel('Importance Weight')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Importance Weights')
    axes[1, 0].set_xlim([0, 10])
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Policy improvement comparison
    methods = ['Behavior\nPolicy', 'Learned\n(IPS)', 'Learned\n(WIS)']
    values = [results['behavior_ctr'], results['ips'], results['wis']]
    colors = ['gray', 'blue', 'green']
    
    bars = axes[1, 1].bar(methods, values, color=colors, alpha=0.7, edgecolor='black')
    axes[1, 1].set_ylabel('Estimated CTR')
    axes[1, 1].set_title('Policy Comparison')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.4f}',
                       ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('test_evaluation.png', dpi=150)
    print("  Saved test_evaluation.png")
    
    print("\n" + "="*60)
    print("Evaluation complete!")
    print("="*60)

if __name__ == "__main__":
    main()
