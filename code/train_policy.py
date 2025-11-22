"""
NeoRetail: Policy-Gradient RL for Banner Position Optimization
Track A: Offline RL / Contextual Bandits

This script implements REINFORCE + baseline for learning which banner position
maximizes click-through rate given user context.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import json
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

# Set seeds for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

set_seed(42)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================================
# 1. DATASET CLASS
# ============================================================================

class BanditDataset(Dataset):
    """Dataset for contextual bandit problem"""
    
    def __init__(self, csv_path, config):
        print(f"Loading {csv_path}...")
        self.df = pd.read_csv(csv_path)
        self.config = config
        self.context_features = config['context_features']
        print(f"  Loaded {len(self.df):,} samples")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Context features
        x_dict = {}
        for feat in self.context_features:
            x_dict[feat] = int(row[feat])
        
        # Action, reward, propensity
        action = int(row['action_id'])
        reward = float(row['click'])
        propensity = float(row['propensity'])
        
        return x_dict, action, reward, propensity

def collate_fn(batch):
    """Custom collate function for dictionary of features"""
    x_dicts, actions, rewards, propensities = zip(*batch)
    
    # Stack each feature
    batch_x_dict = {}
    for feat in x_dicts[0].keys():
        batch_x_dict[feat] = torch.tensor([x[feat] for x in x_dicts], dtype=torch.long)
    
    batch_actions = torch.tensor(actions, dtype=torch.long)
    batch_rewards = torch.tensor(rewards, dtype=torch.float32)
    batch_propensities = torch.tensor(propensities, dtype=torch.float32)
    
    return batch_x_dict, batch_actions, batch_rewards, batch_propensities

# ============================================================================
# 2. POLICY NETWORK
# ============================================================================

class PolicyNetwork(nn.Module):
    """
    Policy network with:
    - Embedding layers for categorical features
    - Shared trunk
    - Policy head (outputs action logits)
    - Value head (outputs state value for baseline)
    """
    
    def __init__(self, config):
        super(PolicyNetwork, self).__init__()
        
        self.config = config
        self.vocab_sizes = config['vocab_sizes']
        self.embedding_dims = config['embedding_dims']
        self.n_actions = config['n_actions']
        
        # Create embedding layers
        self.embeddings = nn.ModuleDict()
        for feat in config['context_features']:
            if feat != 'hour_of_day':
                vocab_size = self.vocab_sizes[feat]
                embed_dim = self.embedding_dims[feat]
                self.embeddings[feat] = nn.Embedding(vocab_size, embed_dim)
        
        # Shared trunk
        input_dim = config['total_embed_dim']
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        
        # Policy head
        self.policy_head = nn.Linear(128, self.n_actions)
        
        # Value head
        self.value_head = nn.Linear(128, 1)
        
    def forward(self, x_dict):
        """
        Forward pass
        Args:
            x_dict: dictionary of tensors {feature_name: tensor}
        Returns:
            logits: (batch_size, n_actions)
            value: (batch_size,)
        """
        embed_list = []
        
        # Embed all features
        for feat in self.config['context_features']:
            if feat == 'hour_of_day':
                # Normalize numerical feature
                hour_norm = x_dict[feat].float() / 24.0
                embed_list.append(hour_norm.unsqueeze(1))
            else:
                # Embed categorical feature
                embedded = self.embeddings[feat](x_dict[feat])
                embed_list.append(embedded)
        
        # Concatenate embeddings
        x = torch.cat(embed_list, dim=1)
        
        # Shared trunk
        trunk_out = self.trunk(x)
        
        # Policy and value heads
        logits = self.policy_head(trunk_out)
        value = self.value_head(trunk_out).squeeze(-1)
        
        return logits, value

# ============================================================================
# 3. REINFORCE TRAINING STEP
# ============================================================================

def reinforce_step(model, optimizer, x_dict, actions, rewards, entropy_coef=0.01):
    """
    Single REINFORCE training step with baseline
    
    Loss = -log π(a|s) * (R - V(s)) + 0.5 * (V(s) - R)^2 - β * H(π)
    """
    # Forward pass
    logits, values = model(x_dict)
    
    # Create policy distribution
    policy = Categorical(logits=logits)
    
    # Log probabilities of taken actions
    log_probs = policy.log_prob(actions)
    
    # Entropy
    entropy = policy.entropy().mean()
    
    # Advantages (with detached values for policy gradient)
    with torch.no_grad():
        advantages = rewards - values
    
    # Policy gradient loss
    policy_loss = -(log_probs * advantages).mean()
    
    # Value function loss
    value_loss = 0.5 * F.mse_loss(values, rewards)
    
    # Total loss
    loss = policy_loss + value_loss - entropy_coef * entropy
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Gradient clipping
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    return {
        'loss': loss.item(),
        'policy_loss': policy_loss.item(),
        'value_loss': value_loss.item(),
        'entropy': entropy.item(),
        'mean_advantage': advantages.mean().item(),
    }

# ============================================================================
# 4. EVALUATION FUNCTIONS
# ============================================================================

def evaluate_policy(model, data_loader, device):
    """
    Evaluate policy on validation set
    Returns:
        - Average reward (CTR)
        - IPS estimate
        - WIS estimate
    """
    model.eval()
    
    total_reward = 0
    ips_sum = 0
    wis_numerator = 0
    wis_denominator = 0
    n_samples = 0
    
    with torch.no_grad():
        for x_dict, actions, rewards, propensities in data_loader:
            # Move to device
            for feat in x_dict:
                x_dict[feat] = x_dict[feat].to(device)
            actions = actions.to(device)
            rewards = rewards.to(device)
            propensities = propensities.to(device)
            
            # Get policy probabilities
            logits, _ = model(x_dict)
            policy_probs = F.softmax(logits, dim=-1)
            
            # Get probability of taken actions
            pi_a = policy_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
            
            # Importance weights (clip for stability)
            weights = (pi_a / propensities).clamp(max=50.0)
            
            # Accumulate metrics
            total_reward += rewards.sum().item()
            ips_sum += (weights * rewards).sum().item()
            wis_numerator += (weights * rewards).sum().item()
            wis_denominator += weights.sum().item()
            n_samples += len(rewards)
    
    # Compute metrics
    avg_reward = total_reward / n_samples
    ips = ips_sum / n_samples
    wis = wis_numerator / wis_denominator if wis_denominator > 0 else 0
    
    return {
        'avg_reward': avg_reward,
        'ips': ips,
        'wis': wis,
        'n_samples': n_samples
    }

# ============================================================================
# 5. TRAINING LOOP
# ============================================================================

def train_model(model, train_loader, val_loader, config, n_epochs=10, lr=1e-4, entropy_coef=0.01):
    """
    Main training loop
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    history = {
        'train_loss': [],
        'train_entropy': [],
        'val_avg_reward': [],
        'val_ips': [],
        'val_wis': [],
    }
    
    for epoch in range(n_epochs):
        model.train()
        epoch_metrics = {
            'loss': [],
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'mean_advantage': [],
        }
        
        # Training
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}")
        for x_dict, actions, rewards, propensities in pbar:
            # Move to device
            for feat in x_dict:
                x_dict[feat] = x_dict[feat].to(device)
            actions = actions.to(device)
            rewards = rewards.to(device)
            
            # Training step
            metrics = reinforce_step(model, optimizer, x_dict, actions, rewards, entropy_coef)
            
            # Accumulate metrics
            for key, value in metrics.items():
                epoch_metrics[key].append(value)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{np.mean(epoch_metrics['loss']):.4f}",
                'entropy': f"{np.mean(epoch_metrics['entropy']):.4f}"
            })
        
        # Epoch summary
        avg_loss = np.mean(epoch_metrics['loss'])
        avg_entropy = np.mean(epoch_metrics['entropy'])
        
        # Validation
        val_metrics = evaluate_policy(model, val_loader, device)
        
        # Store history
        history['train_loss'].append(avg_loss)
        history['train_entropy'].append(avg_entropy)
        history['val_avg_reward'].append(val_metrics['avg_reward'])
        history['val_ips'].append(val_metrics['ips'])
        history['val_wis'].append(val_metrics['wis'])
        
        print(f"\nEpoch {epoch+1}/{n_epochs}")
        print(f"  Train Loss: {avg_loss:.4f}")
        print(f"  Train Entropy: {avg_entropy:.4f}")
        print(f"  Val Avg Reward (CTR): {val_metrics['avg_reward']:.4f}")
        print(f"  Val IPS: {val_metrics['ips']:.4f}")
        print(f"  Val WIS: {val_metrics['wis']:.4f}")
        print("-" * 60)
    
    return history

# ============================================================================
# 6. MAIN EXECUTION
# ============================================================================

def main():
    print("="*60)
    print("NeoRetail: Policy-Gradient RL Training")
    print("="*60)
    
    # Load configuration
    print("\nLoading configuration...")
    with open('feature_config.json', 'r') as f:
        config = json.load(f)
    
    print(f"  Actions: {config['n_actions']}")
    print(f"  Context features: {len(config['context_features'])}")
    print(f"  Total embedding dim: {config['total_embed_dim']}")
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = BanditDataset('train_encoded.csv', config)
    val_dataset = BanditDataset('val_encoded.csv', config)
    
    # Create dataloaders
    batch_size = 512
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    print(f"  Train batches: {len(train_loader):,}")
    print(f"  Val batches: {len(val_loader):,}")
    
    # Create model
    print("\nCreating model...")
    model = PolicyNetwork(config)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    # Train model
    print("\nStarting training...")
    history = train_model(
        model,
        train_loader,
        val_loader,
        config,
        n_epochs=5,  # Start with 5 epochs for testing
        lr=1e-4,
        entropy_coef=0.01
    )
    
    # Save model
    print("\nSaving model...")
    torch.save(model.state_dict(), 'policy_model.pth')
    print("  Saved policy_model.pth")
    
    # Save history
    with open('training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    print("  Saved training_history.json")
    
    # Plot results
    print("\nGenerating plots...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'])
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True)
    
    # Entropy
    axes[0, 1].plot(history['train_entropy'])
    axes[0, 1].set_title('Policy Entropy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Entropy')
    axes[0, 1].grid(True)
    
    # CTR
    axes[1, 0].plot(history['val_avg_reward'], label='Val CTR')
    axes[1, 0].axhline(y=0.1697, color='r', linestyle='--', label='Baseline CTR')
    axes[1, 0].set_title('Validation CTR')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('CTR')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # IPS/WIS
    axes[1, 1].plot(history['val_ips'], label='IPS')
    axes[1, 1].plot(history['val_wis'], label='WIS')
    axes[1, 1].axhline(y=0.1697, color='r', linestyle='--', label='Baseline')
    axes[1, 1].set_title('Off-Policy Evaluation')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Estimated Reward')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150)
    print("  Saved training_curves.png")
    
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)

if __name__ == "__main__":
    main()
