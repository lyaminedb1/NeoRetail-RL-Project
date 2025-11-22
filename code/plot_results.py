import matplotlib.pyplot as plt
import json

print("Loading training history...")

# Load history
with open('training_history.json', 'r') as f:
    history = json.load(f)

print("Creating plots...")

# Create plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Loss
axes[0, 0].plot(range(1, 6), history['train_loss'], marker='o', linewidth=2, markersize=8)
axes[0, 0].set_title('Training Loss', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Epoch', fontsize=12)
axes[0, 0].set_ylabel('Loss', fontsize=12)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_xticks(range(1, 6))

# Entropy
axes[0, 1].plot(range(1, 6), history['train_entropy'], marker='o', color='orange', linewidth=2, markersize=8)
axes[0, 1].set_title('Policy Entropy', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Epoch', fontsize=12)
axes[0, 1].set_ylabel('Entropy', fontsize=12)
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_xticks(range(1, 6))

# CTR
axes[1, 0].plot(range(1, 6), history['val_avg_reward'], marker='o', label='Val CTR', linewidth=2, markersize=8)
axes[1, 0].axhline(y=0.1697, color='r', linestyle='--', linewidth=2, label='Baseline CTR')
axes[1, 0].set_title('Validation CTR', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Epoch', fontsize=12)
axes[1, 0].set_ylabel('CTR', fontsize=12)
axes[1, 0].legend(fontsize=10)
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_xticks(range(1, 6))

# IPS/WIS
axes[1, 1].plot(range(1, 6), history['val_ips'], marker='o', label='IPS', linewidth=2, markersize=8)
axes[1, 1].plot(range(1, 6), history['val_wis'], marker='s', label='WIS', linewidth=2, markersize=8)
axes[1, 1].axhline(y=0.1697, color='r', linestyle='--', linewidth=2, label='Baseline')
axes[1, 1].scatter([3], [0.1971], s=300, c='red', marker='*', zorder=5, label='Peak')
axes[1, 1].set_title('Off-Policy Evaluation', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Epoch', fontsize=12)
axes[1, 1].set_ylabel('Estimated CTR', fontsize=12)
axes[1, 1].legend(fontsize=10)
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_xticks(range(1, 6))

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
print('✅ Saved training_curves.png')
plt.show()

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Best Performance: Epoch 3")
print(f"  IPS: 0.1971 (19.71%)")
print(f"  WIS: 0.1903 (19.03%)")
print(f"  Improvement: +16.1%")
print("="*60)