"""
Fix feature_config.json to match the actual data
"""

import pandas as pd
import json

print("="*60)
print("Fixing feature_config.json")
print("="*60)

# Load all data to get accurate vocab sizes
print("\nLoading data to compute vocabulary sizes...")
train_df = pd.read_csv('train_encoded.csv')
val_df = pd.read_csv('val_encoded.csv')
test_df = pd.read_csv('test_encoded.csv')

# Combine to get global max
all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
print(f"Total samples: {len(all_df):,}")

# Check which columns exist
print(f"\nColumns in data: {all_df.columns.tolist()}")

# Define features based on what exists in the data
numerical_features = ['hour_of_day']

# High-cardinality categorical
high_cardinality_features = [
    'device_id_encoded',
    'device_ip_encoded',
    'device_model_encoded',
    'site_id_encoded',
    'site_domain_encoded',
    'app_id_encoded',
    'C14',
]

# Medium-cardinality categorical
medium_cardinality_features = [
    'C17',
    'app_domain_encoded',
    'C19',
    'C21',
]

# Low-cardinality categorical
low_cardinality_features = [
    'C1',
    'site_category_encoded',
    'app_category_encoded',
    'device_type',
    'device_conn_type',
    'C15',
    'C16',
    'C18',
]

# Context features (all except action, target, and derived columns)
context_features = (
    numerical_features + 
    high_cardinality_features + 
    medium_cardinality_features + 
    low_cardinality_features
)

print(f"\nContext features ({len(context_features)}):")
for feat in context_features:
    if feat in all_df.columns:
        print(f"  ✓ {feat}")
    else:
        print(f"  ✗ {feat} - NOT IN DATA!")

# Compute vocabulary sizes
vocab_sizes = {}
for feat in context_features:
    if feat != 'hour_of_day' and feat in all_df.columns:
        max_val = int(all_df[feat].max())
        vocab_sizes[feat] = max_val + 1

print("\nVocabulary sizes:")
for feat, size in vocab_sizes.items():
    print(f"  {feat}: {size:,}")

# Compute embedding dimensions
embedding_dims = {}

# High-cardinality
for feat in high_cardinality_features:
    if feat in vocab_sizes:
        vocab_size = vocab_sizes[feat]
        if vocab_size > 100_000:
            embedding_dims[feat] = 32
        elif vocab_size > 10_000:
            embedding_dims[feat] = 24
        else:
            embedding_dims[feat] = 16

# Medium-cardinality
for feat in medium_cardinality_features:
    if feat in vocab_sizes:
        vocab_size = vocab_sizes[feat]
        if vocab_size > 10_000:
            embedding_dims[feat] = 16
        elif vocab_size > 1_000:
            embedding_dims[feat] = 12
        else:
            embedding_dims[feat] = 8

# Low-cardinality
for feat in low_cardinality_features:
    if feat in vocab_sizes:
        vocab_size = vocab_sizes[feat]
        if vocab_size > 100:
            embedding_dims[feat] = 8
        elif vocab_size > 20:
            embedding_dims[feat] = 6
        else:
            embedding_dims[feat] = 4

print("\nEmbedding dimensions:")
for feat, dim in embedding_dims.items():
    print(f"  {feat}: {dim}")

# Calculate total embedding dimension
total_embed_dim = sum(embedding_dims.values()) + 1  # +1 for hour_of_day

print(f"\nTotal embedding dimension: {total_embed_dim}")

# Create config
config = {
    'action_feature': 'banner_pos',
    'n_actions': 7,
    'numerical_features': numerical_features,
    'high_cardinality_features': high_cardinality_features,
    'medium_cardinality_features': medium_cardinality_features,
    'low_cardinality_features': low_cardinality_features,
    'context_features': context_features,
    'vocab_sizes': vocab_sizes,
    'embedding_dims': embedding_dims,
    'total_embed_dim': total_embed_dim,
}

# Save
with open('feature_config.json', 'w') as f:
    json.dump(config, f, indent=2)

print("\n✅ Saved updated feature_config.json")
print(f"✅ Context features: {len(context_features)}")
print(f"✅ Total embedding dim: {total_embed_dim}")
print("\nYou can now run: python train_policy.py")
