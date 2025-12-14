# NeoRetail: Policy-Gradient RL for Banner Position Optimization

**Track A: Offline Reinforcement Learning / Contextual Bandits**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

##  Project Overview

This project implements a **REINFORCE-based policy gradient algorithm** to optimize banner ad position selection for maximizing click-through rate (CTR). Using offline RL on logged data from the Avazu CTR dataset, our learned policy achieved an **estimated 16.1% improvement** over the baseline.

### Key Results
- **Baseline CTR**: 16.97%
- **Learned Policy (IPS)**: 19.71%
- **Improvement**: +16.1% relative
- **Model**: 81.2M parameters
- **Training**: 5 epochs on 3.5M samples

---

##  Dataset

**Source:** [Avazu Click-Through Rate Prediction (Kaggle)](https://www.kaggle.com/c/avazu-ctr-prediction)

- **Original Size**: 40.4M samples
- **Our Sample**: 5M samples
- **Split**: 70% train / 15% validation / 15% test

**Why sample?** Computational efficiency while maintaining statistical significance and data diversity.

### Data Structure
```
20 context features → 7 banner positions → Binary click outcome
- Device: device_id, device_ip, device_model, device_type, device_conn_type
- Site: site_id, site_domain, site_category
- App: app_id, app_domain, app_category
- Temporal: hour_of_day (0-23)
- Anonymous: C1, C14-C19, C21
- Action: banner_pos (positions 0-6)
- Target: click (0/1)
```

---

##  Results Summary

| Metric | Baseline | Our Policy | Improvement |
|--------|----------|------------|-------------|
| Validation IPS | 16.97% | **19.71%** | **+16.1%** |
| Validation WIS | 16.97% | **19.03%** | **+12.1%** |
| Test CTR | 16.97% | 16.97% | (logged data) |

**Peak Performance**: Epoch 3

### Key Findings

**Position 6 Discovery** (The Game Changer):
- Behavior policy: 0.1% usage, 32.89% CTR (best but ignored!)
- Our policy: ~50-60% usage (discovered and exploited)

**Position 0 Optimization**:
- Behavior policy: 72.1% usage, 16.45% CTR (overused, underperforming)
- Our policy: ~10-15% usage (reduced waste)

---

##  Architecture
```
Input (20 features) 
    ↓
Embeddings (241 dims)
    ↓
Shared Trunk (256 → 128)
    ↓
Policy Head (7 actions) + Value Head (baseline)
```

**Total Parameters**: 81,184,474

**Key Design Choices**:
- Variable-dimension embeddings (4-32 dims) based on feature cardinality
- Shared trunk for parameter efficiency
- Dropout (0.3) for regularization
- Policy + Value heads for variance reduction

---

##  Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Project Pipeline

**1. Exploratory Data Analysis**
```bash
python code/eda.py
```
Analyzes the dataset to identify opportunities (e.g., position 6 has 32.89% CTR but only 0.1% usage).

**2. Data Preprocessing**
```bash
python code/preprocessing.py
```
Transforms raw data into RL-ready format with proper encoding and propensity estimation.

**3. Training**
```bash
python code/train_policy.py
```
Trains REINFORCE policy for 5 epochs (~2.5 hours on CPU).

**4. Evaluation & Visualization**
```bash
python code/plot_results.py
python code/test_analysis.py
```
Generates training curves and test set analysis with statistical validation.

---

##  Project Structure
```
NeoRetail-RL-Project/
├── code/
│   ├── eda.py                   #  Exploratory data analysis
│   ├── preprocessing.py         #  Data preprocessing pipeline
│   ├── train_policy.py          #  REINFORCE training
│   ├── evaluate_policy.py       #  Policy evaluation
│   ├── plot_results.py          #  Visualization generation
│   ├── test_analysis.py         #  Statistical analysis
│   └── fix_config.py            #  Config utility
├── data/
│   ├── feature_config.json      # Feature configuration
│   ├── action_encoder.pkl       # Action mapping
│   └── categorical_encoders.pkl # Feature encoders
├── outputs/
│   ├── training_curves.png      # Training visualization
│   ├── test_analysis.png        # Test results
│   ├── training_history.json    # Metrics per epoch
│   └── test_analysis.json       # Final results
├── requirements.txt
└── README.md
```

---

##  Method

### Algorithm: REINFORCE + Baseline

**Loss Function**:
```
L = -log π(a|s)(R - V(s)) + 0.5(V(s) - R)² - β·H(π)
```

**Components**:
- **Policy gradient**: Learns action selection
- **Value baseline**: Reduces variance
- **Entropy bonus**: Maintains exploration

### Off-Policy Evaluation

Since we work with logged data (offline RL), we estimate policy performance using:
- **IPS (Inverse Propensity Scoring)**: Unbiased estimate
- **WIS (Weighted Importance Sampling)**: Lower variance
- **Bootstrap CIs**: Confidence intervals (95%)

---

##  Training Details

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adam |
| Learning Rate | 1e-4 |
| Batch Size | 512 |
| Epochs | 5 |
| Entropy Coefficient | 0.01 |
| Gradient Clipping | 1.0 |

**Training Time**: ~2.5 hours (5 epochs on CPU)

---

##  Key Insights

### What the Policy Learned

1. **Position 6 is Best** (32.89% CTR)
   - Behavior policy: Rarely used (0.1%)
   - Our policy: Heavily favored (50-60%)
   - **This is the main source of improvement!**

2. **Position 0 is Overused** (16.45% CTR)
   - Behavior policy: 72.1% usage
   - Our policy: Reduced to 10-15%
   - Major waste in baseline

3. **Context Matters**
   - Mobile devices → Position 6 (less intrusive)
   - Desktop → Balanced distribution
   - Sports sites → More position 6 (engaged users)
   - Evening hours → More position 6 (receptive users)

---

##  Visualizations

### Training Curves
![Training Curves](outputs/training_curves.png)

Shows loss evolution, policy entropy, validation CTR, and off-policy evaluation metrics across 5 epochs.

### Test Analysis
![Test Analysis](outputs/test_analysis.png)

Validates performance on held-out test set with performance comparison, action distribution, CTR by position, and final summary.

---

##  Limitations

### Off-Policy Evaluation
- **19.71% is an estimate**, not actual performance
- Requires A/B testing for true validation
- Assumptions: overlap, known propensities, stationary behavior




##  References
. [Avazu CTR Dataset](https://www.kaggle.com/c/avazu-ctr-prediction) - Kaggle

---

##  Author

**Abdellah Elyamine DALI BRAHAM**

Course: Advanced Machine Learning  
Project: NeoRetail Track A (Offline RL / Contextual Bandits)
