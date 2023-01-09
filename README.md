# Lipschitz MLP with PyTorch
(https://github.com/Li-haoming/lipmlp_torch)

This is the comparative experiments using different methods to verify their performance on three tasks including Attack Robustness, Shape Interpolation & Extrapolation, and Reconstruction with Test Time Optimization.

## Usage
1. 'ae_sdf.py' includes the whole process of verifying the adversarial robustness of a vanilla mlp. (Training, testing, adding adversarial attack, show robustness)
"""
python ae_sdf.py
"""
2. 'ae_lipmlp_sdf.py' includes the whole process of verifying the adversarial robustness of a lipschitz mlp. (Training, testing, adding adversarial attack, show robustness)
"""
python ae_lipmlp_sdf.py
"""
