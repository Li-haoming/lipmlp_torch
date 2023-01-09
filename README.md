# Lipschitz MLP with PyTorch
(https://github.com/Li-haoming/lipmlp_torch)

This is the comparative experiments using different methods to verify their performance on three tasks including Attack Robustness, Shape Interpolation & Extrapolation, and Reconstruction with Test Time Optimization.

## Usage
No need to install additional libs.
1. 'ae_sdf.py' includes the whole process of verifying the adversarial robustness of a vanilla mlp. (Training, testing, adding adversarial attack, show robustness)
```
python ae_sdf.py
```
2. 'ae_lipmlp_sdf.py' includes the whole process of verifying the adversarial robustness of a Lipschitz mlp. (Training, testing, adding adversarial attack, show robustness)
```
python ae_lipmlp_sdf.py
```
## Problems
1. There is something wrong with the Lipschitz Regularization. During the training process, the MSE loss decreases as usual, however, the Lipschitz Regularization increases oddly.
## To Do
1. Apply Lipschitz Regularization and Weight Normalization to the DeepSDF.
2. Test time optimization.
