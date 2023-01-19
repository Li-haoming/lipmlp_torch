# Lipschitz MLP with PyTorch
(https://github.com/Li-haoming/lipmlp_torch)

Learning Smooth Neural Functions via Lipschitz Regularization Hsueh-Ti Derek Liu, Francis Williams, Alec Jacobson, Sanja Fidler, Or Litany SIGGRAPH (North America), 2022 (https://nv-tlabs.github.io/lip-mlp/)

This is the comparative experiments using different methods to verify their performance on two tasks including Attack Robustness, Shape Interpolation & Extrapolation.

## Usage
No need to install additional libs.
1. `ae_sdf.py` includes the whole process of verifying the adversarial robustness of a vanilla mlp. (Training, testing, adding adversarial attack, show robustness)
```
python ae_sdf.py
```
One can check the loss plots and adversarial images in `lipmlp_torch/adversarial_robustness/vanilla_autoencoder/` . Model parameters are in 'ae_params_1.pth'
2. `ae_lipmlp_sdf.py` includes the whole process of verifying the adversarial robustness of a Lipschitz mlp. (Training, testing, adding adversarial attack, show robustness)
```
python ae_lipmlp_sdf.py
```
One can check the loss plots and adversarial images in `lipmlp_torch/adversarial_robustness/lipschitz_autoencoder/` . Model parameters are in `ae_lipmlp_params_1.pth`
3. `deepsdf_2d.py` includes the whole process of 2d interpolation using a vanilla DeepSDF. (Training, testing, showing performance on MNIST)
```
python deepsdf_2d.py
```
One can check the loss plots and interpolation images in `lipmlp_torch/2d_interpolation/deepsdf/` .
4. `deepsdf_2d.py` includes the whole process of 3d interpolation using a vanilla DeepSDF.
```
python deepsdf_3d.py
```
One can check the loss plots in `lipmlp_torch/3d_interpolation/` .
## Problems
1. There is something wrong with the Lipschitz Regularization. During the training process, the MSE loss decreases as usual, however, the Lipschitz Regularization increases oddly. The adversarial images look well though.
2. How to visualize 3D SDF?
## To Do
1. Apply Lipschitz Regularization and Weight Normalization to the DeepSDF.
2. 
