# Score SDE PyTorch Project Analysis

## 1. Project Overview

This is a PyTorch implementation of **Score-Based Generative Modeling through Stochastic Differential Equations (SDEs)**, based on the paper by Yang Song et al. The codebase provides a unified framework for training and evaluating score-based generative models using various SDE types and model architectures.

### Key Features
- Supports multiple SDE types: VESDE, VPSDE, subVPSDE
- Implements various model architectures: NCSN, NCSNv2, NCSN++, DDPM, DDPM++
- Provides training, evaluation, and sampling functionalities
- Supports unconditional and conditional generation
- Includes exact likelihood computation
- Enables latent code manipulation

## 2. Project Structure

```
├── assets/                  # Sample images and statistics
├── configs/                 # Configuration files
│   ├── subvp/               # Sub-VP SDE configs
│   ├── ve/                  # Variance Exploding SDE configs
│   └── vp/                  # Variance Preserving SDE configs
├── models/                  # Model architectures
├── op/                      # Custom operations
├── main.py                  # Entry point
├── run_lib.py               # Training and evaluation logic
├── losses.py                # Loss functions
├── sde_lib.py               # SDE definitions
├── sampling.py              # Sampling algorithms
├── datasets.py              # Data loading
├── evaluation.py            # Evaluation metrics
└── likelihood.py            # Likelihood computation
```

## 3. Training Flow

### 3.1 Entry Point
The training process starts in `main.py`, which parses command-line arguments and calls either `run_lib.train()` or `run_lib.evaluate()` based on the specified mode.

### 3.2 Training Pipeline
The `run_lib.train()` function implements the complete training pipeline:

1. **Initialization**
   - Create directories for logs, samples, and checkpoints
   - Initialize TensorBoard writer
   - Create score model using `mutils.create_model(config)`
   - Initialize EMA (Exponential Moving Average) for model parameters
   - Set up optimizer using `losses.get_optimizer(config)`
   - Restore from checkpoint if available

2. **Data Loading**
   - Load dataset using `datasets.get_dataset(config)`
   - Create data scaler and inverse scaler

3. **SDE Setup**
   - Initialize the appropriate SDE based on `config.training.sde`:
     - `vesde`: Variance Exploding SDE
     - `vpsde`: Variance Preserving SDE
     - `subvpsde`: Sub-Variance Preserving SDE

4. **Training Step Function**
   - Create the training step function using `losses.get_step_fn()`
   - This function handles:
     - Loss computation
     - Backward pass
     - Parameter update
     - EMA update

5. **Training Loop**
   For each iteration from `initial_step` to `num_train_steps`:
   - Get a batch of data
   - Normalize the data using the scaler
   - Execute one training step
   - Log training loss
   - Save intermediate checkpoints for preemption
   - Evaluate on validation set periodically
   - Save full checkpoints and generate samples periodically

### 3.3 Loss Calculation
The core loss is implemented in `losses.get_sde_loss_fn()`:

1. **Sample Time Steps**: Randomly sample time steps `t` from [eps, T]
2. **Perturb Data**: Add noise to data according to the SDE's marginal distribution
3. **Compute Score**: Get the score prediction from the model
4. **Calculate Loss**: Compute the score matching loss
5. **Weight Loss**: Apply likelihood weighting if specified

### 3.4 Optimization
- **Optimizer**: Adam optimizer with configurable learning rate, beta1, and weight decay
- **Learning Rate Warmup**: Linear warmup from 0 to the specified learning rate
- **Gradient Clipping**: Optional gradient clipping to prevent exploding gradients

## 4. Configuration Parameters

### 4.1 Training Parameters (`config.training`)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `batch_size` | Batch size for training | 128 |
| `n_iters` | Total number of training iterations | 1300001 |
| `snapshot_freq` | Frequency to save full checkpoints | 50000 |
| `log_freq` | Frequency to log training loss | 50 |
| `eval_freq` | Frequency to evaluate on validation set | 100 |
| `snapshot_freq_for_preemption` | Frequency to save intermediate checkpoints | 10000 |
| `snapshot_sampling` | Whether to generate samples at each checkpoint | True |
| `likelihood_weighting` | Whether to use likelihood weighting for loss | False |
| `continuous` | Whether the model uses continuous time steps | True |
| `reduce_mean` | Whether to average loss across data dimensions | False |
| `sde` | Type of SDE to use (vesde/vpsde/subvpsde) | vesde |

### 4.2 Model Parameters (`config.model`)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `name` | Model architecture (ncsn/ncsnv2/ncsnpp/ddpm/ddpmpp) | ncsnpp |
| `scale_by_sigma` | Whether to scale model output by sigma | True |
| `ema_rate` | EMA decay rate | 0.999 |
| `normalization` | Normalization method (GroupNorm/BatchNorm) | GroupNorm |
| `nonlinearity` | Activation function | swish |
| `nf` | Number of filters in the first layer | 128 |
| `ch_mult` | Channel multiplier for each resolution level | (1, 2, 2, 2) |
| `num_res_blocks` | Number of residual blocks per resolution | 4 |
| `attn_resolutions` | Resolutions where attention is applied | (16,) |
| `resamp_with_conv` | Whether to use convolution for resampling | True |
| `conditional` | Whether the model is conditional | True |
| `sigma_min` | Minimum sigma for VESDE | 0.01 |
| `sigma_max` | Maximum sigma for VESDE | 50 |
| `beta_min` | Minimum beta for VPSDE/subVPSDE | 0.1 |
| `beta_max` | Maximum beta for VPSDE/subVPSDE | 20.0 |
| `num_scales` | Number of noise levels | 1000 |

### 4.3 Optimization Parameters (`config.optim`)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `weight_decay` | Weight decay for optimizer | 0 |
| `optimizer` | Optimizer type (only Adam supported) | Adam |
| `lr` | Learning rate | 2e-4 |
| `beta1` | Adam beta1 parameter | 0.9 |
| `eps` | Adam epsilon parameter | 1e-8 |
| `warmup` | Number of warmup steps | 5000 |
| `grad_clip` | Gradient clipping threshold | 1.0 |

### 4.4 Sampling Parameters (`config.sampling`)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `method` | Sampling method (pc for Predictor-Corrector) | pc |
| `predictor` | Predictor type (reverse_diffusion/ancestral_sampling) | reverse_diffusion |
| `corrector` | Corrector type (langevin/none) | langevin |
| `n_steps_each` | Number of steps for each predictor/corrector | 1 |
| `noise_removal` | Whether to remove noise at the end | True |
| `probability_flow` | Whether to use probability flow ODE | False |
| `snr` | Signal-to-noise ratio for corrector | 0.16 |

### 4.5 Data Parameters (`config.data`)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `dataset` | Dataset name (CIFAR10/CELEBA/CELEBAHQ/LSUN) | CIFAR10 |
| `image_size` | Image resolution | 32 |
| `random_flip` | Whether to use random horizontal flips | True |
| `centered` | Whether to center data | False |
| `uniform_dequantization` | Whether to use uniform dequantization | False |
| `num_channels` | Number of image channels | 3 |

## 5. Key Components

### 5.1 SDE Implementations
- **VESDE** (Variance Exploding SDE): Constant drift, increasing diffusion
- **VPSDE** (Variance Preserving SDE): Linear drift, constant diffusion coefficient
- **subVPSDE**: Modified VPSDE that excels at likelihood computation

### 5.2 Model Architectures
- **NCSN**: Neural Controlled Differential Equations
- **NCSNv2**: Improved NCSN with better architecture
- **NCSN++**: State-of-the-art architecture with improved performance
- **DDPM**: Denoising Diffusion Probabilistic Models
- **DDPM++**: Improved DDPM architecture

### 5.3 Sampling Methods
- **Predictor-Corrector Sampling**: Combines a predictor step with a corrector step
  - **Predictors**: Reverse diffusion, ancestral sampling, Euler-Maruyama
  - **Correctors**: Langevin dynamics, none

## 6. Evaluation

The `run_lib.evaluate()` function handles model evaluation:

1. **Loss Evaluation**: Computes loss on the full evaluation dataset
2. **Likelihood Evaluation**: Computes bits/dim on the specified dataset
3. **Sampling**: Generates samples and computes evaluation metrics
   - Inception Score (IS)
   - Fréchet Inception Distance (FID)
   - Kernel Inception Distance (KID)

## 7. Extending the Codebase

The codebase is designed to be modular and extensible:

- **Adding New SDEs**: Inherit from `sde_lib.SDE` and implement required methods
- **Adding New Models**: Create a new model class and register it with `@register_model`
- **Adding New Predictors/Correctors**: Implement the update function and register it

## 8. Usage Examples

### Training
```bash
python main.py --config configs/ve/cifar10_ncsnpp_continuous.py --mode train --workdir ./exp/cifar10_ncsnpp_continuous
```

### Evaluation
```bash
python main.py --config configs/ve/cifar10_ncsnpp_continuous.py --mode eval --workdir ./exp/cifar10_ncsnpp_continuous
```

## 9. Conclusion

This codebase provides a comprehensive implementation of score-based generative modeling with SDEs. It offers flexibility in choosing SDE types, model architectures, and training parameters, making it suitable for both research and practical applications. The modular design allows for easy extension and experimentation with new ideas in the field of generative modeling.