# Data-Driven VP SDE Implementation Plan

## 1. Overview

This plan outlines the minimal changes needed to add a new Variance Preserving SDE (VPSDE) with a data-driven noise schedule to the Score SDE PyTorch codebase. The new SDE will use a continuous schedule function computed from dataset samples.

## 2. Implementation Steps

### 2.1 Implement the Schedule Function

**File**: `sde_lib.py`

Add the `get_geo_iso_velocity_schedule_fn` function to compute the data-driven schedule:

```python
def get_geo_iso_velocity_schedule_fn(data, grid):
    """Compute a data-driven continuous schedule function.
    
    Args:
        data: A batch of normalized data samples
        grid: A grid of time points for computing the schedule
        
    Returns:
        A continuous function that takes time t and returns the beta value
    """
    # Implementation of the geo-iso velocity schedule computation
    # This function should analyze the data and return a continuous schedule
    # For example:
    # 1. Compute statistics from the data
    # 2. Fit a curve to the statistics
    # 3. Return a function that evaluates this curve at any time t
    
    # Placeholder implementation - replace with actual schedule computation
    def schedule_fn(t):
        # Example: linear schedule (replace with actual data-driven schedule)
        return 0.1 + t * (20.0 - 0.1)
    
    return schedule_fn
```

### 2.2 Create DataDrivenVPSDE Class

**File**: `sde_lib.py`

Add a new `DataDrivenVPSDE` class that inherits from `VPSDE`:

```python
class DataDrivenVPSDE(VPSDE):
    def __init__(self, schedule_fn, N=1000):
        """Construct a Data-Driven Variance Preserving SDE.
        
        Args:
            schedule_fn: A continuous function that takes time t and returns beta(t)
            N: number of discretization steps
        """
        super().__init__(beta_min=0.1, beta_max=20.0, N=N)  # Dummy values, will be overridden
        self.schedule_fn = schedule_fn
        
        # Precompute discrete betas using the schedule function
        self.discrete_betas = torch.tensor([schedule_fn(t) / N for t in torch.linspace(0, 1, N)])
        self.alphas = 1. - self.discrete_betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1m_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
    
    def sde(self, x, t):
        # Use the schedule function to compute beta_t
        beta_t = self.schedule_fn(t)
        drift = -0.5 * beta_t[:, None, None, None] * x
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion
    
    def marginal_prob(self, x, t):
        # Compute log_mean_coeff using the schedule function
        # This would need to be adjusted based on the actual schedule
        # For a general schedule, we might need to compute an integral
        
        # Example implementation for a linear schedule (replace with actual integration)
        log_mean_coeff = -0.25 * t ** 2 * (20.0 - 0.1) - 0.5 * t * 0.1
        mean = torch.exp(log_mean_coeff[:, None, None, None]) * x
        std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
        return mean, std
```

### 2.3 Modify SDE Initialization in run_lib.py

**File**: `run_lib.py`

Update the SDE initialization code to handle the new data-driven SDE:

```python
# Setup SDEs
if config.training.sde.lower() == 'vpsde':
    sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
elif config.training.sde.lower() == 'subvpsde':
    sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
elif config.training.sde.lower() == 'vesde':
    sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    sampling_eps = 1e-5
elif config.training.sde.lower() == 'datadrivenvpsde':
    # Compute data-driven schedule
    # Get a batch of data samples to compute the schedule
    batch = next(train_iter)['image']._numpy()
    batch = torch.from_numpy(batch).to(config.device).float()
    batch = batch.permute(0, 3, 1, 2)
    batch = scaler(batch)
    
    # Define grid for schedule computation
    grid = torch.linspace(0, 1, 1000, device=config.device)
    
    # Compute the schedule function
    schedule_fn = sde_lib.get_geo_iso_velocity_schedule_fn(batch, grid)
    
    # Initialize the data-driven VPSDE
    sde = sde_lib.DataDrivenVPSDE(schedule_fn=schedule_fn, N=config.model.num_scales)
    sampling_eps = 1e-3
else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")
```

### 2.4 Update Configuration Handling

**File**: No new files needed

Users can now specify the new SDE type in their config files:

```python
# In config file
training.sde = 'datadrivenvpsde'
```

## 3. Key Changes Explained

### 3.1 Schedule Function

The `get_geo_iso_velocity_schedule_fn` function computes a continuous schedule from data samples. This function should be implemented according to the specific data-driven schedule algorithm required.

### 3.2 DataDrivenVPSDE Class

This new class inherits from `VPSDE` and overrides key methods to use the data-driven schedule:
- `__init__`: Precomputes discrete betas using the schedule function
- `sde`: Uses the schedule function to compute beta(t) at any time t
- `marginal_prob`: Computes the marginal distribution parameters using the schedule

### 3.3 SDE Initialization

The main change is in `run_lib.py` where we:
1. Add a case for the new SDE type
2. Load a batch of data before SDE initialization
3. Compute the schedule function from the data
4. Pass the schedule function to the SDE constructor

## 4. Minimal Invasiveness

This implementation follows the principle of minimal changes:
- No modifications to existing SDE classes
- No changes to the training loop
- No changes to loss functions
- No changes to sampling algorithms
- Only adds new code, doesn't modify existing functionality

## 5. Usage

To use the new data-driven VPSDE, users simply need to:
1. Set `training.sde = 'datadrivenvpsde'` in their config file
2. Run training as usual

The SDE will automatically compute the data-driven schedule from the training data during initialization.

## 6. Testing

To test the implementation:
1. Create a config file with `training.sde = 'datadrivenvpsde'`
2. Run training with this config
3. Verify that the SDE is initialized correctly and training proceeds without errors
4. Compare sample quality and training dynamics with the standard VPSDE

## 7. Conclusion

This plan outlines a minimal, non-intrusive way to add a data-driven VP SDE to the Score SDE PyTorch codebase. The implementation leverages existing code structure while adding the new functionality, making it easy to integrate and maintain.