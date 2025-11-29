# ISO_VPSDE Simplified Data Sampling Plan

## 1. Issue Analysis

The user has clarified that since the data is already shuffled during loading (as seen in `datasets.py` line 210), we don't need to implement additional shuffling. This simplifies our approach significantly.

## 2. Implementation Steps

### 2.1 Update run_lib.py for Simple Data Sampling

**File**: `run_lib.py`

Update the ISO_VPSDE initialization code to use the already shuffled data:

```python
elif config.training.sde.lower() == "isovpsde":
    # isoVPSDE requires data to generate the schedule_fn
    num_samples = config.model.num_gen_schedule_samples
    
    # Get a batch of data from the already shuffled train dataset
    # The dataset is already shuffled in datasets.py line 210
    batch = next(iter(train_ds))['image']
    batch_np = batch._numpy()
    
    # If we need more samples than available in one batch, get additional batches
    if batch_np.shape[0] < num_samples:
        next_batch = next(iter(train_ds))['image']._numpy()
        batch_np = np.concatenate([batch_np, next_batch], axis=0)
    
    # Take only the required number of samples
    batch_np = batch_np[:num_samples]
    
    # Convert to torch tensor and preprocess
    batch_tensor = torch.from_numpy(batch_np).to(config.device).float()
    batch_tensor = batch_tensor.permute(0, 3, 1, 2)  # NHWC to NCHW
    batch_tensor = scaler(batch_tensor)
    
    # Flatten the batch for the schedule function
    # The schedule function expects a 2D tensor [B, D]
    B, C, H, W = batch_tensor.shape
    batch_flat = batch_tensor.reshape(B, -1)
    
    # Initialize the isoVPSDE with the sampled data
    sde = sde_lib.isoVPSDE(
        N=config.model.num_scales,
        data_batch=batch_flat,
        pair_distance_method=config.model.pair_distance_method,
        pair_distance_k=config.model.pair_distance_k,
    )
    sampling_eps = 1e-3
```

### 2.2 Update isoVPSDE Constructor (If Needed)

**File**: `sde_lib.py`

Ensure the `isoVPSDE` constructor correctly calls its parent class:

```python
def __init__(
    self,
    data_batch: torch.Tensor,
    grid_size=2000,
    pair_distance_method="k_pairs",
    pair_distance_k=20,
    N=1000,
    beta_min=0.1,
    beta_max=20,
):
    """Construct a Geometric Iso-Velocity Variance Preserving SDE.

    Args:
    data_batch: Input data batch as torch.Tensor for calibration
    grid_size: Grid size for calibration phase
    pair_distance_method: Method for pairwise distance calculation ("full" or "k_pairs")
    pair_distance_k: Number of pairs per point when using "k_pairs" methods
      N: number of discretization steps
    """
    # Fix parent class call - it should inherit from SDE, not VPSDE
    super().__init__(N=N)
    self.N = N
    self.beta_min = beta_min
    self.beta_max = beta_max

    self.schedule_fn = self._get_geometric_iso_velocity_schedule_fn(
        data_batch, grid_size, pair_distance_method, pair_distance_k
    )
    
    # Rest of the constructor remains the same
```

### 2.3 Config File Parameters

**File**: `configs/isovp/cifar10_ddpmpp.py`

Ensure the config file includes the necessary parameters for ISO_VPSDE:

```python
def get_config():
    config = get_default_configs()
    
    # training
    training = config.training
    training.sde = 'isovpsde'  # Set SDE type to isovpsde
    
    # model
    model = config.model
    model.name = 'ddpmpp'
    
    # ISO_VPSDE specific parameters
    model.num_gen_schedule_samples = 1000  # Number of samples for schedule generation
    model.pair_distance_method = 'k_pairs'  # Method for pairwise distance calculation
    model.pair_distance_k = 20  # Number of pairs per point when using k_pairs
    
    # Rest of the config remains the same
    
    return config
```

## 3. Key Changes Explained

### 3.1 Simple Data Sampling

The main change is simplifying the data sampling approach:
1. We just get the first batch from the already shuffled dataset
2. If we need more samples, we get one more batch
3. We take only the required number of samples
4. No additional shuffling is needed since the dataset is already shuffled

### 3.2 Data Preprocessing

The data is properly preprocessed before being passed to the isoVPSDE:
1. Convert from numpy array to torch tensor
2. Permute from NHWC to NCHW format
3. Apply the data scaler
4. Flatten to a 2D tensor [B, D] as expected by the schedule function

### 3.3 Config File Parameters

The config file needs to include the following parameters for the isoVPSDE:
- `num_gen_schedule_samples`: Number of samples to use for schedule generation
- `pair_distance_method`: Method for calculating pairwise distances ("full" or "k_pairs")
- `pair_distance_k`: Number of pairs per point when using the "k_pairs" method

## 4. Minimal Invasiveness

This implementation follows the principle of minimal changes:
- Only modifies the ISO_VPSDE initialization code in `run_lib.py`
- Fixes the parent class call in the `isoVPSDE` constructor if needed
- No changes to the training loop or other parts of the codebase
- Only adds necessary parameters to the config file

## 5. Usage

To use the isoVPSDE, users need to:
1. Set `training.sde = 'isovpsde'` in their config file
2. Add the ISO_VPSDE specific parameters to their config file
3. Run training as usual

## 6. Testing

To test the implementation:
1. Create a config file with the ISO_VPSDE parameters
2. Run training with this config
3. Verify that the SDE is initialized correctly and training proceeds without errors
4. Check that the schedule function is properly generated from the sampled data

## 7. Conclusion

This plan simplifies the data sampling approach since the data is already shuffled during loading. It ensures that the isoVPSDE receives the necessary data for generating its schedule function while maintaining minimal changes to the codebase.