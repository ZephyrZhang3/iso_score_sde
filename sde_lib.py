# coding=utf-8
# Copyright 2020 The Google Research Authors.
# Copyright 2025 The GIVDS Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Abstract SDE classes, Reverse SDE, and VE/VP SDEs."""

import abc

import numpy as np
import scipy
import torch

from tqdm import tqdm


class SDE(abc.ABC):
    """SDE abstract class. Functions are designed for a mini-batch of inputs."""

    def __init__(self, N):
        """Construct an SDE.

        Args:
          N: number of discretization time steps.
        """
        super().__init__()
        self.N = N

    @property
    @abc.abstractmethod
    def T(self):
        """End time of the SDE."""
        pass

    @abc.abstractmethod
    def sde(self, x, t):
        pass

    @abc.abstractmethod
    def marginal_prob(self, x, t):
        """Parameters to determine the marginal distribution of the SDE, $p_t(x)$."""
        pass

    @abc.abstractmethod
    def prior_sampling(self, shape):
        """Generate one sample from the prior distribution, $p_T(x)$."""
        pass

    @abc.abstractmethod
    def prior_logp(self, z):
        """Compute log-density of the prior distribution.

        Useful for computing the log-likelihood via probability flow ODE.

        Args:
          z: latent code
        Returns:
          log probability density
        """
        pass

    def discretize(self, x, t):
        """Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.

        Useful for reverse diffusion sampling and probabiliy flow sampling.
        Defaults to Euler-Maruyama discretization.

        Args:
          x: a torch tensor
          t: a torch float representing the time step (from 0 to `self.T`)

        Returns:
          f, G
        """
        dt = 1 / self.N
        drift, diffusion = self.sde(x, t)
        f = drift * dt
        G = diffusion * torch.sqrt(torch.tensor(dt, device=t.device))
        return f, G

    def reverse(self, score_fn, probability_flow=False):
        """Create the reverse-time SDE/ODE.

        Args:
          score_fn: A time-dependent score-based model that takes x and t and returns the score.
          probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
        """
        N = self.N
        T = self.T
        sde_fn = self.sde
        discretize_fn = self.discretize

        # Build the class for reverse-time SDE.
        class RSDE(self.__class__):
            def __init__(self):
                self.N = N
                self.probability_flow = probability_flow

            @property
            def T(self):
                return T

            def sde(self, x, t):
                """Create the drift and diffusion functions for the reverse SDE/ODE."""
                drift, diffusion = sde_fn(x, t)
                score = score_fn(x, t)
                drift = drift - diffusion[:, None, None, None] ** 2 * score * (
                    0.5 if self.probability_flow else 1.0
                )
                # Set the diffusion function to zero for ODEs.
                diffusion = 0.0 if self.probability_flow else diffusion
                return drift, diffusion

            def discretize(self, x, t):
                """Create discretized iteration rules for the reverse diffusion sampler."""
                f, G = discretize_fn(x, t)
                rev_f = f - G[:, None, None, None] ** 2 * score_fn(x, t) * (
                    0.5 if self.probability_flow else 1.0
                )
                rev_G = torch.zeros_like(G) if self.probability_flow else G
                return rev_f, rev_G

        return RSDE()


class VPSDE(SDE):
    def __init__(self, beta_min=0.1, beta_max=20, N=1000):
        """Construct a Variance Preserving SDE.

        Args:
          beta_min: value of beta(0)
          beta_max: value of beta(1)
          N: number of discretization steps
        """
        super().__init__(N)
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.N = N
        self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
        self.alphas = 1.0 - self.discrete_betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1m_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    @property
    def T(self):
        return 1

    def sde(self, x, t):
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = -0.5 * beta_t[:, None, None, None] * x
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion

    def marginal_prob(self, x, t):
        log_mean_coeff = (
            -0.25 * t**2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        )
        mean = torch.exp(log_mean_coeff[:, None, None, None]) * x
        std = torch.sqrt(1.0 - torch.exp(2.0 * log_mean_coeff))
        return mean, std

    def prior_sampling(self, shape):
        return torch.randn(*shape)

    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        logps = -N / 2.0 * np.log(2 * np.pi) - torch.sum(z**2, dim=(1, 2, 3)) / 2.0
        return logps

    def discretize(self, x, t):
        """DDPM discretization."""
        timestep = (t * (self.N - 1) / self.T).long()
        beta = self.discrete_betas.to(x.device)[timestep]
        alpha = self.alphas.to(x.device)[timestep]
        sqrt_beta = torch.sqrt(beta)
        f = torch.sqrt(alpha)[:, None, None, None] * x - x
        G = sqrt_beta
        return f, G


class subVPSDE(SDE):
    def __init__(self, beta_min=0.1, beta_max=20, N=1000):
        """Construct the sub-VP SDE that excels at likelihoods.

        Args:
          beta_min: value of beta(0)
          beta_max: value of beta(1)
          N: number of discretization steps
        """
        super().__init__(N)
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.N = N

    @property
    def T(self):
        return 1

    def sde(self, x, t):
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = -0.5 * beta_t[:, None, None, None] * x
        discount = 1.0 - torch.exp(
            -2 * self.beta_0 * t - (self.beta_1 - self.beta_0) * t**2
        )
        diffusion = torch.sqrt(beta_t * discount)
        return drift, diffusion

    def marginal_prob(self, x, t):
        log_mean_coeff = (
            -0.25 * t**2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        )
        mean = torch.exp(log_mean_coeff)[:, None, None, None] * x
        std = 1 - torch.exp(2.0 * log_mean_coeff)
        return mean, std

    def prior_sampling(self, shape):
        return torch.randn(*shape)

    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        return -N / 2.0 * np.log(2 * np.pi) - torch.sum(z**2, dim=(1, 2, 3)) / 2.0


class VESDE(SDE):
    def __init__(self, sigma_min=0.01, sigma_max=50, N=1000):
        """Construct a Variance Exploding SDE.

        Args:
          sigma_min: smallest sigma.
          sigma_max: largest sigma.
          N: number of discretization steps
        """
        super().__init__(N)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.discrete_sigmas = torch.exp(
            torch.linspace(np.log(self.sigma_min), np.log(self.sigma_max), N)
        )
        self.N = N

    @property
    def T(self):
        return 1

    def sde(self, x, t):
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        drift = torch.zeros_like(x)
        diffusion = sigma * torch.sqrt(
            torch.tensor(
                2 * (np.log(self.sigma_max) - np.log(self.sigma_min)), device=t.device
            )
        )
        return drift, diffusion

    def marginal_prob(self, x, t):
        std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        mean = x
        return mean, std

    def prior_sampling(self, shape):
        return torch.randn(*shape) * self.sigma_max

    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        return -N / 2.0 * np.log(2 * np.pi * self.sigma_max**2) - torch.sum(
            z**2, dim=(1, 2, 3)
        ) / (2 * self.sigma_max**2)

    def discretize(self, x, t):
        """SMLD(NCSN) discretization."""
        timestep = (t * (self.N - 1) / self.T).long()
        sigma = self.discrete_sigmas.to(t.device)[timestep]
        adjacent_sigma = torch.where(
            timestep == 0,
            torch.zeros_like(t),
            self.discrete_sigmas[timestep - 1].to(t.device),
        )
        f = torch.zeros_like(x)
        G = torch.sqrt(sigma**2 - adjacent_sigma**2)
        return f, G


class isoVPSDE(VPSDE):
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
        super().__init__(beta_min=beta_min, beta_max=beta_max, N=N)
        self.N = N

        self.schedule_fn = self._get_geometric_iso_velocity_schedule_fn(
            data_batch, grid_size, pair_distance_method, pair_distance_k
        )

        # 2. Discretize the Schedule for Training (t = 1/N ... 1.0)
        # Note: We use (i+1)/N to match standard DDPM indexing (1..N)
        t_discrete = torch.linspace(1.0 / N, 1.0, N, device=data_batch.device, dtype=torch.float64)
        
        # Get alpha_bar (signal squared) and sigma (noise)
        # schedule_fn returns (alpha, sigma) where alpha = sqrt(alpha_bar)
        alphas_bar, _ = self.schedule_fn(t_discrete)
        alphas_bar = alphas_bar.to(torch.float32)
        
        # 3. Update super class VPSDE attributes
        # In VPSDE convention: alphas_cumprod is alpha_bar
        self.alphas_cumprod = alphas_bar ** 2
        self.sqrt_alphas_cumprod = alphas_bar
        self.sqrt_1m_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # 4. Derive discrete betas
        # alpha_bar_t = alpha_bar_{t-1} * (1 - beta_t)
        # => beta_t = 1 - (alpha_bar_t / alpha_bar_{t-1})
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=data_batch.device), self.alphas_cumprod[:-1]])
        self.alphas = self.alphas_cumprod / alphas_cumprod_prev
        self.discrete_betas = 1.0 - self.alphas
        
        # Clip betas to prevent numerical instability
        self.discrete_betas = torch.clip(self.discrete_betas, 0.0001, 0.9999)

    @property
    def T(self):
        return 1

    def sde(self, x, t):
        # turn on gradient tracking for t to get d(log_alpha)/dt
        t_ = t.clone().detach().requires_grad_(True)
        mean_coeff, _ = self.schedule_fn(t_)
        
        # log_mean_coeff = log(alpha(t))
        log_mean_coeff = torch.log(mean_coeff)
        
        # beta(t) = -2 * d/dt [log_alpha(t)]
        # using autograd to compute d(log_alpha(t))/dt
        grads = torch.autograd.grad(log_mean_coeff.sum(), t_, create_graph=True)[0]
        beta_t = -2 * grads
        
        # beta_t should have the same shape as x
        beta_t = beta_t.view(-1, *([1] * (x.dim() - 1)))
        
        drift = -0.5 * beta_t * x
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion

    
    def marginal_prob(self, x, t):
        # mean_coeff = sqrt(alpha_bar), std = sqrt(1 - alpha_bar)
        mean_coeff, std = self.schedule_fn(t)
        # make sure mean_coeff and std are on the same device and dtype as x
        mean_coeff = mean_coeff.to(x)
        std = std.to(x)
        # Broadcast mean_coeff and std to match x's shape
        mean_coeff = mean_coeff.view(-1, *([1] * (x.dim() - 1)))
        std = std.view(-1, *([1] * (x.dim() - 1)))


        return x * mean_coeff, std

    # prior sampling, prior_logp and discretize are the same as VPSDE
    def prior_sampling(self, shape):
        return torch.randn(*shape)

    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        logps = -N / 2.0 * np.log(2 * np.pi) - torch.sum(z**2, dim=(1, 2, 3)) / 2.0
        return logps

    def discretize(self, x, t):
        """DDPM discretization."""
        timestep = (t * (self.N - 1) / self.T).long()
        beta = self.discrete_betas.to(x.device)[timestep]
        alpha = self.alphas.to(x.device)[timestep]
        sqrt_beta = torch.sqrt(beta)
        f = torch.sqrt(alpha)[:, None, None, None] * x - x
        G = sqrt_beta
        return f, G

    @staticmethod
    def _compute_pairwise_distances(
        x: torch.Tensor, method: str = "full", k: int = 5
    ) -> torch.Tensor:
        """
        Compute pairwise distances between points in the input tensor.
        Supports two methods:
        1. "full": O(N²) complexity, computes all unique pairwise distances (i < j)
        2. "k_pairs": O(KN) complexity, computes K random pairs per point using torch.roll

        Args:
            x: Input tensor of shape [N, D] where N is the number of points and D is the dimensionality
            p: Norm type for distance calculation (p=2.0 for Euclidean, p=1.0 for Manhattan, etc.)
            method: Distance calculation method, either "full" (default) or "k_pairs"
            k: Number of pairs per point when using "k_pairs" method

        Returns:
            torch.Tensor: 1D tensor containing pairwise distances in float64 precision.
        """
        # Validate input using assert statements
        assert isinstance(x, torch.Tensor), (
            f"Samples must be a torch.Tensor, got {type(x).__name__}"
        )
        assert x.dim() == 2, f"Samples must be 2D tensor, got shape {x.shape}"
        assert x.size(0) > 1, "Need at least 2 points to compute pairwise distances"
        assert method in ["full", "k_pairs"], (
            f"Method must be 'full' or 'k_pairs', got {method}"
        )
        assert k > 0, f"k must be positive, got {k}"

        # For large datasets, use float32 for better performance unless high precision is needed
        if x.size(0) > 10000 and x.size(1) > 100:  # Threshold for large dataset
            x_d = x.to(dtype=torch.float32)
        else:
            x_d = x.to(dtype=torch.float64)  # Maintain stability for smaller datasets

        if method == "full":
            # Use PyTorch's optimized cdist function to compute pairwise distances
            # cdist returns a N x N matrix where dists[i][j] is the distance between x[i] and x[j]
            dist_matrix = torch.cdist(x_d, x_d)

            # Extract the upper triangular part (i < j) to get unique pairwise distances
            # and flatten it to a 1D tensor
            # Exclude the diagonal (distance from a point to itself is 0)
            n = dist_matrix.size(0)
            # Create a mask for upper triangular part (excluding diagonal)
            mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
            all_dists = dist_matrix[mask]
        else:  # method == "k_pairs"
            # K-Random Pairs method with O(KN) complexity
            B = x_d.shape[0]
            dists_list = []

            # Create random pairs using random indices for better performance with clustered data
            for _ in range(k):
                # Generate random indices for pairs
                # Ensure we don't have duplicate pairs by using different random seeds
                idx1 = torch.arange(B, device=x_d.device)
                idx2 = torch.randperm(B, device=x_d.device)

                # Compute distances between random pairs
                x1 = x_d[idx1]
                x2 = x_d[idx2]

                # Optimized Euclidean distance calculation
                diff = x1 - x2
                dists = torch.sqrt(torch.sum(diff.pow(2), dim=1))

                dists_list.append(dists)

            # Concatenate all distances from different random pairs
            all_dists = torch.cat(dists_list)

        # Check for NaN or inf values
        assert not torch.isnan(all_dists).any() and not torch.isinf(all_dists).any(), (
            "Distance calculation resulted in NaN or inf values"
        )

        # Convert back to float64 for consistency in downstream calculations
        return all_dists.to(dtype=torch.float64)

    @staticmethod
    def _get_geometric_iso_velocity_schedule_fn(
        data_batch: torch.Tensor,
        grid_size: int = 2000,
        pairwise_distance_method: str = "k_pairs",
        k: int = 20,
    ):
        """
        Returns a continuous Geometric Iso-Velocity Schedule function.

        Args:
            data_batch: Input data batch as torch.Tensor for calibration
            grid_size: Grid size for calibration phase
            pair_distance_method: Method for pairwise distance calculation ("full" or "k_pairs")
            pair_distance_k: Number of pairs per point when using "k_pairs" methods
        Returns:
            schedule_fn: A callable that takes t (float or numpy array) in [0, 1]
                        and returns (alpha_t, sigma_t) as numpy arrays.
        """

        # 1. Pre-processing
        assert data_batch.dim() == 2, (
            f"Data must be 2D tensor, got shape {data_batch.shape}"
        )
        assert data_batch.numel() > 0, "Data tensor cannot be empty"

        # Flatten data to [B, D]
        B = data_batch.shape[0]
        X = data_batch.reshape(B, -1)

        # --- Phase 1: Calibration ---
        # Define grid of noise angles: 0 (Signal) -> pi/2 (Noise)
        theta_grid = torch.linspace(0, np.pi / 2, grid_size, dtype=torch.float64)
        psi_vals = []

        # Pre-sample noise for calibration stability
        # Using a fixed noise sample ensures the psi curve is smooth
        noise = torch.randn_like(data_batch)

        # -------------------------------------------------------
        # Phase 1: Geometric Calibration
        # -------------------------------------------------------
        for theta in tqdm(theta_grid, desc="Calibrating Geometric Schedule"):
            alpha, sigma = torch.cos(theta), torch.sin(theta)
            X_t = alpha * X + sigma * noise

            all_dists_sq = self._compute_pairwise_distances(
                X_t,
                method=pair_distance_method,
                k=pair_distance_k,
            )

            mean_z, std_z = all_dists_sq.mean(), all_dists_sq.std()

            # Handle potential division by zero
            # psi=0.0 when mean_z is very small (close to zero)
            # This handles the case where all points are very close to each other
            psi = 0.0 if mean_z < 1e-8 else (std_z / mean_z)
            psi_vals.append(psi)

        psi_vals = torch.tensor(psi_vals, dtype=torch.float64)

        # -------------------------------------------------------
        # Phase 2: Continuous Mapping Construction
        # -------------------------------------------------------
        # We need a map: psi -> theta.
        # PchipInterpolator requires x to be strictly increasing.
        psi_increasing = torch.flip(
            psi_vals, dims=[0]
        )  # Low psi (Noise) -> High psi (Signal)
        theta_decreasing = torch.flip(theta_grid, dims=[0])  # High theta -> Low theta

        # 1. Enforce Monotonicity (Filter measurement noise)
        psi_monotonic = torch.cummax(psi_increasing, dim=0).values

        # 2. Enforce STRICT Monotonicity
        # Add a tiny epsilon ramp to break any numerical ties (flat regions)
        epsilon = 1e-6
        ramp = torch.linspace(0, epsilon, len(psi_monotonic), dtype=torch.float64)
        psi_strict = psi_monotonic + ramp

        # 3. Create Interpolator
        # Maps Target Psi -> Required Theta
        psi_to_theta = scipy.interpolate.PchipInterpolator(
            x=psi_strict, y=theta_decreasing
        )

        # Record boundaries from the strict curve
        psi_start_val = psi_strict[-1]  # Corresponds to theta=0
        psi_end_val = psi_strict[0]  # Corresponds to theta=pi/2

        # -------------------------------------------------------
        # Phase 3: Define Closure Function
        # -------------------------------------------------------
        def schedule_fn(t):
            """
            Continuous Iso-Velocity Schedule.
            Args:
                t: Time in [0, 1]. Can be float or torch tensor.
                t=0 -> Data (Signal), t=1 -> Noise.
            Returns:
                alpha_t, sigma_t: Signal and Noise coefficients.
            """
            # 1. 统一转为 CPU Numpy 进行插值计算
            # 1. unify t to numpy arrays
            if isinstance(t, torch.Tensor):
                t_device = t.device
                t_np = t.detach().cpu().numpy().astype(np.float64)
            else:
                t_device = 'cpu'
                t_np = np.array(t, dtype=np.float64)
            
            # Clip t to valid range [0, 1]
            t_np = np.clip(t_np, 0.0, 1.0)

            # 1. Map time to target psi linearly (Iso-Velocity)
                # Map t -> target_psi
                # t=0 -> psi_start (High discernibility)
                # t=1 -> psi_end (Low discernibility)
            target_psi = psi_start_val + t_np * (psi_end_val - psi_start_val)

            
            # 2. Invert to find theta
                # Scipy Interpolation (CPU only)
            theta_np = psi_to_theta(target_psi)

            # 3. Enforce Zero-SNR Boundary Condition
                # Explicitly force theta=pi/2 at t=1 to prevent signal leakage
            theta_np = np.where(t_np >= 1.0 - 1e-5, np.pi / 2, theta_np)
            theta_np = np.where(t_np <= 1e-5, 0.0, theta_np)

            # 4. Compute Coefficients
            alpha_np = np.cos(theta_np)
            sigma_np = np.sin(theta_np)

            # 5. trans Numpy to Tensor and send to original device
            alpha_t = torch.tensor(alpha_np, device=t_device, dtype=torch.float32)
            sigma_t = torch.tensor(sigma_np, device=t_device, dtype=torch.float32)

            return alpha_t, sigma_t

        return schedule_fn
