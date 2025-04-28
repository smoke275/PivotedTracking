"""
Model Predictive Path Integral (MPPI) controller for unicycle model.

This implementation follows the MPPI algorithm for nonlinear systems,
allowing an automated agent to optimally follow a target while respecting
dynamics constraints of the unicycle model.
GPU acceleration is used when available with optimized memory usage and batching.
"""
import numpy as np
from math import sin, cos, pi, sqrt, atan2, exp
import time
from collections import deque, namedtuple
from multitrack.utils.config import *

# Try importing torch for GPU acceleration
try:
    import torch
    TORCH_AVAILABLE = True
    
    # Check if MPS (Metal Performance Shaders) is available for Apple Silicon
    MPS_AVAILABLE = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    CUDA_AVAILABLE = torch.cuda.is_available()
    
    # Get base device info
    if MPS_AVAILABLE:
        BASE_DEVICE_INFO = "Apple Silicon (MPS)"
    elif CUDA_AVAILABLE:
        BASE_DEVICE_INFO = f"NVIDIA GPU: {torch.cuda.get_device_name(0)}"
    else:
        BASE_DEVICE_INFO = "CPU"
    
    # Set device info based on user preference
    if MPPI_USE_GPU:
        DEVICE_INFO = BASE_DEVICE_INFO
        if MPS_AVAILABLE:
            print("Apple Silicon MPS acceleration detected and available.")
        elif CUDA_AVAILABLE:
            print("NVIDIA CUDA acceleration detected and available.")
        else:
            print("PyTorch available, but GPU acceleration not detected. Using CPU.")
    else:
        DEVICE_INFO = "CPU (GPU disabled by user)"
        print("GPU acceleration available but disabled by user setting. Using CPU.")
        
except ImportError:
    TORCH_AVAILABLE = False
    MPS_AVAILABLE = False
    CUDA_AVAILABLE = False
    DEVICE_INFO = "CPU (PyTorch not available)"
    print("PyTorch not available. Using CPU-only implementation.")

# Create a cache entry type for storing previous computations
CacheEntry = namedtuple('CacheEntry', ['state', 'target', 'control', 'trajectory', 'timestamp'])

class MPPIController:
    def __init__(self, horizon=MPPI_HORIZON, samples=MPPI_SAMPLES, dt=0.1, 
                 control_limits=None, cost_weights=None, use_gpu=MPPI_USE_GPU):
        """
        Initialize MPPI controller for unicycle model
        
        Parameters:
        - horizon: Planning horizon (number of time steps)
        - samples: Number of trajectory samples for optimization
        - dt: Time step duration
        - control_limits: Dictionary with max/min limits for controls
        - cost_weights: Dictionary with weights for different cost components
        - use_gpu: Whether to use GPU acceleration if available
        """
        self.horizon = horizon  # Planning horizon
        self.samples = samples  # Number of Monte Carlo samples
        self.dt = dt  # Time step
        
        # Performance monitoring
        self.computation_times = deque(maxlen=10)
        self.last_compute_time = 0
        
        # Set GPU usage flag and handle potential errors
        self.use_gpu = use_gpu and TORCH_AVAILABLE
        # Properly select device based on availability
        if TORCH_AVAILABLE:
            try:
                if self.use_gpu and MPS_AVAILABLE:
                    self.device = torch.device("mps")
                elif self.use_gpu and CUDA_AVAILABLE:
                    self.device = torch.device("cuda")
                else:
                    self.device = torch.device("cpu")
                    
                # Test the device with a small computation to ensure it works
                test_tensor = torch.ones((1, 1), device=self.device)
                _ = test_tensor * 2  # Simple operation to check device works
                
            except Exception as e:
                print(f"Error initializing GPU: {e}")
                print("Falling back to CPU")
                self.device = torch.device("cpu")
                self.use_gpu = False
        else:
            self.device = None
            
        # GPU batch size for memory management
        self.batch_size = min(MPPI_GPU_BATCH_SIZE, samples) if self.use_gpu else samples
            
        # Cache for reusing similar computations (state hash -> control)
        self.computation_cache = deque(maxlen=MPPI_CACHE_SIZE)
        
        # Default control limits if not provided
        if control_limits is None:
            control_limits = {
                'v_min': FOLLOWER_LINEAR_VEL_MIN,
                'v_max': FOLLOWER_LINEAR_VEL_MAX,
                'omega_min': FOLLOWER_ANGULAR_VEL_MIN,
                'omega_max': FOLLOWER_ANGULAR_VEL_MAX
            }
        self.control_limits = control_limits
        
        # Default cost weights if not provided
        if cost_weights is None:
            cost_weights = {
                'target_position': MPPI_WEIGHT_POSITION,
                'target_heading': MPPI_WEIGHT_HEADING,
                'control_effort': MPPI_WEIGHT_CONTROL,
                'collision': MPPI_WEIGHT_COLLISION,
                'forward': MPPI_WEIGHT_FORWARD
            }
        self.cost_weights = cost_weights
        
        # Temperature parameter for softmax weighting
        self.lambda_value = MPPI_LAMBDA
        
        # Initialize nominal control sequence
        self.nominal_controls = np.zeros((horizon, 2))  # [v, omega] for each time step
        
        # Control noise standard deviations
        self.control_noise_sigma = np.array([FOLLOWER_LINEAR_NOISE_SIGMA, FOLLOWER_ANGULAR_NOISE_SIGMA])
        
        # Minimum safety distance
        self.safety_distance = FOLLOWER_SAFETY_DISTANCE
        
        # If using pytorch, prepare tensors
        if self.use_gpu:
            # Ensure all tensors are float32 for MPS compatibility
            self.nominal_controls_tensor = torch.zeros((horizon, 2), dtype=torch.float32, device=self.device)
            self.control_noise_sigma_tensor = torch.tensor(self.control_noise_sigma, dtype=torch.float32, device=self.device)
            self.control_limits_tensor = {
                'v_min': torch.tensor(self.control_limits['v_min'], dtype=torch.float32, device=self.device),
                'v_max': torch.tensor(self.control_limits['v_max'], dtype=torch.float32, device=self.device),
                'omega_min': torch.tensor(self.control_limits['omega_min'], dtype=torch.float32, device=self.device),
                'omega_max': torch.tensor(self.control_limits['omega_max'], dtype=torch.float32, device=self.device)
            }
            
            # Pre-compute trigonometric function lookup for common angles
            self._init_trig_lookup()
    
    def _init_trig_lookup(self):
        """Initialize trigonometric lookup tables for common angles to speed up computations"""
        if not self.use_gpu:
            return
            
        # Create lookup tables for sin and cos of common angles
        angles = torch.linspace(-pi, pi, 1000, dtype=torch.float32, device=self.device)
        self.sin_lookup = torch.sin(angles)
        self.cos_lookup = torch.cos(angles)
        self.angle_step = 2 * pi / 999  # Step size between lookup values
        
    def _fast_trig(self, angles):
        """Fast trigonometric function using lookup tables"""
        if not self.use_gpu:
            return np.cos(angles), np.sin(angles)
            
        # Convert angles to indices in the lookup table
        indices = ((angles + pi) / self.angle_step).long().clamp(0, 999)
        return self.cos_lookup[indices], self.sin_lookup[indices]
    
    def _is_similar_state(self, state1, state2, threshold=5.0):
        """Check if two states are similar enough to reuse computations"""
        if state1 is None or state2 is None:
            return False
            
        # Check position and heading similarity
        pos_diff = np.sqrt((state1[0] - state2[0])**2 + (state1[1] - state2[1])**2)
        heading_diff = abs((state1[2] - state2[2] + pi) % (2 * pi) - pi)
        vel_diff = abs(state1[3] - state2[3])
        
        return pos_diff < threshold and heading_diff < 0.3 and vel_diff < 5.0
    
    def _is_similar_target(self, target1, target2, threshold=10.0):
        """Check if two target trajectories are similar enough to reuse computations"""
        if target1 is None or target2 is None or len(target1) != len(target2):
            return False
            
        # Check similarity of first and last points
        start_diff = np.sqrt((target1[0][0] - target2[0][0])**2 + (target1[0][1] - target2[0][1])**2)
        end_diff = np.sqrt((target1[-1][0] - target2[-1][0])**2 + (target1[-1][1] - target2[-1][1])**2)
        
        return start_diff < threshold and end_diff < threshold
        
    def _check_cache(self, current_state, target_trajectory):
        """Check if we have a similar computation in cache that we can reuse"""
        current_time = time.time()
        
        for entry in self.computation_cache:
            if (self._is_similar_state(current_state, entry.state) and 
                self._is_similar_target(target_trajectory, entry.target) and
                current_time - entry.timestamp < 0.5):  # Only use fresh cache entries
                return entry.control, entry.trajectory
                
        return None, None

    def _add_to_cache(self, current_state, target_trajectory, control, trajectory):
        """Add a computation to cache for potential reuse"""
        entry = CacheEntry(
            state=current_state.copy(),
            target=target_trajectory.copy(),
            control=control.copy(),
            trajectory=trajectory.copy(),
            timestamp=time.time()
        )
        self.computation_cache.append(entry)
        
    def _compute_batch_rollouts(self, initial_state, perturbed_controls):
        """
        Compute multiple rollout trajectories for different control sequences
        
        Parameters:
        - initial_state: Initial state [x, y, theta, v]
        - perturbed_controls: Batch of control sequences, shape (samples, horizon, 2)
        
        Returns:
        - all_states: Batch of state trajectories, shape (samples, horizon+1, 4)
        """
        if self.use_gpu:
            # Use GPU implementation
            return self._compute_batch_rollouts_gpu(initial_state, perturbed_controls)
        
        # Use CPU implementation - optimized batch calculation
        all_states = np.zeros((self.samples, self.horizon + 1, 4))
        all_states[:, 0] = initial_state
        
        # Pre-compute angles for each trajectory sample
        thetas = np.zeros((self.samples, self.horizon))
        thetas[:, 0] = initial_state[2]
        
        # Vectorized implementation for CPU
        for t in range(self.horizon):
            # Get controls for this time step for all samples
            v_ctrls = perturbed_controls[:, t, 0]
            omegas = perturbed_controls[:, t, 1]
            
            # Get current positions
            xs = all_states[:, t, 0]
            ys = all_states[:, t, 1]
            
            # Get theta (either from thetas array or initial state)
            curr_thetas = thetas[:, t] if t > 0 else np.full(self.samples, initial_state[2])
            
            # Vectorized dynamics calculation
            cos_thetas = np.cos(curr_thetas)
            sin_thetas = np.sin(curr_thetas)
            
            # Update positions and headings
            all_states[:, t+1, 0] = xs + v_ctrls * cos_thetas * self.dt
            all_states[:, t+1, 1] = ys + v_ctrls * sin_thetas * self.dt
            
            # Update theta for next time step
            new_thetas = curr_thetas + omegas * self.dt
            # Normalize angles
            new_thetas = (new_thetas + pi) % (2 * pi) - pi
            
            all_states[:, t+1, 2] = new_thetas
            all_states[:, t+1, 3] = v_ctrls
            
            # Store theta for next iteration
            if t < self.horizon - 1:
                thetas[:, t+1] = new_thetas
        
        return all_states
    
    def _compute_costs_gpu(self, all_states, target_trajectory, obstacles=None):
        """
        Compute costs for a batch of trajectories using GPU with batching
        
        Parameters:
        - all_states: Batch of state trajectories as tensor
        - target_trajectory: Target states to follow as tensor
        - obstacles: List of obstacle positions [(x, y, radius), ...]
        
        Returns:
        - total_costs: Total cost for each trajectory sample
        """
        # Convert target_trajectory to tensor if it's not already
        if not isinstance(target_trajectory, torch.Tensor):
            target_trajectory = torch.tensor(target_trajectory, dtype=torch.float32, device=self.device)
        
        # Process in batches to avoid memory issues on GPU
        num_batches = (all_states.size(0) + self.batch_size - 1) // self.batch_size
        all_costs = []
        
        # Precompute weight tensors
        w_pos = torch.tensor(self.cost_weights['target_position'], dtype=torch.float32, device=self.device)
        w_heading = torch.tensor(self.cost_weights['target_heading'], dtype=torch.float32, device=self.device)
        w_control = torch.tensor(self.cost_weights['control_effort'], dtype=torch.float32, device=self.device)
        w_forward = torch.tensor(self.cost_weights['forward'], dtype=torch.float32, device=self.device)
        w_collision = torch.tensor(self.cost_weights['collision'], dtype=torch.float32, device=self.device)
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, all_states.size(0))
            batch_states = all_states[start_idx:end_idx]
            
            batch_size = batch_states.size(0)
            horizon = batch_states.size(1) - 1
            
            # Initialize costs tensor
            batch_costs = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
            
            # Compute costs for all time steps at once
            for t in range(1, horizon + 1):  # Skip initial state
                # Current states for all samples
                x = batch_states[:, t, 0]
                y = batch_states[:, t, 1]
                theta = batch_states[:, t, 2]
                v = batch_states[:, t, 3]
                
                # Previous velocity for control cost
                prev_v = batch_states[:, t-1, 3] if t > 1 else torch.zeros_like(v)
                
                # Target state (use the closest available if horizon exceeds target trajectory length)
                target_idx = min(t, target_trajectory.size(0) - 1)
                tx = target_trajectory[target_idx, 0]
                ty = target_trajectory[target_idx, 1]
                ttheta = target_trajectory[target_idx, 2]
                
                # Vectorized cost calculation
                # Position tracking cost (using squared Euclidean distance)
                dx = x - tx
                dy = y - ty
                pos_cost = (dx*dx + dy*dy) * w_pos
                
                # Heading alignment cost (using squared angular difference)
                heading_diff = (theta - ttheta + pi) % (2 * pi) - pi  # Normalize to [-pi, pi]
                heading_cost = (heading_diff * heading_diff) * w_heading
                
                # Control effort cost
                ctrl_cost = ((v - prev_v) * (v - prev_v)) * w_control
                
                # Forward incentive (negative reward for forward motion)
                forward_cost = -torch.clamp(v, min=0) * w_forward
                forward_cost += torch.clamp(-v, min=0) * w_forward * 2
                
                # Sum costs for this time step
                step_cost = pos_cost + heading_cost + ctrl_cost + forward_cost
                
                # Add to total cost
                batch_costs += step_cost
                
            # Handle obstacles in a separate loop for clarity
            if obstacles is not None:
                collision_cost = torch.zeros_like(batch_costs)
                safety_dist_sq = self.safety_distance * self.safety_distance
                
                for t in range(1, horizon + 1):  # Skip initial state
                    x = batch_states[:, t, 0]
                    y = batch_states[:, t, 1]
                    
                    for ox, oy, radius in obstacles:
                        # Convert obstacle position to tensor
                        ox_t = torch.tensor(ox, dtype=torch.float32, device=self.device)
                        oy_t = torch.tensor(oy, dtype=torch.float32, device=self.device)
                        
                        # Vectorized distance calculation
                        dist_sq = (x - ox_t)**2 + (y - oy_t)**2
                        
                        # Add cost only for points within safety distance
                        mask = dist_sq < safety_dist_sq
                        if mask.any():
                            cost_value = (self.safety_distance - torch.sqrt(dist_sq[mask])) ** 2
                            collision_cost[mask] += cost_value * w_collision
                
                batch_costs += collision_cost
            
            # Add batch costs to the list
            all_costs.append(batch_costs)
            
            # Memory cleanup for MPS
            if MPS_AVAILABLE and self.use_gpu:
                torch.mps.empty_cache()
        
        # Concatenate all batches
        if num_batches == 1:
            return all_costs[0]
        else:
            return torch.cat(all_costs)
    
    def _compute_costs(self, all_states, target_trajectory, obstacles=None):
        """
        Compute costs for a batch of trajectories
        
        Parameters:
        - all_states: Batch of state trajectories
        - target_trajectory: Target states to follow
        - obstacles: List of obstacle positions [(x, y, radius), ...]
        
        Returns:
        - total_costs: Total cost for each trajectory sample
        """
        if self.use_gpu:
            # Use GPU implementation
            return self._compute_costs_gpu(all_states, target_trajectory, obstacles)
        
        # CPU implementation - vectorized version for performance
        total_costs = np.zeros(self.samples)
        weights = self.cost_weights
        
        # Compute costs for the entire batch at once
        for t in range(1, self.horizon + 1):  # Skip initial state
            # Get current states for all samples
            x = all_states[:, t, 0]
            y = all_states[:, t, 1]
            theta = all_states[:, t, 2]
            v = all_states[:, t, 3]
            
            # Previous velocity for control cost
            prev_v = all_states[:, t-1, 3] if t > 1 else np.zeros_like(v)
            
            # Target state at this time step (or closest available)
            target_idx = min(t, len(target_trajectory) - 1)
            tx = target_trajectory[target_idx, 0]
            ty = target_trajectory[target_idx, 1]
            ttheta = target_trajectory[target_idx, 2]
            
            # Vectorized cost calculations
            # Position cost
            pos_cost = ((x - tx) ** 2 + (y - ty) ** 2) * weights['target_position']
            
            # Heading cost
            heading_diff = (theta - ttheta + pi) % (2 * pi) - pi  # Normalize to [-pi, pi]
            heading_cost = (heading_diff ** 2) * weights['target_heading']
            
            # Control effort cost
            ctrl_cost = ((v - prev_v) ** 2) * weights['control_effort']
            
            # Forward incentive
            forward_cost = -np.maximum(0, v) * weights['forward']
            forward_cost += np.maximum(0, -v) * weights['forward'] * 2
            
            # Sum all step costs
            total_costs += pos_cost + heading_cost + ctrl_cost + forward_cost
            
        # Handle obstacles
        if obstacles is not None:
            for i in range(self.samples):
                for t in range(1, self.horizon + 1):
                    x, y = all_states[i, t, 0], all_states[i, t, 1]
                    
                    for ox, oy, radius in obstacles:
                        dist = sqrt((x - ox)**2 + (y - oy)**2)
                        if dist < self.safety_distance:
                            total_costs[i] += ((self.safety_distance - dist)**2) * weights['collision']
        
        return total_costs
        
    def compute_control(self, current_state, target_trajectory, obstacles=None):
        """
        Compute optimal control input using MPPI
        
        Parameters:
        - current_state: Current state [x, y, theta, v]
        - target_trajectory: Sequence of target states to follow
        - obstacles: List of obstacle positions [(x, y, radius), ...]
        
        Returns:
        - optimal_control: Optimal control [v, omega] for the current time step
        - predicted_trajectory: Predicted states over the horizon
        """
        # Start timing
        start_time = time.time()
        
        # Check cache for similar computations to reuse
        cached_control, cached_trajectory = self._check_cache(current_state, target_trajectory)
        if cached_control is not None:
            # Cache hit! Return the cached results
            return cached_control, cached_trajectory
        
        # Use appropriate implementation based on availability
        try:
            if self.use_gpu:
                optimal_control, predicted_trajectory = self._compute_control_gpu(
                    current_state, target_trajectory, obstacles)
            else:
                optimal_control, predicted_trajectory = self._compute_control_cpu(
                    current_state, target_trajectory, obstacles)
                
            # Add to cache for potential future reuse
            self._add_to_cache(current_state, target_trajectory, optimal_control, predicted_trajectory)
            
        except Exception as e:
            # Fallback to CPU in case of GPU error
            print(f"Error in control computation: {e}")
            # If we were using GPU, fall back to CPU
            if self.use_gpu:
                print("Falling back to CPU computation")
                optimal_control, predicted_trajectory = self._compute_control_cpu(
                    current_state, target_trajectory, obstacles)
            else:
                # If already on CPU, re-raise the exception
                raise
                
        # Record computation time
        self.last_compute_time = time.time() - start_time
        self.computation_times.append(self.last_compute_time)
        
        return optimal_control, predicted_trajectory
    
    def _compute_control_gpu(self, current_state, target_trajectory, obstacles=None):
        """GPU implementation of MPPI control computation"""
        # Convert current_state and target_trajectory to tensors
        current_state_tensor = torch.tensor(current_state, dtype=torch.float32, device=self.device)
        target_trajectory_tensor = torch.tensor(target_trajectory, dtype=torch.float32, device=self.device)
        
        try:
            # Process in batches to avoid memory issues
            num_batches = (self.samples + self.batch_size - 1) // self.batch_size
            all_perturbed_controls = []
            
            # Generate perturbed controls in batches
            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = min((batch_idx + 1) * self.batch_size, self.samples)
                batch_size = end_idx - start_idx
                
                # Sample noise
                noise = torch.randn(batch_size, self.horizon, 2, device=self.device, dtype=torch.float32) * \
                        self.control_noise_sigma_tensor.unsqueeze(0).unsqueeze(0)
                
                # Add noise to nominal controls
                batch_perturbed_controls = self.nominal_controls_tensor.unsqueeze(0).expand(batch_size, -1, -1) + noise
                
                # Apply control limits
                batch_perturbed_controls[:, :, 0] = torch.clamp(
                    batch_perturbed_controls[:, :, 0],
                    self.control_limits_tensor['v_min'],
                    self.control_limits_tensor['v_max']
                )
                batch_perturbed_controls[:, :, 1] = torch.clamp(
                    batch_perturbed_controls[:, :, 1],
                    self.control_limits_tensor['omega_min'],
                    self.control_limits_tensor['omega_max']
                )
                
                all_perturbed_controls.append(batch_perturbed_controls)
                
                # Memory cleanup for MPS
                if MPS_AVAILABLE and self.use_gpu:
                    torch.mps.empty_cache()
            
            # Concatenate all batches
            if num_batches == 1:
                perturbed_controls = all_perturbed_controls[0]
            else:
                perturbed_controls = torch.cat(all_perturbed_controls, dim=0)
            
            # Compute rollouts for all perturbed controls
            all_states = self._compute_batch_rollouts_gpu(current_state, perturbed_controls)
            
            # Compute costs in batches
            costs = self._compute_costs_gpu(all_states, target_trajectory_tensor, obstacles)
            
            # Compute weights using softmax (optimized for numerical stability)
            min_cost = torch.min(costs)
            weights = torch.exp(-(costs - min_cost) / self.lambda_value)
            weights = weights / torch.sum(weights)  # Normalize
            
            # Compute weighted average of controls in batches
            weighted_controls = torch.zeros((self.horizon, 2), dtype=torch.float32, device=self.device)
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = min((batch_idx + 1) * self.batch_size, self.samples)
                
                batch_controls = perturbed_controls[start_idx:end_idx]
                batch_weights = weights[start_idx:end_idx].unsqueeze(1).unsqueeze(2)
                
                weighted_batch = batch_controls * batch_weights
                weighted_controls += torch.sum(weighted_batch, dim=0)
                
                # Memory cleanup for MPS
                if MPS_AVAILABLE and self.use_gpu:
                    torch.mps.empty_cache()
            
            # Update nominal controls (shift and append)
            # Ensure dimensions match for concatenation
            last_control = weighted_controls[-1:].clone()
            self.nominal_controls_tensor = torch.cat([weighted_controls[1:], last_control], dim=0)
            
            # Convert back to numpy for return
            optimal_control = weighted_controls[0].cpu().numpy()
            
            # Update numpy version of nominal controls
            self.nominal_controls = self.nominal_controls_tensor.cpu().numpy()
            
            # Compute predicted trajectory using updated nominal controls
            predicted_trajectory = self._compute_rollout(current_state, self.nominal_controls)
            
        except Exception as e:
            # Handle any GPU-specific errors
            print(f"GPU computation error: {e}")
            raise
            
        return optimal_control, predicted_trajectory
    
    def _compute_control_cpu(self, current_state, target_trajectory, obstacles=None):
        """CPU implementation of MPPI control computation - optimized for vectorization"""
        # Generate perturbed controls - using vectorized operations
        perturbed_controls = np.zeros((self.samples, self.horizon, 2))
        
        # Sample noise once for all samples
        noise = np.random.normal(0, 1, (self.samples, self.horizon, 2)) * \
                self.control_noise_sigma.reshape(1, 1, 2)
        
        # Apply noise to nominal controls with vectorized operations
        for i in range(self.samples):
            perturbed_controls[i] = self.nominal_controls + noise[i]
        
        # Apply control limits - vectorized version
        perturbed_controls[:, :, 0] = np.clip(
            perturbed_controls[:, :, 0], 
            self.control_limits['v_min'], 
            self.control_limits['v_max']
        )
        perturbed_controls[:, :, 1] = np.clip(
            perturbed_controls[:, :, 1], 
            self.control_limits['omega_min'], 
            self.control_limits['omega_max']
        )
        
        # Compute all trajectories at once using vectorized implementation
        all_states = self._compute_batch_rollouts(current_state, perturbed_controls)
        
        # Compute costs
        costs = self._compute_costs(all_states, target_trajectory, obstacles)
        
        # Compute weights using softmax - more numerically stable version
        min_cost = np.min(costs)
        weights = np.exp(-(costs - min_cost) / self.lambda_value)
        weights = weights / np.sum(weights)  # Normalize
        
        # Compute weighted average of controls
        weighted_controls = np.zeros((self.horizon, 2))
        for i in range(self.samples):
            weighted_controls += weights[i] * perturbed_controls[i]
        
        # Update nominal controls (shift and append)
        self.nominal_controls = np.vstack([weighted_controls[1:], weighted_controls[-1:]])
        
        # Return first control and predicted trajectory
        optimal_control = weighted_controls[0]
        
        # Compute the predicted trajectory using the updated nominal controls
        predicted_trajectory = self._compute_rollout(current_state, self.nominal_controls)
        
        return optimal_control, predicted_trajectory
    
    def reset(self):
        """Reset the nominal control sequence"""
        self.nominal_controls = np.zeros((self.horizon, 2))
        if self.use_gpu:
            self.nominal_controls_tensor = torch.zeros((self.horizon, 2), dtype=torch.float32, device=self.device)
        # Clear computation cache
        self.computation_cache.clear()
        
    def get_computation_stats(self):
        """Return performance statistics for monitoring"""
        if not self.computation_times:
            return {"avg_time": 0, "last_time": 0, "device": DEVICE_INFO}
            
        avg_time = sum(self.computation_times) / len(self.computation_times)
        return {
            "avg_time": avg_time * 1000,  # Convert to ms
            "last_time": self.last_compute_time * 1000,  # Convert to ms
            "device": DEVICE_INFO
        }
    
    def _compute_rollout(self, initial_state, controls):
        """
        Compute a single rollout trajectory given initial state and control sequence
        
        Parameters:
        - initial_state: Initial state [x, y, theta, v]
        - controls: Sequence of controls [v, omega] for the horizon
        
        Returns:
        - states: Sequence of states over the horizon
        """
        states = np.zeros((self.horizon + 1, 4))  # Extra state for initial condition
        states[0] = initial_state.copy()
        
        # Pre-compute trig values for the entire horizon to avoid redundant calculations
        thetas = np.zeros(self.horizon)
        for t in range(self.horizon):
            if t == 0:
                thetas[t] = initial_state[2]
            else:
                thetas[t] = states[t][2]
                
            # Extract control at time step t
            v_ctrl, omega = controls[t]
            
            # Unicycle dynamics
            if t > 0:
                theta = thetas[t-1] + omega * self.dt
                # Normalize angle to [-pi, pi]
                theta = (theta + pi) % (2 * pi) - pi
                thetas[t] = theta
                
        # Compute all positions using the pre-computed angles
        for t in range(self.horizon):
            # Extract current state
            x, y = states[t][0], states[t][1]
            theta = thetas[t]
            
            # Extract control at time step t
            v_ctrl, omega = controls[t]
            
            # Unicycle dynamics
            new_x = x + v_ctrl * cos(theta) * self.dt
            new_y = y + v_ctrl * sin(theta) * self.dt
            new_theta = theta + omega * self.dt
            # Normalize angle to [-pi, pi]
            new_theta = (new_theta + pi) % (2 * pi) - pi
            
            # Update state
            states[t+1] = [new_x, new_y, new_theta, v_ctrl]
        
        return states
        
    def _compute_batch_rollouts_gpu(self, initial_state, perturbed_controls):
        """
        Compute multiple rollout trajectories for different control sequences using GPU
        with batching to avoid memory issues
        
        Parameters:
        - initial_state: Initial state [x, y, theta, v]
        - perturbed_controls: Batch of control sequences as a tensor
        
        Returns:
        - all_states: Batch of state trajectories
        """
        # Process in batches to avoid memory issues on GPU
        num_batches = (self.samples + self.batch_size - 1) // self.batch_size
        all_states_list = []
        
        # Convert initial state to tensor once
        initial_state_tensor = torch.tensor(initial_state, dtype=torch.float32, device=self.device)
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, self.samples)
            batch_size = end_idx - start_idx
            
            # Extract this batch of controls
            batch_controls = perturbed_controls[start_idx:end_idx]
            
            # Initialize states tensor for this batch [batch_size, horizon+1, 4]
            batch_states = torch.zeros((batch_size, self.horizon + 1, 4), dtype=torch.float32, device=self.device)
            batch_states[:, 0] = initial_state_tensor.expand(batch_size, -1)
            
            # Pre-compute angles for the entire horizon
            thetas = torch.zeros((batch_size, self.horizon), dtype=torch.float32, device=self.device)
            thetas[:, 0] = initial_state_tensor[2].expand(batch_size)
            
            # Compute all states at once with vectorized operations
            for t in range(self.horizon):
                # Extract current states for all samples
                x = batch_states[:, t, 0]
                y = batch_states[:, t, 1]
                theta = batch_states[:, t, 2] if t == 0 else thetas[:, t-1]
                
                # Extract controls for this time step for all samples
                v_ctrl = batch_controls[:, t, 0]
                omega = batch_controls[:, t, 1]
                
                # Compute dynamics for all samples at once
                cos_theta, sin_theta = self._fast_trig(theta) if self.use_gpu else (torch.cos(theta), torch.sin(theta))
                
                new_x = x + v_ctrl * cos_theta * self.dt
                new_y = y + v_ctrl * sin_theta * self.dt
                new_theta = theta + omega * self.dt
                
                # Normalize angles to [-pi, pi]
                new_theta = (new_theta + pi) % (2 * pi) - pi
                
                # Store theta for next iteration to avoid redundant calculation
                if t < self.horizon - 1:
                    thetas[:, t] = new_theta
                
                # Update states
                batch_states[:, t+1, 0] = new_x
                batch_states[:, t+1, 1] = new_y
                batch_states[:, t+1, 2] = new_theta
                batch_states[:, t+1, 3] = v_ctrl
            
            # Add this batch to the list
            all_states_list.append(batch_states)
            
            # Explicit GPU memory cleanup for MPS
            if MPS_AVAILABLE and self.use_gpu:
                torch.mps.empty_cache()
        
        # Concatenate all batches
        if num_batches == 1:
            return all_states_list[0]
        else:
            # Need to concatenate along the batch dimension
            return torch.cat(all_states_list, dim=0)