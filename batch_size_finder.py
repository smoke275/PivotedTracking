#!/usr/bin/env python
"""
GPU Batch Size Finder for MPPI Controller

This script runs a series of tests to determine the optimal batch size 
for GPU processing in the MPPI controller for your specific hardware.
"""
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

try:
    import torch
    TORCH_AVAILABLE = True
    
    # Check if MPS (Metal Performance Shaders) is available for Apple Silicon
    MPS_AVAILABLE = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    CUDA_AVAILABLE = torch.cuda.is_available()
    
    # Get base device info
    if MPS_AVAILABLE:
        DEVICE_INFO = "Apple Silicon (MPS)"
        device = torch.device("mps")
    elif CUDA_AVAILABLE:
        DEVICE_INFO = f"NVIDIA GPU: {torch.cuda.get_device_name(0)}"
        device = torch.device("cuda")
    else:
        DEVICE_INFO = "CPU"
        device = torch.device("cpu")
        
except ImportError:
    TORCH_AVAILABLE = False
    MPS_AVAILABLE = False
    CUDA_AVAILABLE = False
    DEVICE_INFO = "CPU (PyTorch not available)"
    device = None
    print("PyTorch not available. This script requires PyTorch.")
    sys.exit(1)

def test_batch_size(batch_size, total_samples=2000, tensor_size=(30, 2), num_trials=5):
    """Test performance with a specific batch size"""
    # Parameters similar to MPPI controller
    horizon, control_dims = tensor_size
    
    # Skip if batch size is larger than total samples
    if batch_size > total_samples:
        return float('inf'), 0
    
    # Calculate number of batches
    num_batches = (total_samples + batch_size - 1) // batch_size
    
    # Time the batched operation
    total_time = 0
    max_memory = 0
    
    for _ in range(num_trials):
        # Reset CUDA memory if available
        if CUDA_AVAILABLE:
            torch.cuda.empty_cache()
        elif MPS_AVAILABLE:
            torch.mps.empty_cache()
            
        # Track peak memory
        if CUDA_AVAILABLE:
            torch.cuda.reset_peak_memory_stats()
        
        # Similar operations to what happens in MPPI
        start_time = time.time()
        
        all_results = []
        
        # Create a template tensor - similar to nominal controls
        template = torch.zeros(tensor_size, dtype=torch.float32, device=device)
        
        # Process in batches like the MPPI controller
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, total_samples)
            curr_batch_size = end_idx - start_idx
            
            # Generate noise (similar to MPPI)
            noise = torch.randn(curr_batch_size, horizon, control_dims, 
                               device=device, dtype=torch.float32)
            
            # Expand template and add noise (similar to MPPI)
            batch_data = template.unsqueeze(0).expand(curr_batch_size, -1, -1) + noise
            
            # Perform some operations (similar to MPPI dynamics)
            batch_processed = torch.cumsum(batch_data, dim=1)
            batch_squared = batch_processed ** 2
            batch_sum = torch.sum(batch_squared, dim=(1, 2))
            
            # Store results
            all_results.append(batch_sum)
            
            # Memory cleanup for MPS
            if MPS_AVAILABLE:
                torch.mps.empty_cache()
        
        # One more operation to combine results
        if num_batches > 1:
            combined = torch.cat(all_results)
        else:
            combined = all_results[0]
        
        # Final reduction
        final_result = torch.mean(combined)
        
        # Force synchronization to get accurate timing
        if device.type != 'cpu':
            torch.cuda.synchronize() if CUDA_AVAILABLE else torch.mps.synchronize()
            
        # Get execution time
        execution_time = time.time() - start_time
        total_time += execution_time
        
        # Get peak memory usage
        if CUDA_AVAILABLE:
            mem_usage = torch.cuda.max_memory_allocated() / (1024 ** 2)  # Convert to MB
            max_memory = max(max_memory, mem_usage)
        
    # Average execution time
    avg_time = total_time / num_trials
    
    # For MPS (Apple Silicon), we don't have good memory stats, so estimate based on tensor size
    if MPS_AVAILABLE and not CUDA_AVAILABLE:
        # Rough estimate of memory usage based on tensor sizes
        # Each float32 is 4 bytes
        tensor_memory = 4 * batch_size * horizon * control_dims / (1024 ** 2)  # MB
        # Add overhead
        max_memory = tensor_memory * 3  # Rough estimate with overhead
    
    return avg_time, max_memory

def find_optimal_batch_size():
    """Test a range of batch sizes and find the optimal one"""
    print(f"Testing batch sizes for {DEVICE_INFO}")
    print("This may take a few minutes...")
    
    # Define batch sizes to test - expanded to very large batch sizes
    batch_sizes = [2000, 3072, 4096, 5120, 6144, 8192, 10240, 12288, 16384, 20480, 24576, 32768]
    
    # For MPPI-like parameters - increased sample size to test very large batches
    total_samples = 35000  # Significantly increased to allow testing larger batches
    tensor_size = (30, 2)  # (MPPI_HORIZON, control_dims)
    
    # Track results
    results = []
    
    for batch_size in batch_sizes:
        if batch_size > total_samples:
            continue
            
        print(f"Testing batch size: {batch_size}")
        try:
            avg_time, memory_usage = test_batch_size(batch_size, total_samples, tensor_size)
            results.append((batch_size, avg_time, memory_usage))
            print(f"  Time: {avg_time:.4f}s, Memory: {memory_usage:.2f} MB")
        except RuntimeError as e:
            # Catch out of memory errors
            print(f"  Error with batch size {batch_size}: {e}")
            print(f"  This may indicate we've reached the GPU memory limit")
            break
    
    if not results:
        print("No successful batch size tests. Please try with smaller batch sizes.")
        return None
    
    # Find the batch size with minimum execution time
    results.sort(key=lambda x: x[1])  # Sort by time
    optimal_batch_size, best_time, memory_used = results[0]
    
    print("\nResults:")
    print("-" * 60)
    print(f"{'Batch Size':<10} {'Time (s)':<10} {'Memory (MB)':<15}")
    print("-" * 60)
    for batch_size, avg_time, memory_usage in sorted(results, key=lambda x: x[0]):
        print(f"{batch_size:<10} {avg_time:<10.4f} {memory_usage:<15.2f}")
    
    print("\nOptimal batch size for your GPU:")
    print(f"Batch Size: {optimal_batch_size}")
    print(f"Execution Time: {best_time:.4f} seconds")
    print(f"Memory Usage: {memory_used:.2f} MB")
    
    # Create a plot
    batch_sizes = [r[0] for r in results]
    times = [r[1] for r in results]
    memories = [r[2] for r in results]
    
    plt.figure(figsize=(12, 6))
    
    # Time plot
    plt.subplot(1, 2, 1)
    plt.plot(batch_sizes, times, 'o-', color='blue')
    plt.axvline(x=optimal_batch_size, color='red', linestyle='--', 
                label=f'Optimal: {optimal_batch_size}')
    plt.xlabel('Batch Size')
    plt.ylabel('Time (seconds)')
    plt.title('Execution Time vs Batch Size')
    plt.grid(True)
    plt.legend()
    
    # Memory plot
    plt.subplot(1, 2, 2)
    plt.plot(batch_sizes, memories, 'o-', color='green')
    plt.axvline(x=optimal_batch_size, color='red', linestyle='--', 
                label=f'Optimal: {optimal_batch_size}')
    plt.xlabel('Batch Size')
    plt.ylabel('Memory (MB)')
    plt.title('Memory Usage vs Batch Size')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('batch_size_results.png')
    
    # Save numeric results to a CSV file
    with open('batch_size_results.csv', 'w') as f:
        f.write("batch_size,time,memory_mb\n")
        for bs, t, m in sorted(results, key=lambda x: x[0]):
            f.write(f"{bs},{t:.6f},{m:.2f}\n")
    
    return optimal_batch_size

if __name__ == "__main__":
    # Check if GPU is available
    if not (CUDA_AVAILABLE or MPS_AVAILABLE):
        print("No GPU detected. This script is meant for finding optimal GPU batch sizes.")
        sys.exit(1)
        
    # Find optimal batch size
    optimal_batch_size = find_optimal_batch_size()
    
    if optimal_batch_size:
        print("\nRecommendation:")
        print(f"Based on the test results, set MPPI_GPU_BATCH_SIZE = {optimal_batch_size} in your config file.")
        print(f"Location: {os.path.abspath('multitrack/utils/config.py')}")
        print("\nResults saved to batch_size_results.png and batch_size_results.csv")