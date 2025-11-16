"""Sharding utilities for multi-device JAX training and inference.

This module provides utilities for model parallelism and data parallelism,
inspired by the MaxText framework and Google's best practices for TPU training.

Key concepts:
- Model parallelism: Split model layers across devices
- Data parallelism: Replicate model, split batch across devices
- Pipeline parallelism: Split model stages across devices (future)

References:
- MaxText: https://github.com/google/maxtext
- JAX sharding guide: https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html
"""

from typing import Optional, Tuple, Any
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental import mesh_utils
import numpy as np


def create_device_mesh(
    num_devices: Optional[int] = None,
    mesh_shape: Optional[Tuple[int, ...]] = None,
    axis_names: Optional[Tuple[str, ...]] = None
) -> Mesh:
    """Create a device mesh for distributed computation.
    
    Args:
        num_devices: Number of devices to use (default: all available)
        mesh_shape: Shape of the device mesh (e.g., (4, 2) for 8 devices)
                   If None, creates a 1D mesh of all devices
        axis_names: Names for mesh axes (e.g., ('data', 'model'))
                   If None, uses ('devices',) for 1D or ('data', 'model') for 2D
                   
    Returns:
        JAX Mesh object for distributed computation
        
    Example:
        >>> # Create 1D mesh for data parallelism
        >>> mesh = create_device_mesh(num_devices=8, mesh_shape=(8,), axis_names=('data',))
        >>>
        >>> # Create 2D mesh for data + model parallelism
        >>> mesh = create_device_mesh(num_devices=8, mesh_shape=(4, 2), axis_names=('data', 'model'))
    """
    # Get available devices
    devices = jax.devices()
    
    if num_devices is None:
        num_devices = len(devices)
    else:
        assert num_devices <= len(devices), \
            f"Requested {num_devices} devices but only {len(devices)} available"
        devices = devices[:num_devices]
    
    # Default mesh shape: 1D array of all devices
    if mesh_shape is None:
        mesh_shape = (num_devices,)
    
    # Validate mesh shape
    mesh_size = np.prod(mesh_shape)
    assert mesh_size == num_devices, \
        f"Mesh shape {mesh_shape} (size {mesh_size}) doesn't match num_devices ({num_devices})"
    
    # Default axis names
    if axis_names is None:
        if len(mesh_shape) == 1:
            axis_names = ('devices',)
        elif len(mesh_shape) == 2:
            axis_names = ('data', 'model')
        else:
            axis_names = tuple(f'axis_{i}' for i in range(len(mesh_shape)))
    
    assert len(axis_names) == len(mesh_shape), \
        f"Number of axis names ({len(axis_names)}) must match mesh dimensions ({len(mesh_shape)})"
    
    # Create device array
    device_array = mesh_utils.create_device_mesh(mesh_shape, devices)
    
    # Create mesh
    mesh = Mesh(device_array, axis_names)
    
    return mesh


def get_data_parallel_sharding():
    """Get sharding spec for data parallelism.
    
    Data parallelism: Replicate model on all devices, split batch dimension.
    
    Returns:
        Tuple of (batch_spec, param_spec) partition specs
        
    Example:
        >>> batch_spec, param_spec = get_data_parallel_sharding()
        >>> # Shard batch across 'data' axis, replicate params
        >>> sharded_batch = jax.device_put(batch, P(batch_spec))
    """
    # Shard batch dimension across 'data' axis
    batch_spec = P('data')
    
    # Replicate all parameters (no sharding)
    param_spec = P()
    
    return batch_spec, param_spec


def get_model_parallel_sharding(
    num_layers: int,
    devices_per_layer: int = 1
):
    """Get sharding spec for model parallelism.
    
    Model parallelism: Split model layers across devices.
    
    Args:
        num_layers: Total number of transformer layers
        devices_per_layer: Number of devices per layer (for large layers)
        
    Returns:
        Layer sharding specification
        
    Example:
        >>> # Split 24 layers across 8 devices (3 layers per device)
        >>> layer_spec = get_model_parallel_sharding(num_layers=24, devices_per_layer=1)
    """
    # For simple model parallelism, assign consecutive layers to consecutive devices
    # This creates a pipeline where each device handles a subset of layers
    
    # Shard layers across 'model' axis
    layer_spec = P('model')
    
    return layer_spec


def get_hybrid_parallel_sharding(
    num_layers: int,
    data_parallel_size: int,
    model_parallel_size: int
):
    """Get sharding spec for hybrid data + model parallelism.
    
    Hybrid parallelism: Combine data and model parallelism for maximum scalability.
    
    Args:
        num_layers: Total number of transformer layers
        data_parallel_size: Number of data parallel replicas
        model_parallel_size: Number of model parallel partitions
        
    Returns:
        Tuple of (batch_spec, layer_spec, param_spec) partition specs
        
    Example:
        >>> # 16 devices: 4 data parallel × 4 model parallel
        >>> specs = get_hybrid_parallel_sharding(
        ...     num_layers=24,
        ...     data_parallel_size=4,
        ...     model_parallel_size=4
        ... )
    """
    assert data_parallel_size * model_parallel_size <= len(jax.devices()), \
        f"Total parallelism ({data_parallel_size} × {model_parallel_size}) exceeds available devices"
    
    # Shard batch across data axis
    batch_spec = P('data', None)
    
    # Shard layers across model axis
    layer_spec = P(None, 'model')
    
    # Some parameters can be sharded across model axis (e.g., large matrices)
    param_spec = P(None, 'model')
    
    return batch_spec, layer_spec, param_spec


def create_sharded_params(
    params: Any,
    mesh: Mesh,
    sharding_spec: Any
):
    """Shard model parameters across devices.
    
    Args:
        params: Model parameters (nested dict/pytree)
        mesh: Device mesh
        sharding_spec: Sharding specification (PartitionSpec)
        
    Returns:
        Sharded parameters distributed across devices
        
    Example:
        >>> mesh = create_device_mesh(num_devices=8, mesh_shape=(8,))
        >>> _, param_spec = get_data_parallel_sharding()
        >>> sharding = jax.sharding.NamedSharding(mesh, param_spec)
        >>> sharded_params = create_sharded_params(params, mesh, sharding)
    """
    # Use jax.device_put with sharding specification
    with mesh:
        if not isinstance(sharding_spec, jax.sharding.Sharding):
            sharding_spec = jax.sharding.NamedSharding(mesh, sharding_spec)
        sharded = jax.device_put(params, sharding_spec)
    
    return sharded


def print_sharding_info(array: jax.Array, name: str = "array"):
    """Print sharding information for a JAX array.
    
    Useful for debugging and understanding how data is distributed.
    
    Args:
        array: JAX array to inspect
        name: Name for display
    """
    print(f"\n{name}:")
    print(f"  Shape: {array.shape}")
    print(f"  Dtype: {array.dtype}")
    print(f"  Sharding: {array.sharding}")
    
    # Show device placement
    if hasattr(array.sharding, 'device_set'):
        devices = array.sharding.device_set
        print(f"  Devices: {len(devices)} device(s)")
        print(f"  Device IDs: {[d.id for d in list(devices)[:5]]}{'...' if len(devices) > 5 else ''}")


def check_sharding_compatibility(
    mesh: Mesh,
    batch_size: int,
    num_layers: int
) -> bool:
    """Check if sharding configuration is compatible with model/data dimensions.
    
    Args:
        mesh: Device mesh
        batch_size: Batch size
        num_layers: Number of model layers
        
    Returns:
        True if configuration is compatible
        
    Raises:
        ValueError: If configuration is incompatible
    """
    # Get mesh dimensions
    mesh_shape = mesh.devices.shape
    
    # Check data parallelism compatibility
    if 'data' in mesh.axis_names:
        data_axis_idx = mesh.axis_names.index('data')
        data_parallel_size = mesh_shape[data_axis_idx]
        
        if batch_size % data_parallel_size != 0:
            raise ValueError(
                f"Batch size ({batch_size}) must be divisible by data parallel size ({data_parallel_size})"
            )
    
    # Check model parallelism compatibility
    if 'model' in mesh.axis_names:
        model_axis_idx = mesh.axis_names.index('model')
        model_parallel_size = mesh_shape[model_axis_idx]
        
        if num_layers % model_parallel_size != 0:
            raise ValueError(
                f"Number of layers ({num_layers}) must be divisible by model parallel size ({model_parallel_size})"
            )
    
    return True


# Example usage and tests
if __name__ == "__main__":
    print("JAX Sharding Utilities")
    print("=" * 80)
    
    # Check available devices
    devices = jax.devices()
    print(f"\nAvailable devices: {len(devices)}")
    for i, device in enumerate(devices[:4]):  # Show first 4
        print(f"  Device {i}: {device}")
    if len(devices) > 4:
        print(f"  ... and {len(devices) - 4} more")
    
    # Test 1: Create 1D mesh (data parallelism)
    print("\n" + "=" * 80)
    print("Test 1: Data Parallelism (1D mesh)")
    print("=" * 80)
    
    num_devices = min(4, len(devices))
    mesh = create_device_mesh(num_devices=num_devices, mesh_shape=(num_devices,), axis_names=('data',))
    print(f"\nMesh shape: {mesh.devices.shape}")
    print(f"Axis names: {mesh.axis_names}")
    
    # Create example batch
    batch_size = 8
    seq_len = 16
    hidden_dim = 64
    
    batch = jnp.ones((batch_size, seq_len, hidden_dim))
    print(f"\nBatch shape: {batch.shape}")
    
    # Shard batch across data axis
    batch_spec, _ = get_data_parallel_sharding()
    with mesh:
        sharded_batch = jax.device_put(batch, jax.sharding.NamedSharding(mesh, batch_spec))
    
    print_sharding_info(sharded_batch, "Sharded batch")
    
    # Test 2: Create 2D mesh (data + model parallelism)
    if len(devices) >= 4:
        print("\n" + "=" * 80)
        print("Test 2: Hybrid Parallelism (2D mesh)")
        print("=" * 80)
        
        mesh_2d = create_device_mesh(
            num_devices=4,
            mesh_shape=(2, 2),
            axis_names=('data', 'model')
        )
        print(f"\nMesh shape: {mesh_2d.devices.shape}")
        print(f"Axis names: {mesh_2d.axis_names}")
        
        # Check compatibility
        try:
            check_sharding_compatibility(mesh_2d, batch_size=8, num_layers=24)
            print("\n✓ Sharding configuration is compatible")
        except ValueError as e:
            print(f"\n✗ Sharding configuration error: {e}")
    
    print("\n" + "=" * 80)
    print("All tests completed!")
