from ._mlx_backend import MLXBackend
"""Backend for TinyGrad operations."""

MLX = MLXBackend()

__all__ = [key for key in globals().keys() if not key.startswith('_')]
