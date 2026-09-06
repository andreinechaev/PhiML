from ._tinygrad_backend import TinyGradBackend
"""Backend for TinyGrad operations."""

TINY_GRAD = TinyGradBackend()

__all__ = [key for key in globals().keys() if not key.startswith('_')]
