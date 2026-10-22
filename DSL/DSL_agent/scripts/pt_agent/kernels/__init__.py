"""The R4 single-ciphertext mux kernel interface."""

from .layout import FlatLayout, KernelError, MuxLayout, mux_layout, pack_flat, pack_mux, unpack_flat, unpack_mux
from .conv import add_mux, conv2d_mux, downsample_pad_mux
from .poly import EXACT, act_cheb, cheb_eval, cheb_fit
from .pool import mean_mux
from .linear import linear_flat

__all__ = ["MuxLayout", "mux_layout", "pack_mux", "unpack_mux", "conv2d_mux",
           "downsample_pad_mux", "add_mux", "KernelError", "FlatLayout", "pack_flat", "unpack_flat",
           "EXACT", "act_cheb", "cheb_fit", "cheb_eval", "mean_mux", "linear_flat"]
