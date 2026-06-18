"""Neck modules required by SD-MSCF-DET."""

from .auxiliary import GSConvE, SNI
from .soep import CSPOmniKernel, SPDConv

__all__ = ("SPDConv", "CSPOmniKernel", "SNI", "GSConvE")
