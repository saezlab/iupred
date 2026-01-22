#!/usr/bin/env python

#
# This file is part of the `iupred` Python module
#
# Copyright 2026
# Eötvös Loránd University (ELTE)
#
# File author(s): Zsuzsanna Dosztányi, Gábor Erdős
#                 (zsuzsanna.dosztanyi@ttk.elte.hu)
#
# Distributed under the CC-BY-NC-ND-4.0 license
# See the file `LICENSE` for details
#

"""IUPred and AIUPred: Prediction of intrinsically disordered protein regions."""

__all__ = [
    '__version__',
    '__author__',
    # IUPred2 functions
    'iupred',
    'anchor2',
    # AIUPred functions
    'aiupred_disorder',
    'aiupred_binding',
    # Data management
    'ensure_aiupred_data',
    'clear_aiupred_cache',
]

from .aiupred import aiupred_binding, aiupred_disorder
from .iupred2a import iupred, anchor2
from ._download import (
    clear_cache as clear_aiupred_cache,
    ensure_aiupred_data,
)
from ._metadata import __author__, __version__
