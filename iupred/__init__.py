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
    # AIUPred batch processing
    'init_aiupred_models',
    'predict_aiupred_disorder',
    'predict_aiupred_binding',
    # AIUPred low-memory functions
    'low_memory_aiupred_disorder',
    'low_memory_aiupred_binding',
    # Data management
    'ensure_aiupred_data',
    'clear_aiupred_cache',
]

from .aiupred import (
    init_models as init_aiupred_models,
    aiupred_binding,
    predict_binding as predict_aiupred_binding,
    aiupred_disorder,
    predict_disorder as predict_aiupred_disorder,
    low_memory_predict_binding as low_memory_aiupred_binding,
    low_memory_predict_disorder as low_memory_aiupred_disorder,
)
from .iupred2a import iupred, anchor2
from ._download import (
    clear_cache as clear_aiupred_cache,
    ensure_aiupred_data,
)
from ._metadata import __author__, __version__
