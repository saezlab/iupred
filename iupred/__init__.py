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
    'configure_model_urls',
]

from ._metadata import __author__, __version__
from .iupred2a import anchor2, iupred
from .aiupred_lib import aiupred_binding, aiupred_disorder, configure_model_urls
