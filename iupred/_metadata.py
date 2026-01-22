#!/usr/bin/env python

#
# This file is part of the `iupred` Python module
#
# Copyright 2026
# Research Centre for Natural Sciences, Hungarian Academy of Sciences
#
# File author(s): Zsuzsanna Dosztányi, Gábor Erdős (zsuzsanna.dosztanyi@ttk.elte.hu)
#
# Distributed under the MIT license
# See the file `LICENSE` or read a copy at
# https://opensource.org/license/mit
#

"""Package metadata (version, authors, etc)."""

__all__ = ['__version__', '__author__', '__license__']

import importlib.metadata


_FALLBACK_VERSION = '0.1.0'

try:
    __version__ = importlib.metadata.version('iupred')
except importlib.metadata.PackageNotFoundError:
    # Package not installed (e.g. running from source checkout)
    __version__ = _FALLBACK_VERSION

__author__ = 'Zsuzsanna Dosztányi, Gábor Erdős'
__license__ = 'MIT'
