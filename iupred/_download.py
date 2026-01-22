"""Download and cache AIUPred model weights."""

import os
import sys
import logging
from pathlib import Path
import tarfile
import urllib.request


def _get_cache_dir():
    r"""Get the platform-appropriate cache directory.

    Returns:
        Path to cache directory:
        - Linux: ~/.cache/iupred
        - macOS: ~/Library/Caches/iupred
        - Windows: %LOCALAPPDATA%\iupred\Cache
    """
    if sys.platform == 'win32':
        # Windows: use LOCALAPPDATA
        base = os.environ.get('LOCALAPPDATA')
        if base:
            return Path(base) / 'iupred' / 'Cache'
        # Fallback to APPDATA
        base = os.environ.get('APPDATA')
        if base:
            return Path(base) / 'iupred' / 'Cache'
        # Last resort
        return Path.home() / 'iupred' / 'Cache'
    elif sys.platform == 'darwin':
        # macOS
        return Path.home() / 'Library' / 'Caches' / 'iupred'
    else:
        # Linux and other Unix-like systems
        # Respect XDG_CACHE_HOME if set
        xdg_cache = os.environ.get('XDG_CACHE_HOME')
        if xdg_cache:
            return Path(xdg_cache) / 'iupred'
        return Path.home() / '.cache' / 'iupred'


# Cache directory for downloaded data
CACHE_DIR = _get_cache_dir()

# URL for the AIUPred data archive
AIUPRED_DATA_URL = 'https://static.omnipathdb.org/aiupred-data.tar.gz'

# Expected files in the archive
AIUPRED_FILES = [
    'binding_decoder.pt',
    'binding_transform',
    'disorder_decoder.pt',
    'embedding_binding.pt',
    'embedding_disorder.pt',
]

_logger = logging.getLogger(__name__)


def _ensure_cache_dir():
    """Ensure the cache directory exists."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _is_data_available():
    """Check if all AIUPred data files are available in cache."""
    if not CACHE_DIR.exists():
        return False
    return all((CACHE_DIR / f).exists() for f in AIUPRED_FILES)


def _download_and_extract():
    """Download and extract the AIUPred data archive."""
    _ensure_cache_dir()

    archive_path = CACHE_DIR / 'aiupred-data.tar.gz'

    _logger.info(
        f'Downloading AIUPred model weights from {AIUPRED_DATA_URL}...'
    )
    print('Downloading AIUPred model weights (~82 MB)...')

    try:
        urllib.request.urlretrieve(AIUPRED_DATA_URL, archive_path)
    except Exception as e:
        raise RuntimeError(
            f'Failed to download AIUPred data from {AIUPRED_DATA_URL}: {e}'
        ) from e

    _logger.info(f'Extracting archive to {CACHE_DIR}...')
    print('Extracting model weights...')

    try:
        with tarfile.open(archive_path, 'r:gz') as tar:
            # Extract all files to cache directory
            for member in tar.getmembers():
                # Get just the filename (strip any directory prefix)
                member.name = Path(member.name).name
                tar.extract(member, CACHE_DIR)
    except Exception as e:
        raise RuntimeError(
            f'Failed to extract AIUPred data archive: {e}'
        ) from e

    # Clean up the archive
    archive_path.unlink()

    _logger.info('AIUPred model weights ready.')
    print('AIUPred model weights ready.')


def ensure_aiupred_data():
    """Ensure AIUPred data files are available, downloading if necessary.

    This function checks if all required AIUPred model files are present
    in the cache directory. If not, it downloads and extracts them from
    the remote archive.

    The data is cached in ~/.cache/iupred/ and only downloaded once.

    Returns:
        Path to the cache directory containing the data files.

    Raises:
        RuntimeError: If download or extraction fails.
    """
    if not _is_data_available():
        _download_and_extract()

    return CACHE_DIR


def get_aiupred_file(filename):
    """Get the path to an AIUPred data file, downloading if necessary.

    Args:
        filename: Name of the data file (e.g., 'embedding_disorder.pt').

    Returns:
        Path to the requested file.

    Raises:
        FileNotFoundError: If the file is not found after download attempt.
        RuntimeError: If download fails.
    """
    ensure_aiupred_data()

    file_path = CACHE_DIR / filename
    if not file_path.exists():
        raise FileNotFoundError(
            f"AIUPred file '{filename}' not found in cache. "
            f'Expected files: {", ".join(AIUPRED_FILES)}'
        )

    return file_path


def clear_cache():
    """Remove all cached AIUPred data files.

    This can be useful if you need to re-download the data or free up space.
    """
    if CACHE_DIR.exists():
        for f in AIUPRED_FILES:
            file_path = CACHE_DIR / f
            if file_path.exists():
                file_path.unlink()
                _logger.info(f'Removed {file_path}')

        # Also remove any leftover archive
        archive_path = CACHE_DIR / 'aiupred-data.tar.gz'
        if archive_path.exists():
            archive_path.unlink()

        _logger.info('AIUPred cache cleared.')
