"""Tests for optional torch dependency handling."""

from unittest import mock

import pytest


class TestOptionalTorch:
    """Tests that the package works without torch installed."""

    def test_check_torch_available_raises_when_unavailable(self):
        """_check_torch_available should raise ImportError with helpful message."""
        from iupred import aiupred

        # Save original values
        orig_available = aiupred._TORCH_AVAILABLE
        orig_error = aiupred._TORCH_IMPORT_ERROR

        try:
            # Simulate torch not being available
            aiupred._TORCH_AVAILABLE = False
            aiupred._TORCH_IMPORT_ERROR = ImportError("No module named 'torch'")

            with pytest.raises(ImportError) as exc_info:
                aiupred._check_torch_available()

            error_msg = str(exc_info.value)
            assert 'pip install iupred[aiupred]' in error_msg
            assert 'PyTorch' in error_msg
            assert 'IUPred2 and ANCHOR2 work without PyTorch' in error_msg
        finally:
            # Restore original values
            aiupred._TORCH_AVAILABLE = orig_available
            aiupred._TORCH_IMPORT_ERROR = orig_error

    def test_init_models_raises_when_torch_unavailable(self):
        """init_models should raise ImportError when torch is not available."""
        from iupred import aiupred

        orig_available = aiupred._TORCH_AVAILABLE
        orig_error = aiupred._TORCH_IMPORT_ERROR

        try:
            aiupred._TORCH_AVAILABLE = False
            aiupred._TORCH_IMPORT_ERROR = ImportError("No module named 'torch'")

            with pytest.raises(ImportError) as exc_info:
                aiupred.init_models('disorder')

            assert 'pip install iupred[aiupred]' in str(exc_info.value)
        finally:
            aiupred._TORCH_AVAILABLE = orig_available
            aiupred._TORCH_IMPORT_ERROR = orig_error

    def test_aiupred_disorder_raises_when_torch_unavailable(self):
        """aiupred_disorder should raise ImportError when torch is not available."""
        from iupred import aiupred

        orig_available = aiupred._TORCH_AVAILABLE
        orig_error = aiupred._TORCH_IMPORT_ERROR

        try:
            aiupred._TORCH_AVAILABLE = False
            aiupred._TORCH_IMPORT_ERROR = ImportError("No module named 'torch'")

            with pytest.raises(ImportError) as exc_info:
                aiupred.aiupred_disorder('MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNG')

            assert 'pip install iupred[aiupred]' in str(exc_info.value)
        finally:
            aiupred._TORCH_AVAILABLE = orig_available
            aiupred._TORCH_IMPORT_ERROR = orig_error

    def test_aiupred_binding_raises_when_torch_unavailable(self):
        """aiupred_binding should raise ImportError when torch is not available."""
        from iupred import aiupred

        orig_available = aiupred._TORCH_AVAILABLE
        orig_error = aiupred._TORCH_IMPORT_ERROR

        try:
            aiupred._TORCH_AVAILABLE = False
            aiupred._TORCH_IMPORT_ERROR = ImportError("No module named 'torch'")

            with pytest.raises(ImportError) as exc_info:
                aiupred.aiupred_binding('MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNG')

            assert 'pip install iupred[aiupred]' in str(exc_info.value)
        finally:
            aiupred._TORCH_AVAILABLE = orig_available
            aiupred._TORCH_IMPORT_ERROR = orig_error

    def test_iupred2_works_independently(self):
        """IUPred2 functions should work regardless of torch availability."""
        from iupred import iupred, anchor2

        sequence = 'MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNG'
        scores, _ = iupred(sequence, mode='long')

        assert len(scores) == len(sequence)
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_error_message_includes_original_error(self):
        """Error message should include the original ImportError."""
        from iupred import aiupred

        orig_available = aiupred._TORCH_AVAILABLE
        orig_error = aiupred._TORCH_IMPORT_ERROR

        try:
            original_msg = "No module named 'torch'"
            aiupred._TORCH_AVAILABLE = False
            aiupred._TORCH_IMPORT_ERROR = ImportError(original_msg)

            with pytest.raises(ImportError) as exc_info:
                aiupred._check_torch_available()

            assert original_msg in str(exc_info.value)
        finally:
            aiupred._TORCH_AVAILABLE = orig_available
            aiupred._TORCH_IMPORT_ERROR = orig_error
