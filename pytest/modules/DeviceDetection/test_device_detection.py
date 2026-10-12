"""Tests for the device detection module (auto Whisper batch size)."""

from unittest.mock import MagicMock, patch

import pytest

from modules.DeviceDetection.device_detection import auto_whisper_batch_size, free_gpu_memory, release_gpu_memory


def _gpu_props(vram_gb: float):
    props = MagicMock()
    props.total_memory = int(vram_gb * 1024 ** 3)
    return props


class TestAutoWhisperBatchSize:
    """VRAM-based scaling: 16 on 8+ GB, 8 on 6 GB, 4 on 4 GB cards."""

    def test_cpu_keeps_historic_default(self):
        assert auto_whisper_batch_size("cpu") == 16

    def test_cuda_unavailable_keeps_historic_default(self):
        with patch("torch.cuda.is_available", return_value=False):
            assert auto_whisper_batch_size("cuda") == 16

    def test_4gb_card_gets_4(self):
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.get_device_properties",
                   return_value=_gpu_props(4.0)):
            assert auto_whisper_batch_size("cuda") == 4

    def test_6gb_card_gets_8(self):
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.get_device_properties",
                   return_value=_gpu_props(6.0)):
            assert auto_whisper_batch_size("cuda") == 8

    def test_7gb_card_gets_8(self):
        """7 GB is not enough for batch 16 (which needs ~8 GB)."""
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.get_device_properties",
                   return_value=_gpu_props(7.0)):
            assert auto_whisper_batch_size("cuda") == 8

    def test_8gb_card_reporting_slightly_less_gets_16(self):
        """True 8 GB cards often report a bit less than 8.0 GB."""
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.get_device_properties",
                   return_value=_gpu_props(7.9)):
            assert auto_whisper_batch_size("cuda") == 16

    def test_8gb_card_gets_16(self):
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.get_device_properties",
                   return_value=_gpu_props(8.0)):
            assert auto_whisper_batch_size("cuda") == 16

    def test_12gb_card_gets_16(self):
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.get_device_properties",
                   return_value=_gpu_props(12.0)):
            assert auto_whisper_batch_size("cuda") == 16

    def test_query_failure_keeps_historic_default(self):
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.get_device_properties",
                   side_effect=RuntimeError("driver in a bad state")):
            assert auto_whisper_batch_size("cuda") == 16


class TestFreeGpuMemory:
    """Free for a new model: what the driver reports free plus PyTorch's unused cache."""

    def test_no_gpu(self):
        with patch("torch.cuda.is_available", return_value=False):
            assert free_gpu_memory() is None

    def test_free_and_unused_cache(self):
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.mem_get_info", return_value=(1000, 4000)), \
             patch("torch.cuda.memory_reserved", return_value=700), \
             patch("torch.cuda.memory_allocated", return_value=200):
            assert free_gpu_memory() == 1500

    def test_query_failure_is_no_missing_gpu(self):
        # A GPU whose memory cannot be read must not look like "no GPU" (which skips the check)
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.mem_get_info", side_effect=RuntimeError("driver in a bad state")):
            with pytest.raises(RuntimeError):
                free_gpu_memory()

    def test_release_hands_back_the_cache(self):
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.empty_cache") as empty_cache:
            release_gpu_memory()
        empty_cache.assert_called_once()

    def test_release_on_a_gpu_in_an_error_state(self):
        # Called from finally blocks: it must not replace the error being raised
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.empty_cache", side_effect=RuntimeError("CUDA error: an illegal memory access")):
            release_gpu_memory()
