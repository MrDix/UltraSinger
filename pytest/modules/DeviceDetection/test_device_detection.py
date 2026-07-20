"""Tests for the device detection module (auto Whisper batch size)."""

from unittest.mock import MagicMock, patch

from modules.DeviceDetection.device_detection import auto_whisper_batch_size


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
