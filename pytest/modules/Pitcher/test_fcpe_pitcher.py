"""Tests for FCPE pitch detection backend."""

import unittest
from unittest.mock import patch, MagicMock

import numpy as np

from src.modules.Pitcher.pitched_data import PitchedData


class TestFcpePitcherOutput(unittest.TestCase):
    """Test FCPE pitcher produces valid PitchedData."""

    @patch("src.modules.Pitcher.fcpe_pitcher._get_model")
    def test_returns_pitched_data(self, mock_get_model):
        """FCPE wrapper returns PitchedData with correct structure."""
        import torch

        # Mock FCPE model output: [batch=1, frames=100, 1]
        fake_f0 = torch.zeros(1, 100, 1)
        # Set some voiced frames (frequencies > 0)
        for i in range(20, 80):
            fake_f0[0, i, 0] = 440.0  # A4

        mock_model = MagicMock()
        mock_model.infer.return_value = fake_f0
        mock_get_model.return_value = (mock_model, "cpu")

        from src.modules.Pitcher.fcpe_pitcher import get_pitch_with_fcpe

        audio = np.random.randn(16000).astype(np.float32)  # 1s at 16kHz
        result = get_pitch_with_fcpe(audio, 16000)

        self.assertEqual(type(result).__name__, "PitchedData")
        self.assertEqual(len(result.times), 100)
        self.assertEqual(len(result.frequencies), 100)
        self.assertEqual(len(result.confidence), 100)

    @patch("src.modules.Pitcher.fcpe_pitcher._get_model")
    def test_confidence_reflects_voicing(self, mock_get_model):
        """Voiced frames get 0.9 confidence, unvoiced get 0.0."""
        import torch

        fake_f0 = torch.zeros(1, 50, 1)
        fake_f0[0, 10:40, 0] = 440.0  # Voiced in middle

        mock_model = MagicMock()
        mock_model.infer.return_value = fake_f0
        mock_get_model.return_value = (mock_model, "cpu")

        from src.modules.Pitcher.fcpe_pitcher import get_pitch_with_fcpe

        audio = np.random.randn(16000).astype(np.float32)
        result = get_pitch_with_fcpe(audio, 16000)

        # Unvoiced frames should have 0.0 confidence
        for i in range(10):
            self.assertEqual(result.confidence[i], 0.0)
        # Voiced frames should have 0.9 confidence
        for i in range(10, 40):
            self.assertEqual(result.confidence[i], 0.9)

    @patch("src.modules.Pitcher.fcpe_pitcher._get_model")
    def test_timestamps_are_monotonic(self, mock_get_model):
        """Timestamps should be monotonically increasing."""
        import torch

        fake_f0 = torch.zeros(1, 200, 1)
        mock_model = MagicMock()
        mock_model.infer.return_value = fake_f0
        mock_get_model.return_value = (mock_model, "cpu")

        from src.modules.Pitcher.fcpe_pitcher import get_pitch_with_fcpe

        audio = np.random.randn(32000).astype(np.float32)
        result = get_pitch_with_fcpe(audio, 16000)

        for i in range(1, len(result.times)):
            self.assertGreater(result.times[i], result.times[i - 1])

    @patch("src.modules.Pitcher.fcpe_pitcher._get_model")
    def test_negative_frequencies_clipped(self, mock_get_model):
        """Negative frequency values should be clipped to 0."""
        import torch

        fake_f0 = torch.tensor([[[-10.0], [0.0], [440.0]]])
        mock_model = MagicMock()
        mock_model.infer.return_value = fake_f0
        mock_get_model.return_value = (mock_model, "cpu")

        from src.modules.Pitcher.fcpe_pitcher import get_pitch_with_fcpe

        audio = np.random.randn(16000).astype(np.float32)
        result = get_pitch_with_fcpe(audio, 16000)

        self.assertEqual(result.frequencies[0], 0.0)
        self.assertEqual(result.frequencies[1], 0.0)
        self.assertAlmostEqual(result.frequencies[2], 440.0, places=1)


class TestPitcherSetting(unittest.TestCase):
    """Test that the pitcher setting controls which backend is used."""

    def test_default_pitcher_is_swiftf0(self):
        """Default pitcher should be swiftf0."""
        from src.Settings import Settings
        s = Settings()
        self.assertEqual(s.pitcher, "swiftf0")

    def test_pitcher_setting_accepts_fcpe(self):
        """Pitcher setting should accept 'fcpe'."""
        from src.Settings import Settings
        s = Settings()
        s.pitcher = "fcpe"
        self.assertEqual(s.pitcher, "fcpe")


if __name__ == "__main__":
    unittest.main()
