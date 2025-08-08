import pytest
import numpy as np
import torch
import sys
import os

# Add the project root to the Python path to allow for absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.normalization import PheromoneNormalizer

@pytest.fixture
def normalizer():
    """Pytest fixture to provide a PheromoneNormalizer instance."""
    return PheromoneNormalizer()

class TestPheromoneNormalizer:
    """Unit tests for the PheromoneNormalizer class."""

    # 1. Behavior Normalization Tests
    def test_normalize_behavior_numpy(self, normalizer):
        """Test behavior normalization with a NumPy array."""
        raw_scores = np.array([1.0, 2.0, 1.5])
        normalized = normalizer.normalize_behavior(raw_scores)
        assert isinstance(normalized, np.ndarray)
        assert np.isclose(np.sum(normalized), 1.0)
        assert np.all(normalized > 0)

    def test_normalize_behavior_torch(self, normalizer):
        """Test behavior normalization with a PyTorch tensor."""
        raw_scores = torch.tensor([1.0, 2.0, 1.5])
        normalized = normalizer.normalize_behavior(raw_scores)
        assert isinstance(normalized, torch.Tensor)
        assert torch.isclose(torch.sum(normalized), torch.tensor(1.0))
        assert torch.all(normalized > 0)

    # 2. Emotion Normalization Tests
    def test_normalize_emotion_numpy(self, normalizer):
        """Test emotion normalization with a NumPy array."""
        raw_emotions = np.array([-1.0, 0.0, 1.0, 0.5, -0.5])
        normalized = normalizer.normalize_emotion(raw_emotions)
        expected = np.array([0.0, 0.5, 1.0, 0.75, 0.25])
        assert isinstance(normalized, np.ndarray)
        assert np.allclose(normalized, expected)

    def test_normalize_emotion_torch(self, normalizer):
        """Test emotion normalization with a PyTorch tensor."""
        raw_emotions = torch.tensor([-1.0, 0.0, 1.0, 0.5, -0.5])
        normalized = normalizer.normalize_emotion(raw_emotions)
        expected = torch.tensor([0.0, 0.5, 1.0, 0.75, 0.25])
        assert isinstance(normalized, torch.Tensor)
        assert torch.allclose(normalized, expected)

    # 3. Social Normalization Tests
    def test_normalize_social_numpy(self, normalizer):
        """Test social normalization with a NumPy array."""
        raw_counts = np.array([0, 25, 50, 75])
        normalized = normalizer.normalize_social(raw_counts)
        expected = np.array([0.0, 0.5, 1.0, 1.0]) # 75 should be clipped to 1.0
        assert isinstance(normalized, np.ndarray)
        assert np.allclose(normalized, expected)

    def test_normalize_social_torch(self, normalizer):
        """Test social normalization with a PyTorch tensor."""
        raw_counts = torch.tensor([0, 25, 50, 75])
        normalized = normalizer.normalize_social(raw_counts)
        expected = torch.tensor([0.0, 0.5, 1.0, 1.0]) # 75 should be clipped to 1.0
        assert isinstance(normalized, torch.Tensor)
        assert torch.allclose(normalized, expected)
        
    def test_normalize_social_custom_max(self, normalizer):
        """Test social normalization with a custom max_exchanges value."""
        raw_counts = np.array([0, 50, 100])
        normalized = normalizer.normalize_social(raw_counts, max_exchanges=100)
        expected = np.array([0.0, 0.5, 1.0])
        assert np.allclose(normalized, expected)

    # 4. Context Normalization Tests
    def test_normalize_context_numpy_1d(self, normalizer):
        """Test context normalization with a 1D NumPy array (should behave like a single feature)."""
        # For a 1D array, it's treated as a single feature, so min-max over the whole array
        raw_context = np.array([10, 20, 30, 40, 50])
        normalized = normalizer.normalize_context(raw_context)
        expected = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        assert isinstance(normalized, np.ndarray)
        assert normalized.ndim == 1
        assert np.allclose(normalized, expected)

    def test_normalize_context_numpy_2d(self, normalizer):
        """Test context normalization with a 2D NumPy array (feature-wise)."""
        raw_context = np.array([
            [10, 100], # min
            [20, 150],
            [30, 200]  # max
        ])
        normalized = normalizer.normalize_context(raw_context)
        expected = np.array([
            [0.0, 0.0],
            [0.5, 0.5],
            [1.0, 1.0]
        ])
        assert isinstance(normalized, np.ndarray)
        assert normalized.shape == raw_context.shape
        assert np.allclose(normalized, expected)

    def test_normalize_context_torch_2d(self, normalizer):
        """Test context normalization with a 2D PyTorch tensor (feature-wise)."""
        raw_context = torch.tensor([
            [10., 100.], # min
            [20., 150.],
            [30., 200.]  # max
        ], dtype=torch.float32)
        normalized = normalizer.normalize_context(raw_context)
        expected = torch.tensor([
            [0.0, 0.0],
            [0.5, 0.5],
            [1.0, 1.0]
        ])
        assert isinstance(normalized, torch.Tensor)
        assert normalized.shape == raw_context.shape
        assert torch.allclose(normalized, expected)
        
    def test_normalize_context_zero_denominator(self, normalizer):
        """Test context normalization when a feature column is constant."""
        raw_context = np.array([
            [10, 5],
            [20, 5],
            [30, 5]
        ])
        normalized = normalizer.normalize_context(raw_context)
        # The second column should not result in NaN due to zero division
        assert not np.isnan(normalized).any()
        # The second column should be all zeros as (x - min) / (max - min) -> (5-5)/(5-5) -> 0/0 -> handled as 0
        assert np.allclose(normalized[:, 1], 0.0)
        assert np.allclose(normalized[:, 0], np.array([0.0, 0.5, 1.0]))

if __name__ == "__main__":
    pytest.main()
