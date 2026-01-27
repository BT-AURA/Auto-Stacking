"""Tests for SubnetAnalyzer."""

import unittest
from unittest.mock import Mock, patch

from core.analyzer import NeuronMetrics, SubnetAnalyzer


class TestSubnetAnalyzer(unittest.TestCase):
    """Test cases for SubnetAnalyzer."""

    def test_prettify_time(self):
        """Test time formatting."""
        analyzer = SubnetAnalyzer(Mock())
        result = analyzer._prettify_time(90061)  # 1 day, 1 hour, 1 minute, 1 second
        self.assertIn("01d", result)
        self.assertIn("01h", result)
        self.assertIn("01m", result)

    def test_calculate_staking_scores(self):
        """Test score calculation."""
        analyzer = SubnetAnalyzer(Mock())

        metrics = [
            NeuronMetrics(
                uid=1,
                address="1.2.3.4:8091",
                stake=100.0,
                emission=0.5,
                trust=0.8,
                vtrust=0.0,
                daily_rewards_tao=1.0,
                daily_rewards_usd=100.0,
                last_update=10,
                since_registration="10d",
                is_immune=False,
                is_validator=False,
                coldkey="test",
                hotkey="test",
                is_mine=False,
                duplicate_ip=False,
                axon_version="1.0.0",
            ),
            NeuronMetrics(
                uid=2,
                address="1.2.3.5:8091",
                stake=200.0,
                emission=1.0,
                trust=0.9,
                vtrust=0.5,
                daily_rewards_tao=2.0,
                daily_rewards_usd=200.0,
                last_update=5,
                since_registration="5d",
                is_immune=False,
                is_validator=True,
                coldkey="test2",
                hotkey="test2",
                is_mine=False,
                duplicate_ip=False,
                axon_version="1.0.0",
            ),
        ]

        strategy = {
            "weights": {"emission": 1.0, "trust": 1.0, "validator_bonus": 0.5},
            "filters": {"min_stake": 0.0, "max_stake": 10000.0},
        }

        scored = analyzer.calculate_staking_scores(metrics, strategy)

        # Should be sorted by score descending
        self.assertGreater(scored[0].score, scored[1].score)
        # Validator should have higher score due to bonus
        self.assertGreater(scored[0].score, 0)


if __name__ == "__main__":
    unittest.main()
