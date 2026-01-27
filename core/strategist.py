"""Staking Strategist - Decides where to stake based on strategy configuration."""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import bittensor
from bittensor import Balance

from .analyzer import NeuronMetrics, SubnetAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class StakingDecision:
    """Represents a staking decision for a specific neuron."""

    uid: int
    hotkey: str
    amount_tao: float
    reason: str
    priority: int  # Lower is higher priority


class StakingStrategist:
    """Makes staking decisions based on strategy configuration."""

    def __init__(self, analyzer: SubnetAnalyzer, strategy: Dict):
        """Initialize strategist with analyzer and strategy config."""
        self.analyzer = analyzer
        self.strategy = strategy
        self.config = analyzer.config

    def _get_available_balance(self) -> float:
        """Get available TAO balance for staking."""
        try:
            wallet = bittensor.wallet(config=self.config)
            balance = self.analyzer.subtensor.get_balance(wallet.coldkeypub.ss58_address)
            return balance.tao
        except Exception as e:
            logger.error(f"Error getting balance: {e}")
            return 0.0

    def _get_current_stakes(self) -> Dict[int, float]:
        """Get current stake amounts for all neurons."""
        stakes = {}
        try:
            wallet = bittensor.wallet(config=self.config)
            coldkey_address = wallet.coldkeypub.ss58_address

            # Get all current stakes
            metagraph = self.analyzer.metagraph
            if metagraph is None:
                return stakes

            for uid in metagraph.uids.tolist():
                neuron = metagraph.neurons[uid]
                # Check if we have stake in this neuron
                for stake_info in neuron.stake:
                    if stake_info[0] == coldkey_address:
                        stakes[uid] = stake_info[1].tao
                        break

        except Exception as e:
            logger.error(f"Error getting current stakes: {e}")

        return stakes

    def make_decisions(
        self, metrics: List[NeuronMetrics], dry_run: bool = False
    ) -> List[StakingDecision]:
        """Make staking decisions based on metrics and strategy."""
        logger.info("Making staking decisions...")

        # Calculate scores if not already calculated
        if not metrics or metrics[0].score == 0.0:
            metrics = self.analyzer.calculate_staking_scores(metrics, self.strategy)
        
        # Apply APY and risk filters
        metrics = self._apply_apy_risk_filters(metrics)

        # Filter out invalid neurons
        valid_metrics = [m for m in metrics if m.score > 0]

        # Strategy parameters
        max_targets = self.strategy.get("max_targets", 10)
        min_stake_per_target = self.strategy.get("min_stake_per_target", 1.0)
        max_stake_per_target = self.strategy.get("max_stake_per_target", 1000.0)
        allocation_method = self.strategy.get("allocation_method", "equal")  # equal, proportional, top_n

        # Get available balance
        available_balance = self._get_available_balance()
        logger.info(f"Available balance: {available_balance:.4f} TAO")

        if available_balance < min_stake_per_target:
            logger.warning("Insufficient balance for staking")
            return []

        # Get current stakes
        current_stakes = self._get_current_stakes()

        # Select top targets
        top_targets = valid_metrics[:max_targets]

        decisions = []
        total_to_stake = 0.0

        if allocation_method == "equal":
            stake_per_target = min(
                available_balance / len(top_targets), max_stake_per_target
            )
            stake_per_target = max(stake_per_target, min_stake_per_target)

            for i, metric in enumerate(top_targets):
                # Check if we already have stake
                current_stake = current_stakes.get(metric.uid, 0.0)
                target_stake = stake_per_target

                # Adjust for existing stake
                if current_stake > 0:
                    if current_stake >= target_stake:
                        continue  # Already staked enough
                    target_stake = target_stake - current_stake

                if target_stake < min_stake_per_target:
                    continue

                if total_to_stake + target_stake > available_balance:
                    break

                decisions.append(
                    StakingDecision(
                        uid=metric.uid,
                        hotkey=metric.hotkey,
                        amount_tao=target_stake,
                        reason=f"Top {i+1} target (score: {metric.score:.4f})",
                        priority=i + 1,
                    )
                )
                total_to_stake += target_stake

        elif allocation_method == "proportional":
            # Allocate proportionally based on scores
            total_score = sum(m.score for m in top_targets)
            if total_score > 0:
                for i, metric in enumerate(top_targets):
                    proportion = metric.score / total_score
                    target_stake = min(
                        available_balance * proportion, max_stake_per_target
                    )
                    target_stake = max(target_stake, min_stake_per_target)

                    current_stake = current_stakes.get(metric.uid, 0.0)
                    if current_stake > 0:
                        if current_stake >= target_stake:
                            continue
                        target_stake = target_stake - current_stake

                    if target_stake < min_stake_per_target:
                        continue

                    if total_to_stake + target_stake > available_balance:
                        break

                    decisions.append(
                        StakingDecision(
                            uid=metric.uid,
                            hotkey=metric.hotkey,
                            amount_tao=target_stake,
                            reason=f"Proportional allocation (score: {metric.score:.4f})",
                            priority=i + 1,
                        )
                    )
                    total_to_stake += target_stake

        elif allocation_method == "top_n":
            # Stake maximum amount in top N targets
            n = self.strategy.get("top_n_count", 3)
            top_n = top_targets[:n]
            stake_per_target = min(
                available_balance / len(top_n), max_stake_per_target
            )

            for i, metric in enumerate(top_n):
                current_stake = current_stakes.get(metric.uid, 0.0)
                target_stake = stake_per_target

                if current_stake > 0:
                    if current_stake >= target_stake:
                        continue
                    target_stake = target_stake - current_stake

                if target_stake < min_stake_per_target:
                    continue

                if total_to_stake + target_stake > available_balance:
                    break

                decisions.append(
                    StakingDecision(
                        uid=metric.uid,
                        hotkey=metric.hotkey,
                        amount_tao=target_stake,
                        reason=f"Top {i+1} target (score: {metric.score:.4f})",
                        priority=i + 1,
                    )
                )
                total_to_stake += target_stake

        logger.info(f"Made {len(decisions)} staking decisions totaling {total_to_stake:.4f} TAO")
        return decisions
    
    def _apply_apy_risk_filters(self, metrics: List[NeuronMetrics]) -> List[NeuronMetrics]:
        """Apply APY and risk-based filters from strategy."""
        min_apy_7d = self.strategy.get("min_apy_7d", 0.0)
        min_apy_30d = self.strategy.get("min_apy_30d", 0.0)
        max_risk_score = self.strategy.get("max_risk_score", 1.0)
        exclude_flags = self.strategy.get("exclude_risk_flags", [])
        
        filtered = []
        for metric in metrics:
            # APY filters
            if min_apy_7d > 0 and metric.apy_7d < min_apy_7d:
                continue
            if min_apy_30d > 0 and metric.apy_30d < min_apy_30d:
                continue
            
            # Risk score filter
            if metric.risk_score > max_risk_score:
                continue
            
            # Risk flag exclusions
            if any(flag in metric.risk_flags for flag in exclude_flags):
                continue
            
            filtered.append(metric)
        
        logger.info(f"Filtered {len(metrics)} -> {len(filtered)} neurons based on APY/risk criteria")
        return filtered

    def should_restake(self, metrics: List[NeuronMetrics]) -> bool:
        """Determine if restaking is needed based on strategy."""
        rebalance_threshold = self.strategy.get("rebalance_threshold", 0.1)
        rebalance_interval_blocks = self.strategy.get("rebalance_interval_blocks", 1000)

        # Check if enough time has passed
        if self.analyzer.metagraph is None:
            return False

        # This would need to track last rebalance time
        # For now, always return True if threshold is met
        return True
