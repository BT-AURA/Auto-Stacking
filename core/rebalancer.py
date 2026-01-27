"""Stake Rebalancer - Automatically rebalances stakes based on strategy."""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import bittensor

from .analyzer import NeuronMetrics, SubnetAnalyzer
from .executor import StakingExecutor
from .strategist import StakingDecision, StakingStrategist

logger = logging.getLogger(__name__)


@dataclass
class RebalanceAction:
    """Represents a rebalancing action."""

    action: str  # "add", "remove", or "reallocate"
    uid: int
    current_stake: float
    target_stake: float
    amount_tao: float  # Amount to add/remove
    reason: str
    priority: int


class StakeRebalancer:
    """Rebalances stakes based on strategy and current allocations."""

    def __init__(
        self,
        analyzer: SubnetAnalyzer,
        strategist: StakingStrategist,
        executor: StakingExecutor,
        strategy: Dict,
    ):
        """Initialize rebalancer with analyzer, strategist, executor, and strategy."""
        self.analyzer = analyzer
        self.strategist = strategist
        self.executor = executor
        self.strategy = strategy

    def _get_current_allocations(self) -> Dict[int, float]:
        """Get current stake allocations per UID."""
        return self.strategist._get_current_stakes()

    def calculate_rebalance_actions(
        self, metrics: List[NeuronMetrics], dry_run: bool = False
    ) -> List[RebalanceAction]:
        """Calculate rebalancing actions needed based on strategy."""
        logger.info("Calculating rebalancing actions...")

        # Get current allocations
        current_stakes = self._get_current_allocations()

        # Get target allocations from strategist
        target_decisions = self.strategist.make_decisions(metrics, dry_run=dry_run)

        # Build target allocation map
        target_allocations = {d.uid: d.amount_tao for d in target_decisions}

        actions = []
        rebalance_threshold = self.strategy.get("rebalance_threshold", 0.15)

        # Check existing stakes that need adjustment
        for uid, current_stake in current_stakes.items():
            target_stake = target_allocations.get(uid, 0.0)

            # Calculate drift
            if current_stake > 0:
                drift = abs(target_stake - current_stake) / current_stake
            else:
                drift = 1.0 if target_stake > 0 else 0.0

            # If drift exceeds threshold, create action
            if drift > rebalance_threshold:
                if target_stake > current_stake:
                    # Need to add stake
                    amount = target_stake - current_stake
                    actions.append(
                        RebalanceAction(
                            action="add",
                            uid=uid,
                            current_stake=current_stake,
                            target_stake=target_stake,
                            amount_tao=amount,
                            reason=f"Rebalance: drift {drift:.2%} exceeds threshold",
                            priority=1,
                        )
                    )
                elif target_stake < current_stake and target_stake > 0:
                    # Need to reduce stake (but keep some)
                    amount = current_stake - target_stake
                    actions.append(
                        RebalanceAction(
                            action="remove",
                            uid=uid,
                            current_stake=current_stake,
                            target_stake=target_stake,
                            amount_tao=amount,
                            reason=f"Rebalance: drift {drift:.2%} exceeds threshold",
                            priority=2,
                        )
                    )
                elif target_stake == 0:
                    # Remove entirely (not in target list)
                    actions.append(
                        RebalanceAction(
                            action="remove",
                            uid=uid,
                            current_stake=current_stake,
                            target_stake=0.0,
                            amount_tao=current_stake,
                            reason="Not in target allocation",
                            priority=3,
                        )
                    )

        # Check for new targets that need initial stake
        for decision in target_decisions:
            if decision.uid not in current_stakes:
                actions.append(
                    RebalanceAction(
                        action="add",
                        uid=decision.uid,
                        current_stake=0.0,
                        target_stake=decision.amount_tao,
                        amount_tao=decision.amount_tao,
                        reason=decision.reason,
                        priority=0,  # Highest priority for new stakes
                    )
                )

        # Sort by priority (lower = higher priority)
        actions.sort(key=lambda x: x.priority)

        logger.info(f"Calculated {len(actions)} rebalancing actions")
        return actions

    def execute_rebalance(
        self, actions: List[RebalanceAction], dry_run: bool = False
    ) -> Dict[str, int]:
        """Execute rebalancing actions."""
        results = {"add": 0, "remove": 0, "failed": 0}

        logger.info(f"Executing {len(actions)} rebalancing actions...")

        for action in actions:
            try:
                if action.action == "add":
                    if self.executor.dry_run or dry_run:
                        logger.info(
                            f"[DRY RUN] Would add {action.amount_tao:.4f} TAO to UID {action.uid}"
                        )
                        results["add"] += 1
                    else:
                        # Create a decision for adding stake
                        decision = StakingDecision(
                            uid=action.uid,
                            hotkey="",  # Will be fetched in executor
                            amount_tao=action.amount_tao,
                            reason=action.reason,
                            priority=action.priority,
                        )
                        result = self.executor.execute_stake(decision)
                        if result:
                            results["add"] += 1
                        else:
                            results["failed"] += 1

                elif action.action == "remove":
                    if self.executor.dry_run or dry_run:
                        logger.info(
                            f"[DRY RUN] Would remove {action.amount_tao:.4f} TAO from UID {action.uid}"
                        )
                        results["remove"] += 1
                    else:
                        success = self.executor.unstake(
                            action.uid, action.amount_tao
                        )
                        if success:
                            results["remove"] += 1
                        else:
                            results["failed"] += 1

            except Exception as e:
                logger.error(f"Error executing rebalance action for UID {action.uid}: {e}")
                results["failed"] += 1

        logger.info(
            f"Rebalance complete: {results['add']} added, {results['remove']} removed, "
            f"{results['failed']} failed"
        )
        return results

    def should_rebalance(self) -> bool:
        """Check if rebalancing is needed based on strategy settings."""
        rebalance_interval = self.strategy.get("rebalance_interval_blocks", 2000)
        rebalance_threshold = self.strategy.get("rebalance_threshold", 0.15)

        # Check if enough time has passed (would need to track last rebalance)
        # For now, always return True if threshold is set
        # In production, this would check last rebalance block
        return True

    def get_rebalance_summary(self) -> Dict:
        """Get summary of current vs target allocations."""
        current_stakes = self._get_current_allocations()
        total_current = sum(current_stakes.values())

        # Get target decisions (would need fresh metrics)
        # For now, return current state
        return {
            "total_staked": total_current,
            "allocations": len(current_stakes),
            "current_stakes": current_stakes,
        }
