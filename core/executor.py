"""Staking Executor - Signs and sends staking transactions with safety features."""

import logging
from typing import List, Optional

import bittensor
from bittensor import Balance

from .strategist import StakingDecision

logger = logging.getLogger(__name__)


class StakingExecutor:
    """Executes staking transactions with safety checks and confirmation."""

    def __init__(self, config: bittensor.Config, dry_run: bool = False):
        """Initialize executor with config and dry-run mode."""
        self.config = config
        self.dry_run = dry_run
        self.subtensor = bittensor.Subtensor(
            config=config, network=config.chain_endpoint, log_verbose=False
        )
        self.wallet = bittensor.wallet(config=config)

    def _confirm_transaction(self, decision: StakingDecision) -> bool:
        """Confirm transaction with user (if not dry-run)."""
        if self.dry_run:
            return True

        # In interactive mode, this would prompt user
        # For now, return True (can be enhanced with CLI prompts)
        return True

    def execute_stake(
        self, decision: StakingDecision, wait_for_inclusion: bool = True
    ) -> Optional[str]:
        """Execute a single staking transaction."""
        try:
            if not self._confirm_transaction(decision):
                logger.info(f"Skipping stake to UID {decision.uid} (user cancelled)")
                return None

            if self.dry_run:
                logger.info(
                    f"[DRY RUN] Would stake {decision.amount_tao:.4f} TAO to UID {decision.uid} "
                    f"({decision.hotkey[:12]}...) - {decision.reason}"
                )
                return "dry_run_success"

            logger.info(
                f"Staking {decision.amount_tao:.4f} TAO to UID {decision.uid} "
                f"({decision.hotkey[:12]}...)"
            )

            # Get hotkey address from metagraph
            metagraph = self.subtensor.metagraph(self.config.netuid)
            metagraph.sync(lite=False)
            if decision.uid not in metagraph.uids.tolist():
                logger.error(f"UID {decision.uid} not found in metagraph")
                return None

            neuron = metagraph.neurons[decision.uid]
            hotkey_address = neuron.hotkey

            # Execute stake
            success = self.subtensor.add_stake(
                wallet=self.wallet,
                hotkey_ss58=hotkey_address,
                amount=Balance.from_tao(decision.amount_tao),
                wait_for_inclusion=wait_for_inclusion,
            )

            if success:
                logger.info(f"Successfully staked {decision.amount_tao:.4f} TAO to UID {decision.uid}")
                return hotkey_address
            else:
                logger.error(f"Failed to stake to UID {decision.uid}")
                return None

        except Exception as e:
            logger.error(f"Error executing stake to UID {decision.uid}: {e}")
            return None

    def execute_batch(
        self, decisions: List[StakingDecision], wait_for_inclusion: bool = True
    ) -> List[Optional[str]]:
        """Execute multiple staking transactions."""
        results = []
        total_amount = sum(d.amount_tao for d in decisions)

        logger.info(f"Executing batch of {len(decisions)} stakes totaling {total_amount:.4f} TAO")

        for i, decision in enumerate(decisions, 1):
            logger.info(f"Processing stake {i}/{len(decisions)}")
            result = self.execute_stake(decision, wait_for_inclusion=wait_for_inclusion)
            results.append(result)

            # Small delay between transactions to avoid rate limiting
            if not self.dry_run and i < len(decisions):
                import time
                time.sleep(1)

        successful = sum(1 for r in results if r is not None)
        logger.info(f"Batch complete: {successful}/{len(decisions)} successful")
        return results

    def unstake(
        self, uid: int, amount_tao: float, wait_for_inclusion: bool = True
    ) -> bool:
        """Unstake TAO from a neuron."""
        try:
            if self.dry_run:
                logger.info(f"[DRY RUN] Would unstake {amount_tao:.4f} TAO from UID {uid}")
                return True

            metagraph = self.subtensor.metagraph(self.config.netuid)
            metagraph.sync(lite=False)
            if uid not in metagraph.uids.tolist():
                logger.error(f"UID {uid} not found in metagraph")
                return False

            neuron = metagraph.neurons[uid]
            hotkey_address = neuron.hotkey

            success = self.subtensor.unstake(
                wallet=self.wallet,
                hotkey_ss58=hotkey_address,
                amount=Balance.from_tao(amount_tao),
                wait_for_inclusion=wait_for_inclusion,
            )

            if success:
                logger.info(f"Successfully unstaked {amount_tao:.4f} TAO from UID {uid}")
            else:
                logger.error(f"Failed to unstake from UID {uid}")

            return success

        except Exception as e:
            logger.error(f"Error unstaking from UID {uid}: {e}")
            return False

    def get_stake_summary(self) -> dict:
        """Get summary of current stakes."""
        try:
            metagraph = self.subtensor.metagraph(self.config.netuid)
            if metagraph is not None:
                metagraph.sync(lite=False)

            coldkey_address = self.wallet.coldkeypub.ss58_address
            total_staked = 0.0
            stake_count = 0

            for uid in metagraph.uids.tolist():
                neuron = metagraph.neurons[uid]
                for stake_info in neuron.stake:
                    if stake_info[0] == coldkey_address:
                        total_staked += stake_info[1].tao
                        stake_count += 1

            return {
                "total_staked_tao": total_staked,
                "stake_count": stake_count,
                "coldkey": coldkey_address,
            }
        except Exception as e:
            logger.error(f"Error getting stake summary: {e}")
            return {"total_staked_tao": 0.0, "stake_count": 0, "coldkey": ""}
