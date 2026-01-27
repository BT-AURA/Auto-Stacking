# Bittensor Auto-Staker

Automated staking system for Bittensor subnets that analyzes subnet metrics, calculates optimal staking targets, and executes staking transactions based on configurable strategies.

## Features

- **Comprehensive Analysis**: Detailed metrics calculation for subnet analysis
- **Strategy-Based Decisions**: YAML-configurable strategies (conservative, aggressive, custom)
- **Safe Execution**: Dry-run mode, confirmation prompts, and transaction safety checks
- **Historical Caching**: Optional SQLite cache for tracking subnet metrics over time
- **User-Friendly CLI**: Rich terminal interface with clear output and progress indicators

## Installation

```bash
# Create virtual environment
python -m venv venv
source ./venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Analyze a Subnet (No Staking)

```bash
# Analyze subnet 0 and show top 10 targets
python cli.py analyze --netuid 0 --top-n 10

# Use a strategy for scoring
python cli.py analyze --netuid 0 --strategy strategies/conservative.yaml
```

### 2. Dry Run (Test Without Staking)

```bash
# Simulate staking without executing transactions
python cli.py stake --netuid 0 --strategy strategies/conservative.yaml --dry-run
```

### 3. Execute Staking

```bash
# Execute staking with confirmation prompt
python cli.py stake --netuid 0 --strategy strategies/conservative.yaml

# Skip confirmation (use with caution!)
python cli.py stake --netuid 0 --strategy strategies/conservative.yaml --yes
```

### 4. Check Status

```bash
# View current staking status
python cli.py status --netuid 0
```

## Project Structure

```
bittensor-auto-staker/
├── core/
│   ├── analyzer.py       # Subnet metrics calculation and analysis
│   ├── strategist.py     # Makes staking decisions based on strategy
│   └── executor.py       # Signs & sends staking transactions safely
├── data/
│   └── historical/       # SQLite cache for subnet metrics (optional)
├── strategies/
│   ├── conservative.yaml # Conservative strategy (high trust, lower risk)
│   └── aggressive.yaml   # Aggressive strategy (high emission, higher risk)
├── cli.py                # User-friendly CLI interface
├── requirements.txt
├── README.md
└── tests/
```

## Strategy Configuration

Strategies are defined in YAML files. See `strategies/conservative.yaml` and `strategies/aggressive.yaml` for examples.

### Strategy Parameters

- **weights**: Scoring weights for different metrics
  - `emission`: Weight for emission-based scoring
  - `trust`: Weight for trust-based scoring
  - `validator_bonus`: Bonus score for validators

- **filters**: Criteria to exclude neurons
  - `min_stake`: Minimum existing stake to consider
  - `max_stake`: Maximum stake threshold
  - `exclude_immune`: Exclude immune neurons
  - `exclude_duplicate_ip`: Exclude duplicate IP addresses
  - `min_trust`: Minimum trust score
  - `min_emission`: Minimum emission threshold

- **allocation**: How to distribute stakes
  - `max_targets`: Maximum number of targets
  - `min_stake_per_target`: Minimum TAO per target
  - `max_stake_per_target`: Maximum TAO per target
  - `allocation_method`: `equal`, `proportional`, or `top_n`

- **rebalancing**: When to rebalance
  - `rebalance_threshold`: Drift threshold for rebalancing
  - `rebalance_interval_blocks`: Minimum blocks between rebalances

### Creating Custom Strategies

Create a new YAML file in the `strategies/` directory:

```yaml
name: my_custom_strategy
description: "My custom staking strategy"

weights:
  emission: 1.5
  trust: 1.0
  validator_bonus: 0.3

filters:
  min_stake: 50.0
  max_stake: 5000.0
  exclude_immune: false
  exclude_duplicate_ip: true
  min_trust: 0.2
  min_emission: 0.01

max_targets: 8
min_stake_per_target: 20.0
max_stake_per_target: 1000.0
allocation_method: proportional

rebalance_threshold: 0.15
rebalance_interval_blocks: 1500

max_total_stake_ratio: 0.85
require_confirmation: true
```

## CLI Commands

### `analyze`

Analyze subnet metrics without executing any staking.

```bash
python cli.py analyze [OPTIONS]
```

Options:
- `--netuid`: Subnet UID (default: 0)
- `--strategy`: Strategy file path (optional, for scoring)
- `--top-n`: Number of top targets to display (default: 10)
- `--skip-registration`: Skip registration block queries (faster)

### `stake`

Execute staking based on strategy configuration.

```bash
python cli.py stake [OPTIONS]
```

Options:
- `--netuid`: Subnet UID (default: 0)
- `--strategy`: Strategy YAML file path (required)
- `--dry-run`: Simulate without executing transactions
- `--yes`: Skip confirmation prompt (use with caution!)
- `--wait`: Wait for transaction inclusion (default: True)
- `--skip-registration`: Skip registration block queries (faster)

### `status`

Show current staking status and summary.

```bash
python cli.py status [OPTIONS]
```

Options:
- `--netuid`: Subnet UID (default: 0)

### `check platform`

Quick platform check: connectivity, chain, TAO price, and subnet hyperparameters. Uses a lite metagraph sync, so it finishes in seconds.

```bash
python cli.py check platform [--netuid 0]
```

Options:
- `--netuid`: Subnet UID for hyperparameters (default: 0)

Example output: chain endpoint, current block, TAO price, plus `tempo`, `immunity_period`, `neurons`, and `validators` for the chosen subnet.

## Safety Features

1. **Dry Run Mode**: Test strategies without executing transactions
2. **Confirmation Prompts**: Require explicit confirmation before transactions
3. **Balance Checks**: Verify sufficient balance before transactions
4. **Transaction Limits**: Configurable min/max amounts
5. **Strategy Validation**: Validate strategy configuration before execution

## Examples

### Conservative Staking

```bash
# Analyze with conservative strategy
python cli.py analyze --netuid 0 --strategy strategies/conservative.yaml

# Dry run
python cli.py stake --netuid 0 --strategy strategies/conservative.yaml --dry-run

# Execute
python cli.py stake --netuid 0 --strategy strategies/conservative.yaml
```

### Aggressive Staking

```bash
# Analyze with aggressive strategy
python cli.py analyze --netuid 1 --strategy strategies/aggressive.yaml --top-n 20

# Execute (aggressive strategy has require_confirmation: false, so use --yes carefully)
python cli.py stake --netuid 1 --strategy strategies/aggressive.yaml --yes
```

### Custom Network

```bash
# Use custom network endpoint
python cli.py analyze --netuid 0 --subtensor.network finney --strategy strategies/conservative.yaml
```

## Data Caching

Historical metrics are optionally cached in SQLite for analysis:

- Location: `data/historical/metrics.db`
- Automatically created on first use
- Can be cleared with cache management (future feature)

## Development

### Running Tests

```bash
python -m pytest tests/
```

### Code Formatting

```bash
black .
```

### Pre-commit Hooks

```bash
pre-commit install
pre-commit run --all-files
```

## Safety Disclaimers

⚠️ **IMPORTANT WARNINGS:**

1. **This tool executes real transactions** on the Bittensor blockchain. Always test with `--dry-run` first.
2. **Start with small amounts** to verify behavior before large transactions.
3. **Review strategy configurations** carefully before execution.
4. **Monitor your positions** regularly and adjust strategies as needed.
5. **No guarantees** - subnet performance can change, and decisions are based on historical data.
6. **Use at your own risk** - the authors are not responsible for any losses.

## Contributing

Contributions are welcome! Please:

1. Use pre-commit hooks (black, etc.)
2. Add tests for new features
3. Update documentation
4. Submit a PR with clear description

## License

See LICENSE file.

## Original Tool

The core analysis functionality is implemented in `core/analyzer.py`.
