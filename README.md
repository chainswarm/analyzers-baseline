# Analyzers-Baseline

**Baseline analytics algorithms for blockchain pattern detection and feature engineering.**

This package provides the official baseline implementation for the ChainSwarm Analytics Tournament.

## Overview

`analyzers-baseline` extracts core analytical algorithms from `analytics-pipeline/packages/analyzers/` and provides:

1. **Feature Computation** - 70+ features per address including volume, graph, temporal, and behavioral metrics
2. **Pattern Detection** - 7 pattern types: cycles, layering paths, smurfing networks, motifs, proximity risk, temporal bursts, and threshold evasion

## Documentation

For detailed architecture and implementation details, see:
- [**ARCHITECTURE.md**](docs/ARCHITECTURE.md) - Full architecture documentation

## Installation

### Using UV (Recommended)

[UV](https://docs.astral.sh/uv/) is a fast Python package manager written in Rust.

```bash
# Install uv if you haven't
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a new virtual environment with Python 3.13
cd analyzers-baseline
uv venv --python 3.13

# Activate the virtual environment
# Linux/macOS:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

# Install the package with all dependencies
uv pip install -e .

# For development (includes pytest)
uv pip install -e ".[dev]"
```

### Using pip

```bash
# Basic installation (includes ClickHouse support via chainswarm-core)
pip install analyzers-baseline

# For development
pip install -e ".[dev]"
```

## Quick Start

### Production Usage (ClickHouse)

```python
from chainswarm_core.db import ClientFactory, get_connection_params
from analyzers_baseline import BaselineAnalyzersPipeline
from analyzers_baseline.adapters import ClickHouseAdapter

# Get connection params from environment (uses CLICKHOUSE_* env vars)
connection_params = get_connection_params(
    network="torus",
    database_prefix="analytics"
)

# Create client using chainswarm-core
factory = ClientFactory(connection_params)
client = factory.create_client()

# Create adapter with ClickHouse client
adapter = ClickHouseAdapter(client=client, network="torus")

# Run pipeline
pipeline = BaselineAnalyzersPipeline(adapter=adapter)
result = pipeline.run(
    start_timestamp_ms=1700000000000,
    end_timestamp_ms=1702600000000,
    window_days=30,
    processing_date="2025-01-15",
    network="torus"
)
```

### Tournament Testing (Parquet)

```python
from analyzers_baseline import BaselineAnalyzersPipeline
from analyzers_baseline.adapters import ParquetAdapter

# Create adapter with file paths
adapter = ParquetAdapter(
    input_path="./input",
    output_path="./output"
)

# Run pipeline
pipeline = BaselineAnalyzersPipeline(adapter=adapter)
result = pipeline.run(
    start_timestamp_ms=1700000000000,
    end_timestamp_ms=1702600000000,
    window_days=30,
    processing_date="2025-01-15",
    network="torus"
)
```

### CLI Usage

The CLI auto-extracts metadata (network, date, window) from the input path structure:

```
data/input/{network}/{processing_date}/{window_days}/
```

```bash
# Run full pipeline - metadata auto-extracted from path
run-pipeline --input data/input/torus/2025-01-15/30

# Output auto-constructed as data/output/torus/2025-01-15/30/

# Override extracted values if needed
run-pipeline \
    --input data/input/torus/2025-01-15/30 \
    --output ./custom-output \
    --network bittensor  # Override network

# Run full pipeline (ClickHouse mode - uses CLICKHOUSE_* env vars)
run-pipeline \
    --clickhouse \
    --network torus \
    --window-days 30 \
    --processing-date 2025-01-15

# Run features only
run-features --input data/input/torus/2025-01-15/30

# Run patterns only
run-patterns --input data/input/torus/2025-01-15/30
```

## Package Structure

```
analyzers_baseline/
├── protocols/      # Abstract interfaces (Python Protocols)
├── features/       # Feature computation implementations
├── patterns/       # Pattern detection implementations
├── adapters/       # I/O adapters (Parquet, ClickHouse)
├── graph/          # Graph building utilities
├── pipeline/       # Production pipeline
├── config/         # Configuration management
└── scripts/        # Script entry points
```

## Data Directory Structure

For Parquet mode, the path structure encodes metadata:

```
data/
├── input/{network}/{processing_date}/{window_days}/
│   ├── transfers.parquet           # Required
│   ├── money_flows.parquet         # Optional
│   ├── assets.parquet              # Optional
│   ├── asset_prices.parquet        # Optional
│   └── address_labels.parquet      # Optional
└── output/{network}/{processing_date}/{window_days}/
    ├── features_w{window_days}.parquet
    ├── patterns_cycle.parquet
    ├── patterns_layering.parquet
    └── ...
```

Example:
```
data/
├── input/torus/2025-01-15/30/
│   └── transfers.parquet
└── output/torus/2025-01-15/30/
    ├── features_w30.parquet
    └── patterns_*.parquet
```

## Data Schemas

All data schemas match `data-pipeline` core tables for compatibility.

### Input Files (Parquet Mode)

#### transfers.parquet
Balance transfer data matching `core_transfers` schema:

| Column | Type | Description |
|--------|------|-------------|
| `tx_id` | String | Transaction hash (EVM/Substrate/UTXO) |
| `event_index` | String | Event index within transaction |
| `edge_index` | String | Edge disambiguator (UTXO) |
| `block_height` | UInt32 | Block number |
| `block_timestamp` | UInt64 | Milliseconds since epoch |
| `from_address` | String | Source address |
| `to_address` | String | Destination address |
| `asset_symbol` | String | Asset symbol (TAO, USDT, etc.) |
| `asset_contract` | String | Contract address or 'native' |
| `amount` | Decimal128(18) | Native token amount |
| `amount_usd` | Decimal128(18) | USD value at transaction time |
| `fee` | Decimal128(18) | Transaction fee |

#### assets.parquet
Asset metadata matching `core_assets` schema:

| Column | Type | Description |
|--------|------|-------------|
| `asset_symbol` | String | Asset symbol |
| `asset_contract` | String | Contract address or 'native' |
| `network` | String | Network name (torus, bittensor, etc.) |
| `verified` | Boolean | Verification status |
| `verification_source` | String | Source of verification |
| `first_seen_timestamp` | UInt64 | Discovery timestamp |

#### asset_prices.parquet
Historical USD prices matching `core_asset_prices` schema:

| Column | Type | Description |
|--------|------|-------------|
| `asset_symbol` | String | Asset symbol |
| `asset_contract` | String | Contract address or 'native' |
| `price_date` | Date32 | Price date |
| `price_usd` | Decimal128(18) | USD price |
| `source` | String | Price source |

#### address_labels.parquet
Known address labels matching `core_address_labels` schema:

| Column | Type | Description |
|--------|------|-------------|
| `address` | String | Blockchain address |
| `label` | String | Human-readable label |
| `address_type` | String | Type classification |
| `trust_level` | String | Trust level |
| `source` | String | Label source |

### Output Files

| File | Description |
|------|-------------|
| `features.parquet` | 70+ computed features per address |
| `patterns_cycle.parquet` | Cycle patterns |
| `patterns_layering.parquet` | Layering path patterns |
| `patterns_network.parquet` | Smurfing network patterns |
| `patterns_proximity.parquet` | Proximity risk patterns |
| `patterns_motif.parquet` | Fan-in/fan-out motif patterns |
| `patterns_burst.parquet` | Temporal burst patterns |
| `patterns_threshold.parquet` | Threshold evasion patterns |

### S3 Export Compatibility

Input files are compatible with `chain-synthetics` S3 exports:
```
s3://bucket/snapshots/{network}/{processing_date}/{window_days}/
├── transfers.parquet         ← Input for analyzers-baseline
├── assets.parquet            ← Input for analyzers-baseline
├── asset_prices.parquet      ← Input for analyzers-baseline
├── address_labels.parquet    ← Input for analyzers-baseline
├── ground_truth.parquet      ← For benchmark validation
└── META.json
```

## Pattern Types

Pattern types use lowercase values from `chainswarm_core.constants.PatternTypes`:

| Pattern Type | Value | Description |
|--------------|-------|-------------|
| Cycle | `cycle` | Circular transaction patterns |
| Layering Path | `layering_path` | Long transaction chains |
| Smurfing Network | `smurfing_network` | Fragmented value transfers |
| Motif Fan-In | `motif_fanin` | Many-to-one patterns |
| Motif Fan-Out | `motif_fanout` | One-to-many patterns |
| Proximity Risk | `proximity_risk` | Distance to risky addresses |
| Temporal Burst | `temporal_burst` | High-frequency activity |
| Threshold Evasion | `threshold_evasion` | Structuring below limits |

## Requirements

- Python >= 3.11 (tested with 3.11, 3.12, 3.13)
- chainswarm-core >= 0.1.13 (provides clickhouse-connect, loguru, pydantic)
- networkx >= 3.0
- numpy >= 1.24
- pandas >= 2.0
- pyarrow >= 14.0
- click >= 8.0

## License

Apache-2.0