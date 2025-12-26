# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-12-26

### Changed

- **BREAKING**: `MoneyFlow` dataclass expanded from 4 to 15 fields - now includes edge-level temporal context, reciprocity metrics, and behavioral patterns from `core_money_flows_view`
  - Added: `first_seen_timestamp`, `last_seen_timestamp`, `active_days`, `avg_tx_size_usd`
  - Added: `unique_assets`, `dominant_asset`, `hourly_pattern[24]`, `weekly_pattern[7]`
  - Added: `reciprocity_ratio`, `is_bidirectional`
- **BREAKING**: `ClickHouseAdapter.read_money_flows()` now queries `core_money_flows_view` instead of re-aggregating `core_transfers`
  - Returns lifetime aggregates for edges active in time window (semantic change)
  - 5x faster query performance via pre-computed materialized view
- **BREAKING**: `BurstDetector.detect()` signature now requires `timestamp_data` parameter
  - Burst detection uses dedicated transfer timestamp query for precision
  - Other pattern detectors unaffected (use MV graph)

### Added

- `ClickHouseAdapter.read_transfer_timestamps()` - Targeted query for burst detection temporal analysis
- `AddressFeatureAnalyzer._compute_edge_features()` - 7 new relationship-context features:
  - `avg_relationship_age_days`, `max_relationship_age_days` - Relationship maturity metrics
  - `bidirectional_relationship_ratio`, `avg_edge_reciprocity` - Reciprocity analysis
  - `multi_asset_edge_ratio` - Asset diversity across relationships
  - `edge_hourly_entropy`, `edge_weekly_entropy` - Temporal pattern diversity
- **Schema**: 7 new columns in `analyzers_features` table leveraging MV edge data

### Performance

- Overall pipeline speedup: ~5x (MV eliminates 4 of 5 transfer table scans)
- Feature computation: Behavioral/temporal features now use pre-aggregated edge arrays
- Graph building: 13 edge attributes from MV vs 2 from re-aggregation

## [0.1.3] - 2025-12-10

### Changed

- **BREAKING**: `ClickHouseAdapter.delete_partition()` has been removed and replaced with two separate methods:
  - `delete_features_partition(window_days, processing_date)` - Deletes only from `analyzers_features` table
  - `delete_patterns_partition(window_days, processing_date)` - Deletes only from `analyzers_patterns_*` tables
  - This prevents unintended data loss when running feature computation and pattern detection sequentially

## [0.1.1] - 2025-12-10

### Fixed

- **Zero USD Price Handling** - Fixed multiple ZeroDivisionError crashes when processing networks with missing USD price data
  - `NetworkDetector._detect_smurfing()`: Added fallback to tx_count as weight when USD values are zero
  - `AddressFeatureAnalyzer._compute_flow_features()`: Added total_volume > 0 check for concentration_ratio
  - `AddressFeatureAnalyzer._compute_pagerank()`: Fallback to unweighted PageRank when graph has zero weights
  - `AddressFeatureAnalyzer._compute_closeness_centrality()`: Fallback to hop-based distance when weights are zero
  - `AddressFeatureAnalyzer._compute_clustering_coefficient()`: Fallback to unweighted clustering when weights are zero
  - `AddressFeatureAnalyzer._compute_community_detection()`: Fallback to tx_count as weight for Leiden algorithm
  - `MotifDetector._calculate_time_concentration()`: Added protection against floating-point edge cases

## [0.1.0] - 2025-12-08

### Added

- Initial release of `chainswarm-analyzers-baseline`
- **I/O Adapters**
  - `ParquetAdapter` for file-based I/O (tournament testing)
  - `ClickHouseAdapter` for database I/O (production)
- **Feature Computation**
  - `AddressFeatureAnalyzer` with 70+ features per address
  - Volume, statistical, flow, graph, behavioral, and label-based features
- **Pattern Detection**
  - `CycleDetector` - Circular transaction patterns
  - `LayeringDetector` - Long transaction chains
  - `NetworkDetector` - Smurfing network patterns
  - `ProximityDetector` - Distance to risky addresses
  - `MotifDetector` - Fan-in/fan-out patterns
  - `BurstDetector` - Temporal burst patterns
  - `ThresholdDetector` - Threshold evasion patterns
- **Pipeline**
  - `BaselineAnalyzersPipeline` orchestrator
  - `create_pipeline()` factory function
- **Configuration**
  - `SettingsLoader` for network-specific settings
  - JSON-based configuration with inheritance
- **Protocols**
  - `InputAdapter`, `OutputAdapter`, `DataAdapter` interfaces
  - `FeatureAnalyzer`, `PatternAnalyzer` protocols
  - Data models: `Transfer`, `AddressFeatures`, `PatternDetection`
- **CLI Scripts**
  - `run-pipeline` - Full pipeline execution
  - `run-features` - Feature computation only
  - `run-patterns` - Pattern detection only