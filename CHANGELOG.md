# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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