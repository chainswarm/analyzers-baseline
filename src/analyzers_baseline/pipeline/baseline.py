import time
from typing import Union, Dict, Any, List, Protocol

from loguru import logger

from ..adapters.parquet import ParquetAdapter
from ..adapters.clickhouse import ClickHouseAdapter
from ..aggregates.transfer_aggregates import compute_transfer_aggregates
from ..features.address_feature_analyzer import AddressFeatureAnalyzer
from ..patterns.structural_pattern_analyzer import StructuralPatternAnalyzer
from ..graph.builder import build_money_flow_graph, extract_addresses_from_flows


class FeatureAnalyzerProtocol(Protocol):
    def analyze(
        self,
        graph: Any,
        address_labels: Dict[str, Dict[str, Any]],
        transfer_aggregates: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        ...


class PatternAnalyzerProtocol(Protocol):
    def analyze(
        self,
        money_flows: List[Dict[str, Any]],
        address_labels: Dict[str, Dict[str, Any]],
        window_days: int,
        processing_date: str
    ) -> List[Dict[str, Any]]:
        ...


class BaselineAnalyzersPipeline:
    def __init__(
        self,
        adapter: Union[ParquetAdapter, ClickHouseAdapter],
        feature_analyzer: FeatureAnalyzerProtocol,
        pattern_analyzer: PatternAnalyzerProtocol,
    ):
        self.adapter = adapter
        self.feature_analyzer = feature_analyzer
        self.pattern_analyzer = pattern_analyzer
    
    def run(
        self,
        start_timestamp_ms: int,
        end_timestamp_ms: int,
        window_days: int,
        processing_date: str,
        network: str = "unknown",
        run_features: bool = True,
        run_patterns: bool = True,
    ) -> Dict[str, Any]:
        start_time = time.time()
        
        logger.info(
            f"Starting pipeline: window_days={window_days}, "
            f"processing_date={processing_date}, network={network}"
        )
        logger.info(
            f"Time range: {start_timestamp_ms} to {end_timestamp_ms}"
        )
        
        features_count = 0
        patterns_count = 0
        
        logger.info("Loading transfers...")
        transfers = self.adapter.read_transfers(start_timestamp_ms, end_timestamp_ms)
        logger.info(f"Loaded {len(transfers)} transfers")
        
        if not transfers:
            raise ValueError(f"No transfers found in time range {start_timestamp_ms} to {end_timestamp_ms}")
        
        logger.info("Computing transfer aggregates...")
        transfer_aggregates = compute_transfer_aggregates(transfers)
        logger.info(f"Computed aggregates for {len(transfer_aggregates)} addresses")
        
        logger.info("Aggregating to money flows...")
        money_flows = self.adapter.read_money_flows(start_timestamp_ms, end_timestamp_ms)
        logger.info(f"Aggregated to {len(money_flows)} money flows")
        
        if not money_flows:
            raise ValueError(f"No money flows found in time range {start_timestamp_ms} to {end_timestamp_ms}")
        
        addresses = extract_addresses_from_flows(money_flows)
        logger.info(f"Found {len(addresses)} unique addresses")
        
        logger.info("Loading address labels...")
        address_labels = self.adapter.read_address_labels(addresses)
        logger.info(f"Loaded labels for {len(address_labels)} addresses")
        
        logger.info("Building transaction graph...")
        graph = build_money_flow_graph(money_flows)
        logger.info(f"Graph built: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        
        if run_features:
            features_count = self._run_feature_analysis(
                graph=graph,
                address_labels=address_labels,
                transfer_aggregates=transfer_aggregates,
                window_days=window_days,
                processing_date=processing_date,
            )
        
        # Run pattern analysis
        if run_patterns:
            patterns_count = self._run_pattern_analysis(
                money_flows=money_flows,
                address_labels=address_labels,
                window_days=window_days,
                processing_date=processing_date,
            )
        
        duration = time.time() - start_time
        
        logger.info(
            f"Pipeline complete: {features_count} features, "
            f"{patterns_count} patterns in {duration:.2f}s"
        )
        
        return {
            "features_count": features_count,
            "patterns_count": patterns_count,
            "duration_seconds": duration
        }
    
    def run_features_only(
        self,
        start_timestamp_ms: int,
        end_timestamp_ms: int,
        window_days: int,
        processing_date: str,
        network: str = "unknown",
    ) -> int:
        result = self.run(
            start_timestamp_ms=start_timestamp_ms,
            end_timestamp_ms=end_timestamp_ms,
            window_days=window_days,
            processing_date=processing_date,
            network=network,
            run_features=True,
            run_patterns=False,
        )
        return result["features_count"]
    
    def run_patterns_only(
        self,
        start_timestamp_ms: int,
        end_timestamp_ms: int,
        window_days: int,
        processing_date: str,
        network: str = "unknown",
    ) -> int:
        result = self.run(
            start_timestamp_ms=start_timestamp_ms,
            end_timestamp_ms=end_timestamp_ms,
            window_days=window_days,
            processing_date=processing_date,
            network=network,
            run_features=False,
            run_patterns=True,
        )
        return result["patterns_count"]
    
    def _run_feature_analysis(
        self,
        graph,
        address_labels: Dict[str, Dict[str, Any]],
        transfer_aggregates: Dict[str, Dict[str, Any]],
        window_days: int,
        processing_date: str,
    ) -> int:
        logger.info("Running feature analysis...")
        
        features_dict = self.feature_analyzer.analyze(
            graph, address_labels, transfer_aggregates
        )
        
        # Convert dict-of-dicts to list for adapter
        features_list = list(features_dict.values())
        
        if not features_list:
            raise ValueError("No features computed from graph analysis")
        
        logger.info(f"Writing {len(features_list)} features...")
        self.adapter.write_features(
            features=features_list,
            window_days=window_days,
            processing_date=processing_date
        )
        logger.info(f"Feature analysis complete: {len(features_list)} features")
        
        return len(features_list)
    
    def _run_pattern_analysis(
        self,
        money_flows: List[Dict[str, Any]],
        address_labels: Dict[str, Dict[str, Any]],
        window_days: int,
        processing_date: str,
    ) -> int:
        logger.info("Running pattern analysis...")
        
        # Detect patterns
        patterns = self.pattern_analyzer.analyze(
            money_flows=money_flows,
            address_labels=address_labels,
            window_days=window_days,
            processing_date=processing_date
        )
        
        if patterns:
            logger.info(f"Writing {len(patterns)} patterns...")
            self.adapter.write_patterns(
                patterns=patterns,
                window_days=window_days,
                processing_date=processing_date
            )
            logger.info(f"Pattern analysis complete: {len(patterns)} patterns")
        else:
            logger.info("No patterns detected")
        
        return len(patterns)


def create_pipeline(
    adapter: Union[ParquetAdapter, ClickHouseAdapter],
    feature_analyzer: FeatureAnalyzerProtocol = None,
    pattern_analyzer: PatternAnalyzerProtocol = None,
) -> BaselineAnalyzersPipeline:
    return BaselineAnalyzersPipeline(
        adapter=adapter,
        feature_analyzer=feature_analyzer or AddressFeatureAnalyzer(),
        pattern_analyzer=pattern_analyzer or StructuralPatternAnalyzer(),
    )