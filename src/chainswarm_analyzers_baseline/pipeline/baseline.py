import time
from typing import Union, Dict, Any, List, Protocol

from loguru import logger

from ..adapters.parquet import ParquetAdapter
from ..adapters.clickhouse import ClickHouseAdapter
from ..aggregates.transfer_aggregates import compute_transfer_aggregates
from ..config import SettingsLoader
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
        timestamp_data: Dict[str, List[Dict[str, Any]]],
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
        network: str,
    ):
        self.adapter = adapter
        self.feature_analyzer = feature_analyzer
        self.pattern_analyzer = pattern_analyzer
        self.network = network
    
    def run(
        self,
        start_timestamp_ms: int,
        end_timestamp_ms: int,
        window_days: int,
        processing_date: str,
        run_features: bool = True,
        run_patterns: bool = True,
    ) -> Dict[str, Any]:
        
        money_flows = self.adapter.read_money_flows(start_timestamp_ms, end_timestamp_ms)
        logger.info(f"Loaded {len(money_flows)} money flows from MV")
        
        graph = build_money_flow_graph(money_flows)
        addresses = extract_addresses_from_flows(money_flows)
        address_labels = self.adapter.read_address_labels(addresses)
        
        transfer_aggregates = {}
        timestamp_data = {}
        
        if run_patterns:
            transfers = self.adapter.read_transfers(start_timestamp_ms, end_timestamp_ms)
            transfer_aggregates = compute_transfer_aggregates(transfers)
            timestamp_data = self.adapter.read_transfer_timestamps(
                start_timestamp_ms, end_timestamp_ms, addresses
            )
            logger.info(f"Loaded transfer data for burst detection")
        
        features_count = 0
        patterns_count = 0
        
        if run_features:
            features_count = self._run_feature_analysis(
                graph=graph,
                address_labels=address_labels,
                transfer_aggregates=transfer_aggregates,
                window_days=window_days,
                processing_date=processing_date,
            )
        
        if run_patterns:
            patterns_count = self._run_pattern_analysis(
                money_flows=money_flows,
                address_labels=address_labels,
                timestamp_data=timestamp_data,
                window_days=window_days,
                processing_date=processing_date,
            )
        
        return {
            'features_count': features_count,
            'patterns_count': patterns_count,
        }
    
    def run_features_only(
        self,
        start_timestamp_ms: int,
        end_timestamp_ms: int,
        window_days: int,
        processing_date: str,
    ) -> int:
        result = self.run(
            start_timestamp_ms=start_timestamp_ms,
            end_timestamp_ms=end_timestamp_ms,
            window_days=window_days,
            processing_date=processing_date,
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
    ) -> int:
        result = self.run(
            start_timestamp_ms=start_timestamp_ms,
            end_timestamp_ms=end_timestamp_ms,
            window_days=window_days,
            processing_date=processing_date,
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
        timestamp_data: Dict[str, List[Dict[str, Any]]],
        window_days: int,
        processing_date: str,
    ) -> int:
        
        patterns = self.pattern_analyzer.analyze(
            money_flows=money_flows,
            address_labels=address_labels,
            timestamp_data=timestamp_data,
            window_days=window_days,
            processing_date=processing_date
        )
        
        self.adapter.write_patterns(patterns, window_days, processing_date)
        
        return len(patterns)


def create_pipeline(
    adapter: Union[ParquetAdapter, ClickHouseAdapter],
    network: str,
    settings_loader: SettingsLoader,
) -> BaselineAnalyzersPipeline:
    feature_analyzer = AddressFeatureAnalyzer()
    pattern_analyzer = StructuralPatternAnalyzer(
        settings_loader=settings_loader,
        network=network
    )
    
    return BaselineAnalyzersPipeline(
        adapter=adapter,
        feature_analyzer=feature_analyzer,
        pattern_analyzer=pattern_analyzer,
        network=network,
    )
