from typing import Dict, List, Any, Optional

import networkx as nx
from loguru import logger

from analyzers_baseline.patterns.base_detector import BasePatternDetector
from analyzers_baseline.patterns.detectors import (
    CycleDetector,
    LayeringDetector,
    NetworkDetector,
    ProximityDetector,
    MotifDetector,
    BurstDetector,
    ThresholdDetector,
)
from analyzers_baseline.graph.builder import build_money_flow_graph


class StructuralPatternAnalyzer:

    def __init__(
        self,
        detectors: Optional[List[BasePatternDetector]] = None,
        config: Optional[Dict[str, Any]] = None,
        network: Optional[str] = None
    ):
        self.config = config or self._get_default_config()
        self.network = network
        
        if detectors is None:
            self.detectors = self._create_default_detectors()
        else:
            self.detectors = detectors
        
        logger.info(
            f"Initialized StructuralPatternAnalyzer with {len(self.detectors)} detectors"
        )

    def _get_default_config(self) -> Dict[str, Any]:
        return {
            "cycle_detection": {
                "min_cycle_length": 3,
                "max_cycle_length": 10,
                "max_cycles_per_scc": 100,
            },
            "path_analysis": {
                "min_path_length": 3,
                "max_path_length": 10,
                "max_paths_to_check": 1000,
                "high_volume_percentile": 90,
                "max_source_nodes": 50,
                "max_target_nodes": 50,
                "layering_min_volume": 1000,
                "layering_cv_threshold": 0.5,
            },
            "scc_analysis": {
                "min_scc_size": 5,
            },
            "network_analysis": {
                "min_community_size": 5,
                "max_community_size": 100,
                "small_transaction_threshold": 1000,
                "small_transaction_ratio_threshold": 0.7,
                "density_threshold": 0.1,
            },
            "sybil_detection": {
                "min_network_size": 10,
                "similarity_threshold": 0.8,
            },
            "proximity_analysis": {
                "max_distance": 3,
                "distance_decay_factor": 1.0,
            },
            "risk_identification": {
                "high_volume_threshold": 100000,
                "high_degree_threshold": 50,
            },
            "motif_detection": {
                "degree_percentile_threshold": 90,
                "fanin_max_out_degree": 3,
                "fanout_max_in_degree": 3,
                "min_spoke_count": 5,
            },
            "burst_detection": {
                "min_burst_intensity": 3.0,
                "min_burst_transactions": 10,
                "time_window_seconds": 3600,
                "z_score_threshold": 2.0,
            },
            "threshold_detection": {
                "reporting_threshold_usd": 10000,
                "min_transactions_near_threshold": 5,
                "clustering_score_threshold": 0.7,
                "size_consistency_threshold": 0.8,
                "near_threshold_lower_pct": 0.80,
                "near_threshold_upper_pct": 0.99,
            },
        }

    def _create_default_detectors(self) -> List[BasePatternDetector]:
        return [
            CycleDetector(
                config=self.config, 
                address_labels_cache={},
                network=self.network
            ),
            LayeringDetector(
                config=self.config,
                address_labels_cache={},
                network=self.network
            ),
            NetworkDetector(
                config=self.config,
                address_labels_cache={},
                network=self.network
            ),
            ProximityDetector(
                config=self.config,
                address_labels_cache={},
                network=self.network
            ),
            MotifDetector(
                config=self.config,
                address_labels_cache={},
                network=self.network
            ),
            BurstDetector(
                config=self.config,
                address_labels_cache={},
                network=self.network
            ),
            ThresholdDetector(
                config=self.config,
                address_labels_cache={},
                network=self.network
            ),
        ]

    def analyze(
        self,
        money_flows: List[Dict[str, Any]],
        address_labels: Dict[str, Dict[str, Any]],
        window_days: int,
        processing_date: str
    ) -> List[Dict[str, Any]]:
        if not money_flows:
            raise ValueError("No money flows provided for pattern analysis")
        
        logger.info(f"Building graph from {len(money_flows)} money flows")
        G = build_money_flow_graph(money_flows)
        
        if G.number_of_nodes() == 0:
            raise ValueError("Empty graph built from money flows")
        
        logger.info(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        return self.analyze_graph(G, address_labels, window_days, processing_date)

    def analyze_graph(
        self,
        graph: nx.DiGraph,
        address_labels: Dict[str, Dict[str, Any]],
        window_days: int,
        processing_date: str
    ) -> List[Dict[str, Any]]:
        all_patterns = []
        
        if graph.number_of_nodes() == 0:
            raise ValueError("Empty graph, no patterns to detect")
        
        logger.info(
            f"Starting pattern analysis on graph with "
            f"{graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
        )
        
        for detector in self.detectors:
            detector_name = detector.__class__.__name__
            
            try:
                logger.debug(f"Running {detector_name}")
                
                patterns = detector.detect(
                    G=graph,
                    address_labels=address_labels,
                    window_days=window_days,
                    processing_date=processing_date
                )
                
                logger.info(f"{detector_name}: detected {len(patterns)} patterns")
                all_patterns.extend(patterns)
                
            except Exception as e:
                logger.error(f"{detector_name} failed: {e}")
                raise
        
        logger.info(f"Total patterns detected: {len(all_patterns)}")
        return all_patterns

    def analyze_with_config(
        self,
        graph: nx.DiGraph,
        address_labels: Dict[str, Dict[str, Any]],
        config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        window_days = config.get('window_days', 30)
        processing_date = config.get('processing_date', '')
        
        return self.analyze_graph(
            graph=graph,
            address_labels=address_labels,
            window_days=window_days,
            processing_date=processing_date
        )

    def get_detector_names(self) -> List[str]:
        return [d.__class__.__name__ for d in self.detectors]

    def add_detector(self, detector: BasePatternDetector) -> None:
        self.detectors.append(detector)
        logger.info(f"Added detector: {detector.__class__.__name__}")

    def remove_detector(self, detector_name: str) -> bool:
        for i, detector in enumerate(self.detectors):
            if detector.__class__.__name__ == detector_name:
                self.detectors.pop(i)
                logger.info(f"Removed detector: {detector_name}")
                return True
        return False