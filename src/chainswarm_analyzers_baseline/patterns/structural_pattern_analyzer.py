from typing import Dict, List, Any, Optional

import networkx as nx
from loguru import logger

from chainswarm_analyzers_baseline.config import SettingsLoader
from chainswarm_analyzers_baseline.patterns.base_detector import BasePatternDetector
from chainswarm_analyzers_baseline.patterns.detectors import (
    CycleDetector,
    LayeringDetector,
    NetworkDetector,
    ProximityDetector,
    MotifDetector,
    BurstDetector,
    ThresholdDetector,
)
from chainswarm_analyzers_baseline.graph.builder import build_money_flow_graph


class StructuralPatternAnalyzer:

    def __init__(
        self,
        settings_loader: SettingsLoader,
        network: str,
        detectors: Optional[List[BasePatternDetector]] = None,
    ):
        self.network = network
        self.config = settings_loader.load(network)
        
        if detectors is None:
            self.detectors = self._create_default_detectors()
        else:
            self.detectors = detectors
        
        logger.info(
            f"Initialized StructuralPatternAnalyzer with {len(self.detectors)} detectors for {network}"
        )

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
        timestamp_data: Dict[str, List[Dict[str, Any]]],
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
        
        return self.analyze_graph(G, address_labels, timestamp_data, window_days, processing_date)

    def analyze_graph(
        self,
        graph: nx.DiGraph,
        address_labels: Dict[str, Dict[str, Any]],
        timestamp_data: Dict[str, List[Dict[str, Any]]],
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
                logger.info(f"Running {detector_name}")
                
                if isinstance(detector, BurstDetector):
                    patterns = detector.detect(graph, address_labels, window_days, processing_date, timestamp_data)
                else:
                    patterns = detector.detect(graph, address_labels, window_days, processing_date)
                
                if patterns:
                    all_patterns.extend(patterns)
                    logger.info(f"{detector_name}: {len(patterns)} patterns")
                
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
