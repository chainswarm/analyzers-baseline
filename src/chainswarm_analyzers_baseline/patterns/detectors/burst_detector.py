from typing import Dict, List, Any, Optional
from collections import defaultdict

import networkx as nx
import numpy as np
from loguru import logger

from chainswarm_analyzers_baseline.patterns.base_detector import (
    BasePatternDetector,
    PatternType,
    DetectionMethod,
    Severity,
    generate_pattern_hash,
    generate_pattern_id,
)


class BurstDetector(BasePatternDetector):

    @property
    def pattern_type(self) -> str:
        return PatternType.TEMPORAL_BURST

    def detect(
        self,
        G: nx.DiGraph,
        address_labels: Dict[str, Dict[str, Any]],
        window_days: int,
        processing_date: str,
        timestamp_data: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        self._address_labels_cache = address_labels
        
        if G.number_of_nodes() == 0:
            return []
        
        patterns_by_hash = {}
        
        min_burst_intensity = self._get_config_value(
            'burst_detection', 'min_burst_intensity', 3.0
        )
        min_burst_transactions = self._get_config_value(
            'burst_detection', 'min_burst_transactions', 10
        )
        time_window_seconds = self._get_config_value(
            'burst_detection', 'time_window_seconds', 3600
        )
        z_score_threshold = self._get_config_value(
            'burst_detection', 'z_score_threshold', 2.0
        )
        
        for node in G.nodes():
            node_timestamps = timestamp_data.get(node, [])
            if not node_timestamps:
                continue
            
            burst_pattern = self._analyze_temporal_bursts(
                node, node_timestamps, time_window_seconds, min_burst_intensity,
                min_burst_transactions, z_score_threshold
            )
            
            if burst_pattern:
                pattern_hash = generate_pattern_hash(
                    PatternType.TEMPORAL_BURST,
                    [node, str(burst_pattern['burst_start_ms'])]
                )
                
                if pattern_hash in patterns_by_hash:
                    continue
                
                pattern_id = generate_pattern_id(
                    PatternType.TEMPORAL_BURST, pattern_hash
                )
                
                addresses_involved = [node] + burst_pattern.get('counterparties', [])
                
                pattern = {
                    'pattern_id': pattern_id,
                    'pattern_type': PatternType.TEMPORAL_BURST,
                    'pattern_hash': pattern_hash,
                    'addresses_involved': sorted(set(addresses_involved)),
                    'address_roles': {node: 'burst_source'},
                    'transaction_ids': [],
                    'total_amount_usd': burst_pattern['burst_volume_usd'],
                    'detection_method': DetectionMethod.TEMPORAL_ANALYSIS,
                    'confidence_score': self._calculate_burst_confidence(
                        burst_pattern['z_score'],
                        burst_pattern['burst_intensity'],
                        burst_pattern['burst_tx_count']
                    ),
                    'severity': self._determine_burst_severity(
                        node, burst_pattern
                    ),
                    'evidence': {
                        'burst_start_ms': burst_pattern['burst_start_ms'],
                        'burst_end_ms': burst_pattern['burst_end_ms'],
                        'tx_count': burst_pattern['burst_tx_count'],
                        'volume_usd': burst_pattern['burst_volume_usd'],
                        'addresses_involved': addresses_involved,
                        'intensity_score': burst_pattern['burst_intensity'],
                    },
                    'window_days': window_days,
                    'processing_date': processing_date,
                    'network': self.network or '',
                }
                
                patterns_by_hash[pattern_hash] = pattern
        
        logger.info(f"Detected {len(patterns_by_hash)} burst patterns")
        return list(patterns_by_hash.values())

    def _analyze_temporal_bursts(
        self,
        node: str,
        timestamp_data: List[Dict[str, Any]],
        time_window: int,
        min_intensity: float,
        min_transactions: int,
        z_threshold: float
    ) -> Optional[Dict[str, Any]]:
        
        timestamps = [t['timestamp'] for t in timestamp_data]
        volumes = [t['volume'] for t in timestamp_data]
        counterparties = [t['counterparty'] for t in timestamp_data]
        
        if len(timestamps) < min_transactions:
            return None
        
        sorted_indices = np.argsort(timestamps)
        timestamps = [timestamps[i] for i in sorted_indices]
        volumes = [volumes[i] for i in sorted_indices]
        counterparties = [counterparties[i] for i in sorted_indices]
        
        burst = self._find_burst_window(
            timestamps, volumes, counterparties,
            time_window, min_transactions, min_intensity, z_threshold
        )
        
        return burst

    def _find_burst_window(
        self,
        timestamps: List[int],
        volumes: List[float],
        counterparties: List[str],
        time_window: int,
        min_transactions: int,
        min_intensity: float,
        z_threshold: float
    ) -> Optional[Dict[str, Any]]:
        if len(timestamps) < min_transactions:
            return None
        
        time_window_ms = time_window * 1000
        
        total_span_ms = timestamps[-1] - timestamps[0]
        if total_span_ms <= 0:
            return None
        
        baseline_rate = len(timestamps) / (total_span_ms / time_window_ms)
        
        best_burst = None
        best_z_score = 0
        
        for i in range(len(timestamps)):
            window_start = timestamps[i]
            window_end = window_start + time_window_ms
            
            window_indices = [
                j for j in range(i, len(timestamps))
                if timestamps[j] <= window_end
            ]
            
            if len(window_indices) < min_transactions:
                continue
            
            window_count = len(window_indices)
            window_volume = sum(volumes[j] for j in window_indices)
            window_counterparties = [counterparties[j] for j in window_indices]
            
            window_rate = window_count
            intensity = window_rate / max(baseline_rate, 0.1)
            
            if intensity < min_intensity:
                continue
            
            z_score = self._calculate_z_score(
                window_count, baseline_rate, len(timestamps)
            )
            
            if z_score < z_threshold:
                continue
            
            if z_score > best_z_score:
                best_z_score = z_score
                actual_end = max(timestamps[j] for j in window_indices)
                
                best_burst = {
                    'burst_start_ms': window_start,
                    'burst_end_ms': actual_end,
                    'burst_tx_count': window_count,
                    'burst_volume_usd': window_volume,
                    'burst_intensity': intensity,
                    'z_score': z_score,
                    'counterparties': list(set(window_counterparties)),
                    'normal_tx_rate': baseline_rate,
                    'burst_tx_rate': window_rate,
                }
        
        return best_burst

    def _calculate_z_score(
        self,
        observed: int,
        expected_rate: float,
        total_observations: int
    ) -> float:
        if expected_rate <= 0 or total_observations <= 1:
            return 0.0
        
        expected = expected_rate
        std = np.sqrt(expected) if expected > 0 else 1
        
        z_score = (observed - expected) / max(std, 0.1)
        return z_score

    def _calculate_burst_confidence(
        self,
        z_score: float,
        intensity: float,
        tx_count: int
    ) -> float:
        confidence = 0.5
        
        if z_score > 4:
            confidence += 0.2
        elif z_score > 3:
            confidence += 0.15
        elif z_score > 2:
            confidence += 0.1
        
        if intensity > 5:
            confidence += 0.15
        elif intensity > 3:
            confidence += 0.1
        
        if tx_count > 20:
            confidence += 0.1
        elif tx_count > 10:
            confidence += 0.05
        
        return min(confidence, 1.0)

    def _determine_burst_severity(
        self,
        node: str,
        burst: Dict[str, Any]
    ) -> str:
        if self._is_fraudulent_address(node):
            return Severity.CRITICAL
        
        counterparties = burst.get('counterparties', [])
        has_fraudulent = any(
            self._is_fraudulent_address(addr) 
            for addr in counterparties
        )
        if has_fraudulent:
            return Severity.HIGH
        
        volume = burst.get('burst_volume_usd', 0)
        intensity = burst.get('burst_intensity', 0)
        z_score = burst.get('z_score', 0)
        
        if volume > 100000 and intensity > 5:
            return Severity.HIGH
        
        if volume > 50000 or z_score > 4:
            return Severity.MEDIUM
        
        return Severity.LOW
