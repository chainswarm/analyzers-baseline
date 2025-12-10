from typing import Dict, List, Any

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


class MotifDetector(BasePatternDetector):

    @property
    def pattern_type(self) -> str:
        return PatternType.MOTIF_FANIN

    def detect(
        self,
        G: nx.DiGraph,
        address_labels: Dict[str, Dict[str, Any]],
        window_days: int,
        processing_date: str
    ) -> List[Dict[str, Any]]:
        self._address_labels_cache = address_labels
        
        if G.number_of_nodes() == 0:
            return []
        
        patterns_by_hash = {}
        
        degree_percentile = self._get_config_value(
            'motif_detection', 'degree_percentile_threshold', 90
        )
        fanin_max_out_degree = self._get_config_value(
            'motif_detection', 'fanin_max_out_degree', 3
        )
        fanout_max_in_degree = self._get_config_value(
            'motif_detection', 'fanout_max_in_degree', 3
        )
        min_spoke_count = self._get_config_value(
            'motif_detection', 'min_spoke_count', 5
        )
        
        in_degrees = [G.in_degree(node) for node in G.nodes()]
        out_degrees = [G.out_degree(node) for node in G.nodes()]
        
        if not in_degrees or not out_degrees:
            return []
        
        in_degree_threshold = np.percentile(in_degrees, degree_percentile)
        out_degree_threshold = np.percentile(out_degrees, degree_percentile)
        
        in_degree_threshold = max(in_degree_threshold, min_spoke_count)
        out_degree_threshold = max(out_degree_threshold, min_spoke_count)
        
        for node in G.nodes():
            in_deg = G.in_degree(node)
            out_deg = G.out_degree(node)
            
            if in_deg >= in_degree_threshold and out_deg <= fanin_max_out_degree:
                fanin_pattern = self._create_fanin_pattern(
                    G, node, in_deg, out_deg, window_days, processing_date
                )
                if fanin_pattern:
                    pattern_hash = fanin_pattern['pattern_hash']
                    if pattern_hash not in patterns_by_hash:
                        patterns_by_hash[pattern_hash] = fanin_pattern
            
            if out_deg >= out_degree_threshold and in_deg <= fanout_max_in_degree:
                fanout_pattern = self._create_fanout_pattern(
                    G, node, in_deg, out_deg, window_days, processing_date
                )
                if fanout_pattern:
                    pattern_hash = fanout_pattern['pattern_hash']
                    if pattern_hash not in patterns_by_hash:
                        patterns_by_hash[pattern_hash] = fanout_pattern
        
        logger.info(f"Detected {len(patterns_by_hash)} motif patterns")
        return list(patterns_by_hash.values())

    def _create_fanin_pattern(
        self,
        G: nx.DiGraph,
        center: str,
        in_deg: int,
        out_deg: int,
        window_days: int,
        processing_date: str
    ) -> Dict[str, Any]:
        in_neighbors = list(G.predecessors(center))
        all_addresses = [center] + in_neighbors
        
        pattern_hash = generate_pattern_hash(
            PatternType.MOTIF_FANIN, sorted(all_addresses)
        )
        pattern_id = generate_pattern_id(PatternType.MOTIF_FANIN, pattern_hash)
        
        fanin_volume = sum(
            data.get('amount_usd_sum', 0)
            for _, _, data in G.in_edges(center, data=True)
        )
        
        time_concentration = self._calculate_time_concentration(G, center, 'in')
        
        address_roles = {center: 'center'}
        for addr in in_neighbors:
            address_roles[addr] = 'source'
        
        return {
            'pattern_id': pattern_id,
            'pattern_type': PatternType.MOTIF_FANIN,
            'pattern_hash': pattern_hash,
            'addresses_involved': sorted(all_addresses),
            'address_roles': address_roles,
            'transaction_ids': [],
            'total_amount_usd': fanin_volume,
            'detection_method': DetectionMethod.MOTIF_DETECTION,
            'confidence_score': self._calculate_motif_confidence(
                in_deg, fanin_volume
            ),
            'severity': self._determine_motif_severity(
                all_addresses, fanin_volume, in_deg
            ),
            'evidence': {
                'center_address': center,
                'spoke_count': in_deg,
                'total_volume_usd': fanin_volume,
                'time_concentration': time_concentration,
            },
            'window_days': window_days,
            'processing_date': processing_date,
            'network': self.network or '',
        }

    def _create_fanout_pattern(
        self,
        G: nx.DiGraph,
        center: str,
        in_deg: int,
        out_deg: int,
        window_days: int,
        processing_date: str
    ) -> Dict[str, Any]:
        out_neighbors = list(G.successors(center))
        all_addresses = [center] + out_neighbors
        
        pattern_hash = generate_pattern_hash(
            PatternType.MOTIF_FANOUT, sorted(all_addresses)
        )
        pattern_id = generate_pattern_id(PatternType.MOTIF_FANOUT, pattern_hash)
        
        fanout_volume = sum(
            data.get('amount_usd_sum', 0)
            for _, _, data in G.out_edges(center, data=True)
        )
        
        time_concentration = self._calculate_time_concentration(G, center, 'out')
        
        address_roles = {center: 'center'}
        for addr in out_neighbors:
            address_roles[addr] = 'destination'
        
        return {
            'pattern_id': pattern_id,
            'pattern_type': PatternType.MOTIF_FANOUT,
            'pattern_hash': pattern_hash,
            'addresses_involved': sorted(all_addresses),
            'address_roles': address_roles,
            'transaction_ids': [],
            'total_amount_usd': fanout_volume,
            'detection_method': DetectionMethod.MOTIF_DETECTION,
            'confidence_score': self._calculate_motif_confidence(
                out_deg, fanout_volume
            ),
            'severity': self._determine_motif_severity(
                all_addresses, fanout_volume, out_deg
            ),
            'evidence': {
                'center_address': center,
                'spoke_count': out_deg,
                'total_volume_usd': fanout_volume,
                'time_concentration': time_concentration,
            },
            'window_days': window_days,
            'processing_date': processing_date,
            'network': self.network or '',
        }

    def _calculate_time_concentration(
        self,
        G: nx.DiGraph,
        center: str,
        direction: str
    ) -> float:
        if direction == 'in':
            edges = G.in_edges(center, data=True)
        else:
            edges = G.out_edges(center, data=True)
        
        timestamps = []
        for edge in edges:
            edge_data = edge[2] if len(edge) > 2 else {}
            ts = edge_data.get('timestamp') or edge_data.get('block_timestamp')
            if ts:
                timestamps.append(ts)
        
        if len(timestamps) < 2:
            return 0.5
        
        timestamps.sort()
        gaps = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
        
        if not gaps:
            return 0.5
        
        mean_gap = np.mean(gaps)
        std_gap = np.std(gaps)
        
        if mean_gap == 0:
            return 1.0  # All transactions at same time = maximum concentration
        
        # Protect against floating point edge cases
        cv = std_gap / max(mean_gap, 0.001)
        concentration = 1.0 / (1.0 + cv)
        
        return min(concentration, 1.0)

    def _calculate_motif_confidence(
        self,
        spoke_count: int,
        volume: float
    ) -> float:
        confidence = 0.5
        
        if spoke_count >= 10:
            confidence += 0.2
        elif spoke_count >= 5:
            confidence += 0.1
        
        if volume > 100000:
            confidence += 0.2
        elif volume > 10000:
            confidence += 0.1
        
        return min(confidence, 1.0)

    def _determine_motif_severity(
        self,
        addresses: List[str],
        volume: float,
        spoke_count: int
    ) -> str:
        center = addresses[0] if addresses else None
        if center and self._is_fraudulent_address(center):
            return Severity.CRITICAL
        
        has_fraudulent = any(
            self._is_fraudulent_address(addr) 
            for addr in addresses[1:]
        )
        if has_fraudulent:
            return Severity.HIGH
        
        if volume > 100000 and spoke_count >= 20:
            return Severity.HIGH
        
        if volume > 50000 or spoke_count >= 30:
            return Severity.MEDIUM
        
        return Severity.LOW
