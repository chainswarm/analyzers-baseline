from typing import Dict, List, Any

import networkx as nx
from loguru import logger

from analyzers_baseline.patterns.base_detector import (
    BasePatternDetector,
    PatternType,
    DetectionMethod,
    Severity,
    generate_pattern_hash,
    generate_pattern_id,
)


class ProximityDetector(BasePatternDetector):

    @property
    def pattern_type(self) -> str:
        return PatternType.PROXIMITY_RISK

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
        
        max_distance = self._get_config_value('proximity_analysis', 'max_distance', 3)
        distance_decay_factor = self._get_config_value(
            'proximity_analysis', 'distance_decay_factor', 1.0
        )
        
        all_addresses = list(G.nodes())
        risk_addresses = self._get_fraudulent_addresses(all_addresses)
        
        if not risk_addresses:
            risk_addresses = self._identify_risk_addresses(G)
        
        if not risk_addresses:
            return []
        
        logger.debug(f"Analyzing proximity to {len(risk_addresses)} risk addresses")
        
        for risk_addr in risk_addresses:
            G_undirected = G.to_undirected()
            distances = nx.single_source_shortest_path_length(
                G_undirected, risk_addr, cutoff=max_distance
            )
            
            for address, distance in distances.items():
                if address == risk_addr or distance == 0:
                    continue
                
                pattern_hash = generate_pattern_hash(
                    PatternType.PROXIMITY_RISK, 
                    [risk_addr, address]
                )
                
                if pattern_hash in patterns_by_hash:
                    continue
                
                risk_score = distance_decay_factor / (distance + 1)
                address_volume = self._calculate_address_volume(G, address)
                flag_type = self._get_flag_type(risk_addr)
                severity = self._determine_proximity_severity(
                    distance, flag_type, address_volume
                )
                
                address_roles = {
                    risk_addr: 'risk_source',
                    address: 'suspect'
                }
                
                pattern_id = generate_pattern_id(
                    PatternType.PROXIMITY_RISK, pattern_hash
                )
                
                pattern = {
                    'pattern_id': pattern_id,
                    'pattern_type': PatternType.PROXIMITY_RISK,
                    'pattern_hash': pattern_hash,
                    'addresses_involved': sorted([risk_addr, address]),
                    'address_roles': address_roles,
                    'transaction_ids': [],
                    'total_amount_usd': address_volume,
                    'detection_method': DetectionMethod.PROXIMITY_ANALYSIS,
                    'confidence_score': self._calculate_proximity_confidence(
                        distance, address_volume
                    ),
                    'severity': severity,
                    'evidence': {
                        'hop_distance': distance,
                        'flagged_address': risk_addr,
                        'flag_type': flag_type,
                        'risk_multiplier': risk_score,
                    },
                    'window_days': window_days,
                    'processing_date': processing_date,
                    'network': self.network or '',
                }
                
                patterns_by_hash[pattern_hash] = pattern
        
        logger.info(f"Detected {len(patterns_by_hash)} proximity risk patterns")
        return list(patterns_by_hash.values())

    def _get_fraudulent_addresses(
        self,
        addresses: List[str]
    ) -> List[str]:
        return [addr for addr in addresses if self._is_fraudulent_address(addr)]

    def _identify_risk_addresses(
        self,
        G: nx.DiGraph
    ) -> List[str]:
        risk_addresses = []
        
        high_volume_threshold = self._get_config_value(
            'risk_identification', 'high_volume_threshold', 100000
        )
        high_degree_threshold = self._get_config_value(
            'risk_identification', 'high_degree_threshold', 50
        )
        
        for node in G.nodes():
            in_volume = sum(
                data.get('amount_usd_sum', 0)
                for _, _, data in G.in_edges(node, data=True)
            )
            out_volume = sum(
                data.get('amount_usd_sum', 0)
                for _, _, data in G.out_edges(node, data=True)
            )
            total_volume = in_volume + out_volume
            
            degree = G.degree(node)
            
            if total_volume > high_volume_threshold and degree > high_degree_threshold:
                risk_addresses.append(node)
        
        return risk_addresses

    def _calculate_address_volume(
        self,
        G: nx.DiGraph,
        address: str
    ) -> float:
        in_volume = sum(
            data.get('amount_usd_sum', 0)
            for _, _, data in G.in_edges(address, data=True)
        )
        out_volume = sum(
            data.get('amount_usd_sum', 0)
            for _, _, data in G.out_edges(address, data=True)
        )
        return in_volume + out_volume

    def _get_flag_type(
        self,
        address: str
    ) -> str:
        label_info = self._address_labels_cache.get(address, {})
        address_type = label_info.get('address_type', 'UNKNOWN')
        
        flag_type_map = {
            'MIXER': 'MIXER',
            'SCAM': 'SCAM',
            'DARK_MARKET': 'DARK_MARKET',
            'SANCTIONED': 'SANCTIONED',
        }
        
        return flag_type_map.get(address_type, 'HIGH_RISK')

    def _determine_proximity_severity(
        self,
        distance: int,
        flag_type: str,
        volume: float
    ) -> str:
        if flag_type == 'SANCTIONED':
            return Severity.CRITICAL
        
        if distance == 1:
            if flag_type in ['MIXER', 'DARK_MARKET', 'SCAM']:
                return Severity.CRITICAL
            return Severity.HIGH
        
        if distance == 2:
            if volume > 100000:
                return Severity.HIGH
            if flag_type in ['MIXER', 'DARK_MARKET']:
                return Severity.HIGH
            return Severity.MEDIUM
        
        if volume > 100000:
            return Severity.MEDIUM
        
        return Severity.LOW

    def _calculate_proximity_confidence(
        self,
        distance: int,
        volume: float
    ) -> float:
        confidence = 1.0 - (distance * 0.2)
        
        if volume > 100000:
            confidence += 0.1
        if volume > 1000000:
            confidence += 0.1
        
        return min(max(confidence, 0.1), 1.0)