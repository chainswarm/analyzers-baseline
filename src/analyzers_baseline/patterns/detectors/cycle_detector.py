from typing import Dict, List, Any

import networkx as nx
import numpy as np
from loguru import logger

from analyzers_baseline.patterns.base_detector import (
    BasePatternDetector,
    PatternType,
    DetectionMethod,
    Severity,
    generate_pattern_hash,
    generate_pattern_id,
)


class CycleDetector(BasePatternDetector):

    @property
    def pattern_type(self) -> str:
        return PatternType.CYCLE

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
        
        min_cycle_length = self._get_config_value('cycle_detection', 'min_cycle_length', 3)
        max_cycle_length = self._get_config_value('cycle_detection', 'max_cycle_length', 10)
        max_cycles_per_scc = self._get_config_value('cycle_detection', 'max_cycles_per_scc', 100)
        
        sccs = list(nx.strongly_connected_components(G))
        
        for scc in sccs:
            if len(scc) < min_cycle_length:
                continue
                
            scc_graph = G.subgraph(scc).copy()
            cycles_found = 0
            
            for cycle in nx.simple_cycles(scc_graph):
                if cycles_found >= max_cycles_per_scc:
                    break
                
                cycle_length = len(cycle)
                if cycle_length < min_cycle_length or cycle_length > max_cycle_length:
                    continue
                
                sorted_cycle = sorted(cycle)
                pattern_hash = generate_pattern_hash(PatternType.CYCLE, sorted_cycle)
                
                if pattern_hash in patterns_by_hash:
                    continue
                
                cycle_volume, avg_edge_amount, tx_count = self._calculate_cycle_metrics(G, cycle)
                severity = self._determine_cycle_severity(cycle, cycle_volume)
                confidence = self._calculate_cycle_confidence(cycle_length, cycle_volume, avg_edge_amount)
                address_roles = {addr: 'participant' for addr in cycle}
                is_wash_trading = self._check_wash_trading(G, cycle, avg_edge_amount)
                
                pattern_id = generate_pattern_id(PatternType.CYCLE, pattern_hash)
                
                pattern = {
                    'pattern_id': pattern_id,
                    'pattern_type': PatternType.CYCLE,
                    'pattern_hash': pattern_hash,
                    'addresses_involved': sorted_cycle,
                    'address_roles': address_roles,
                    'transaction_ids': [],
                    'total_amount_usd': cycle_volume,
                    'detection_method': DetectionMethod.CYCLE_DETECTION,
                    'confidence_score': confidence,
                    'severity': severity,
                    'evidence': {
                        'cycle_length': cycle_length,
                        'cycle_addresses': cycle,
                        'avg_edge_amount': avg_edge_amount,
                        'is_wash_trading': is_wash_trading,
                    },
                    'window_days': window_days,
                    'processing_date': processing_date,
                    'network': self.network or '',
                }
                
                patterns_by_hash[pattern_hash] = pattern
                cycles_found += 1
                
        logger.info(f"Detected {len(patterns_by_hash)} unique cycle patterns")
        return list(patterns_by_hash.values())

    def _calculate_cycle_metrics(
        self,
        G: nx.DiGraph,
        cycle: List[str]
    ) -> tuple:
        total_volume = 0.0
        total_tx_count = 0
        edge_count = 0
        
        for i in range(len(cycle)):
            from_addr = cycle[i]
            to_addr = cycle[(i + 1) % len(cycle)]
            
            if G.has_edge(from_addr, to_addr):
                edge_data = G[from_addr][to_addr]
                total_volume += edge_data.get('amount_usd_sum', 0)
                total_tx_count += edge_data.get('tx_count', 1)
                edge_count += 1
        
        avg_edge_amount = total_volume / edge_count if edge_count > 0 else 0.0
        
        return total_volume, avg_edge_amount, total_tx_count

    def _determine_cycle_severity(
        self,
        cycle: List[str],
        cycle_volume: float
    ) -> str:
        has_fraudulent = any(self._is_fraudulent_address(addr) for addr in cycle)
        
        if has_fraudulent:
            return Severity.CRITICAL
        
        if cycle_volume > 100000 or len(cycle) > 5:
            return Severity.HIGH
        
        if cycle_volume > 10000:
            return Severity.MEDIUM
        
        return Severity.LOW

    def _calculate_cycle_confidence(
        self,
        cycle_length: int,
        cycle_volume: float,
        avg_edge_amount: float
    ) -> float:
        confidence = 0.5
        
        if cycle_length >= 4:
            confidence += 0.1
        if cycle_length >= 6:
            confidence += 0.1
            
        if cycle_volume > 10000:
            confidence += 0.1
        if cycle_volume > 100000:
            confidence += 0.1
            
        if avg_edge_amount > 1000:
            confidence += 0.1
            
        return min(confidence, 1.0)

    def _check_wash_trading(
        self,
        G: nx.DiGraph,
        cycle: List[str],
        avg_edge_amount: float
    ) -> bool:
        if len(cycle) > 3:
            return False
            
        amounts = []
        for i in range(len(cycle)):
            from_addr = cycle[i]
            to_addr = cycle[(i + 1) % len(cycle)]
            if G.has_edge(from_addr, to_addr):
                amounts.append(G[from_addr][to_addr].get('amount_usd_sum', 0))
        
        if not amounts:
            return False
            
        if len(amounts) > 1:
            mean_amt = np.mean(amounts)
            std_amt = np.std(amounts)
            cv = std_amt / max(mean_amt, 1.0)
            
            return cv < 0.2
            
        return False