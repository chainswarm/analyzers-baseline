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


class ThresholdDetector(BasePatternDetector):

    @property
    def pattern_type(self) -> str:
        return PatternType.THRESHOLD_EVASION

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
        
        thresholds = self._get_thresholds()
        
        min_transactions = self._get_config_value(
            'threshold_detection', 'min_transactions_near_threshold', 5
        )
        clustering_threshold = self._get_config_value(
            'threshold_detection', 'clustering_score_threshold', 0.7
        )
        consistency_threshold = self._get_config_value(
            'threshold_detection', 'size_consistency_threshold', 0.8
        )
        
        for node in G.nodes():
            for threshold_type, threshold_value in thresholds.items():
                evasion_pattern = self._analyze_threshold_evasion(
                    G, node, threshold_value, threshold_type,
                    min_transactions, clustering_threshold, consistency_threshold
                )
                
                if evasion_pattern:
                    pattern_hash = generate_pattern_hash(
                        PatternType.THRESHOLD_EVASION,
                        [node, threshold_type, str(threshold_value)]
                    )
                    
                    if pattern_hash in patterns_by_hash:
                        continue
                    
                    pattern_id = generate_pattern_id(
                        PatternType.THRESHOLD_EVASION, pattern_hash
                    )
                    
                    total_structured = (
                        evasion_pattern['avg_transaction_size'] *
                        evasion_pattern['transactions_near_threshold']
                    )
                    
                    pattern = {
                        'pattern_id': pattern_id,
                        'pattern_type': PatternType.THRESHOLD_EVASION,
                        'pattern_hash': pattern_hash,
                        'addresses_involved': [node],
                        'address_roles': {node: 'primary_address'},
                        'transaction_ids': [],
                        'total_amount_usd': total_structured,
                        'detection_method': DetectionMethod.TEMPORAL_ANALYSIS,
                        'confidence_score': self._calculate_evasion_confidence(
                            evasion_pattern
                        ),
                        'severity': self._determine_evasion_severity(
                            node, evasion_pattern, threshold_type
                        ),
                        'evidence': {
                            'threshold_amount_usd': threshold_value,
                            'structured_txs': evasion_pattern['transactions_near_threshold'],
                            'total_structured_usd': total_structured,
                            'evasion_score': evasion_pattern['threshold_avoidance_score'],
                        },
                        'window_days': window_days,
                        'processing_date': processing_date,
                        'network': self.network or '',
                    }
                    
                    patterns_by_hash[pattern_hash] = pattern
        
        logger.info(f"Detected {len(patterns_by_hash)} threshold evasion patterns")
        return list(patterns_by_hash.values())

    def _get_thresholds(self) -> Dict[str, float]:
        thresholds = {}
        
        reporting_threshold = self._get_config_value(
            'threshold_detection', 'reporting_threshold_usd', 10000
        )
        if reporting_threshold:
            thresholds["reporting"] = float(reporting_threshold)
        
        custom_thresholds = self._get_config_value(
            'threshold_detection', 'custom_thresholds', {}
        )
        if custom_thresholds:
            for threshold_name, threshold_value in custom_thresholds.items():
                thresholds[threshold_name] = float(threshold_value)
        
        if not thresholds:
            thresholds = {
                "reporting_10k": 10000.0,
                "reporting_15k": 15000.0,
            }
        
        return thresholds

    def _analyze_threshold_evasion(
        self,
        G: nx.DiGraph,
        node: str,
        threshold: float,
        threshold_type: str,
        min_transactions: int,
        clustering_threshold: float,
        consistency_threshold: float
    ) -> Dict[str, Any]:
        transaction_amounts = []
        
        for _, target, data in G.out_edges(node, data=True):
            amount = data.get('amount_usd_sum', 0)
            tx_count = data.get('tx_count', 1)
            
            if tx_count > 1:
                avg_amount = amount / tx_count
                transaction_amounts.extend([avg_amount] * tx_count)
            else:
                transaction_amounts.append(amount)
        
        if len(transaction_amounts) < min_transactions:
            return None
        
        near_threshold_lower_pct = self._get_config_value(
            'threshold_detection', 'near_threshold_lower_pct', 0.80
        )
        near_threshold_upper_pct = self._get_config_value(
            'threshold_detection', 'near_threshold_upper_pct', 0.99
        )
        
        near_threshold_lower = threshold * near_threshold_lower_pct
        near_threshold_upper = threshold * near_threshold_upper_pct
        
        near_threshold_txs = [
            amt for amt in transaction_amounts
            if near_threshold_lower <= amt <= near_threshold_upper
        ]
        
        if len(near_threshold_txs) < min_transactions:
            return None
        
        clustering_score = len(near_threshold_txs) / len(transaction_amounts)
        
        if clustering_score < clustering_threshold:
            return None
        
        near_amounts = np.array(near_threshold_txs)
        
        if len(near_amounts) > 1:
            cv = np.std(near_amounts) / max(np.mean(near_amounts), 1.0)
            size_consistency = max(0, 1.0 - cv)
        else:
            size_consistency = 1.0
        
        if size_consistency < consistency_threshold:
            return None
        
        return {
            'transactions_near_threshold': len(near_threshold_txs),
            'avg_transaction_size': float(np.mean(near_amounts)),
            'max_transaction_size': float(np.max(near_amounts)),
            'size_consistency': size_consistency,
            'clustering_score': clustering_score,
            'threshold_avoidance_score': clustering_score * size_consistency,
        }

    def _calculate_evasion_confidence(
        self,
        evasion: Dict[str, Any]
    ) -> float:
        confidence = 0.5
        
        clustering = evasion.get('clustering_score', 0)
        if clustering > 0.8:
            confidence += 0.2
        elif clustering > 0.7:
            confidence += 0.1
        
        consistency = evasion.get('size_consistency', 0)
        if consistency > 0.9:
            confidence += 0.15
        elif consistency > 0.8:
            confidence += 0.1
        
        tx_count = evasion.get('transactions_near_threshold', 0)
        if tx_count > 20:
            confidence += 0.1
        elif tx_count > 10:
            confidence += 0.05
        
        return min(confidence, 1.0)

    def _determine_evasion_severity(
        self,
        node: str,
        evasion: Dict[str, Any],
        threshold_type: str
    ) -> str:
        if self._is_fraudulent_address(node):
            return Severity.CRITICAL
        
        avoidance_score = evasion.get('threshold_avoidance_score', 0)
        tx_count = evasion.get('transactions_near_threshold', 0)
        
        if avoidance_score > 0.85 and tx_count > 20:
            return Severity.CRITICAL
        
        if threshold_type in ['reporting', 'reporting_10k', 'CTR']:
            if avoidance_score > 0.7:
                return Severity.HIGH
        
        if avoidance_score > 0.8 or tx_count > 15:
            return Severity.HIGH
        
        if avoidance_score > 0.7 or tx_count > 10:
            return Severity.MEDIUM
        
        return Severity.LOW