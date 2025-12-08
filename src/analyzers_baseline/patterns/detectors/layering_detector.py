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


class LayeringDetector(BasePatternDetector):

    @property
    def pattern_type(self) -> str:
        return PatternType.LAYERING_PATH

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
        
        min_path_length = self._get_config_value('path_analysis', 'min_path_length', 3)
        max_path_length = self._get_config_value('path_analysis', 'max_path_length', 10)
        max_paths_to_check = self._get_config_value('path_analysis', 'max_paths_to_check', 1000)
        high_volume_percentile = self._get_config_value('path_analysis', 'high_volume_percentile', 90)
        max_source_nodes = self._get_config_value('path_analysis', 'max_source_nodes', 50)
        max_target_nodes = self._get_config_value('path_analysis', 'max_target_nodes', 50)
        layering_min_volume = self._get_config_value('path_analysis', 'layering_min_volume', 1000)
        layering_cv_threshold = self._get_config_value('path_analysis', 'layering_cv_threshold', 0.5)
        
        node_volumes = {}
        for node in G.nodes():
            in_volume = sum(
                data.get('amount_usd_sum', 0) 
                for _, _, data in G.in_edges(node, data=True)
            )
            out_volume = sum(
                data.get('amount_usd_sum', 0) 
                for _, _, data in G.out_edges(node, data=True)
            )
            node_volumes[node] = in_volume + out_volume
        
        if not node_volumes:
            return []
            
        volume_threshold = np.percentile(
            list(node_volumes.values()), 
            high_volume_percentile
        )
        high_volume_nodes = [
            node for node, vol in node_volumes.items() 
            if vol >= volume_threshold
        ]
        
        if len(high_volume_nodes) < 2:
            return []
        
        paths_checked = 0
        
        for source in high_volume_nodes[:max_source_nodes]:
            if paths_checked >= max_paths_to_check:
                break
                
            for target in high_volume_nodes[:max_target_nodes]:
                if source == target or paths_checked >= max_paths_to_check:
                    continue
                    
                try:
                    paths = nx.all_simple_paths(
                        G, source, target, cutoff=max_path_length
                    )
                    
                    for path in paths:
                        paths_checked += 1
                        if paths_checked >= max_paths_to_check:
                            break
                            
                        if len(path) < min_path_length:
                            continue
                            
                        path_volume = self._calculate_path_volume(G, path)
                        
                        if self._is_layering_pattern(
                            G, path, path_volume, 
                            layering_min_volume, layering_cv_threshold
                        ):
                            sorted_path = sorted(path)
                            pattern_hash = generate_pattern_hash(
                                PatternType.LAYERING_PATH, sorted_path
                            )
                            
                            if pattern_hash in patterns_by_hash:
                                continue
                            
                            unique_assets = self._get_path_assets(G, path)
                            layering_score = self._calculate_layering_score(G, path, path_volume)
                            severity = self._determine_layering_severity(path, path_volume)
                            
                            address_roles = {path[0]: 'source'}
                            for addr in path[1:-1]:
                                address_roles[addr] = 'intermediary'
                            address_roles[path[-1]] = 'destination'
                            
                            pattern_id = generate_pattern_id(
                                PatternType.LAYERING_PATH, pattern_hash
                            )
                            
                            pattern = {
                                'pattern_id': pattern_id,
                                'pattern_type': PatternType.LAYERING_PATH,
                                'pattern_hash': pattern_hash,
                                'addresses_involved': sorted_path,
                                'address_roles': address_roles,
                                'transaction_ids': [],
                                'total_amount_usd': path_volume,
                                'detection_method': DetectionMethod.PATH_ANALYSIS,
                                'confidence_score': layering_score,
                                'severity': severity,
                                'evidence': {
                                    'path_length': len(path),
                                    'path_addresses': path,
                                    'unique_assets': unique_assets,
                                    'layering_score': layering_score,
                                },
                                'window_days': window_days,
                                'processing_date': processing_date,
                                'network': self.network or '',
                            }
                            
                            patterns_by_hash[pattern_hash] = pattern
                                
                except nx.NetworkXNoPath:
                    continue
        
        logger.info(f"Detected {len(patterns_by_hash)} unique layering patterns")
        return list(patterns_by_hash.values())

    def _calculate_path_volume(
        self,
        G: nx.DiGraph,
        path: List[str]
    ) -> float:
        total_volume = 0.0
        for i in range(len(path) - 1):
            if G.has_edge(path[i], path[i + 1]):
                total_volume += G[path[i]][path[i + 1]].get('amount_usd_sum', 0)
        return total_volume

    def _is_layering_pattern(
        self,
        G: nx.DiGraph,
        path: List[str],
        volume: float,
        min_volume: float,
        cv_threshold: float
    ) -> bool:
        if len(path) < 3 or volume < min_volume:
            return False
        
        volumes = []
        for i in range(len(path) - 1):
            if G.has_edge(path[i], path[i + 1]):
                volumes.append(G[path[i]][path[i + 1]].get('amount_usd_sum', 0))
        
        if not volumes:
            return False
            
        mean_vol = np.mean(volumes)
        std_vol = np.std(volumes)
        cv = std_vol / max(mean_vol, 1.0)
        
        return cv < cv_threshold

    def _get_path_assets(
        self,
        G: nx.DiGraph,
        path: List[str]
    ) -> List[str]:
        assets = set()
        for i in range(len(path) - 1):
            if G.has_edge(path[i], path[i + 1]):
                edge_data = G[path[i]][path[i + 1]]
                asset = edge_data.get('asset_symbol')
                if asset:
                    assets.add(asset)
        return list(assets)

    def _calculate_layering_score(
        self,
        G: nx.DiGraph,
        path: List[str],
        path_volume: float
    ) -> float:
        score = 0.5
        
        if len(path) >= 4:
            score += 0.1
        if len(path) >= 6:
            score += 0.1
            
        if path_volume > 10000:
            score += 0.1
        if path_volume > 100000:
            score += 0.1
            
        volumes = []
        for i in range(len(path) - 1):
            if G.has_edge(path[i], path[i + 1]):
                volumes.append(G[path[i]][path[i + 1]].get('amount_usd_sum', 0))
        
        if len(volumes) > 1:
            cv = np.std(volumes) / max(np.mean(volumes), 1.0)
            if cv < 0.2:
                score += 0.1
                
        return min(score, 1.0)

    def _determine_layering_severity(
        self,
        path: List[str],
        path_volume: float
    ) -> str:
        has_fraudulent = any(self._is_fraudulent_address(addr) for addr in path)
        
        if has_fraudulent:
            return Severity.CRITICAL
        
        if path_volume > 100000 and len(path) >= 5:
            return Severity.HIGH
        
        if path_volume > 50000 or len(path) >= 6:
            return Severity.HIGH
        
        if path_volume > 10000:
            return Severity.MEDIUM
        
        return Severity.LOW