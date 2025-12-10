from typing import Dict, List, Any, Set

import networkx as nx
from loguru import logger

from chainswarm_analyzers_baseline.patterns.base_detector import (
    BasePatternDetector,
    PatternType,
    DetectionMethod,
    Severity,
    generate_pattern_hash,
    generate_pattern_id,
)


class NetworkDetector(BasePatternDetector):

    @property
    def pattern_type(self) -> str:
        return PatternType.SMURFING_NETWORK

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
        
        patterns = []
        
        scc_patterns = self._analyze_scc(G, window_days, processing_date)
        patterns.extend(scc_patterns)
        
        smurfing_patterns = self._detect_smurfing(G, window_days, processing_date)
        patterns.extend(smurfing_patterns)
        
        sybil_patterns = self._detect_sybil_networks(G, window_days, processing_date)
        patterns.extend(sybil_patterns)
        
        logger.info(f"Detected {len(patterns)} network patterns")
        return patterns

    def _analyze_scc(
        self,
        G: nx.DiGraph,
        window_days: int,
        processing_date: str
    ) -> List[Dict[str, Any]]:
        patterns_by_hash = {}
        sccs = list(nx.strongly_connected_components(G))
        
        min_scc_size = self._get_config_value('scc_analysis', 'min_scc_size', 5)
        
        scc_sizes = [len(scc) for scc in sccs]
        if not scc_sizes:
            return []
        
        for scc in sccs:
            scc_size = len(scc)
            
            if scc_size < min_scc_size:
                continue
                
            sorted_scc = sorted(list(scc))
            pattern_hash = generate_pattern_hash(PatternType.SMURFING_NETWORK, sorted_scc)
            
            if pattern_hash in patterns_by_hash:
                continue
                
            scc_graph = G.subgraph(scc)
            total_volume = sum(
                data.get('amount_usd_sum', 0) 
                for _, _, data in scc_graph.edges(data=True)
            )
            density = nx.density(scc_graph)
            
            coordination_score = self._calculate_coordination_score(scc_graph)
            hub_addresses = self._identify_hubs_in_network(scc_graph)
            
            address_roles = {
                addr: 'hub' if addr in hub_addresses else 'participant'
                for addr in sorted_scc
            }
            
            pattern_id = generate_pattern_id('scc', pattern_hash)
            
            pattern = {
                'pattern_id': pattern_id,
                'pattern_type': PatternType.SMURFING_NETWORK,
                'pattern_hash': pattern_hash,
                'addresses_involved': sorted_scc,
                'address_roles': address_roles,
                'transaction_ids': [],
                'total_amount_usd': total_volume,
                'detection_method': DetectionMethod.SCC_ANALYSIS,
                'confidence_score': min(density + 0.3, 1.0),
                'severity': self._determine_network_severity(
                    sorted_scc, total_volume, density
                ),
                'evidence': {
                    'network_size': scc_size,
                    'hub_addresses': hub_addresses,
                    'spoke_count': scc_size - len(hub_addresses),
                    'coordination_score': coordination_score,
                },
                'window_days': window_days,
                'processing_date': processing_date,
                'network': self.network or '',
            }
            
            patterns_by_hash[pattern_hash] = pattern
                
        return list(patterns_by_hash.values())

    def _detect_smurfing(
        self,
        G: nx.DiGraph,
        window_days: int,
        processing_date: str
    ) -> List[Dict[str, Any]]:
        patterns_by_hash = {}
        
        min_community_size = self._get_config_value('network_analysis', 'min_community_size', 5)
        max_community_size = self._get_config_value('network_analysis', 'max_community_size', 100)
        small_tx_threshold = self._get_config_value('network_analysis', 'small_transaction_threshold', 1000)
        small_tx_ratio_threshold = self._get_config_value('network_analysis', 'small_transaction_ratio_threshold', 0.7)
        density_threshold = self._get_config_value('network_analysis', 'density_threshold', 0.1)
        
        G_undirected = G.to_undirected()
        
        # Check if graph has meaningful weights for community detection
        # NetworkX greedy_modularity_communities requires sum of weights > 0
        total_weight = sum(
            data.get('weight', 0) for _, _, data in G_undirected.edges(data=True)
        )
        
        if total_weight == 0:
            # Fallback: Use tx_count as weight for community detection
            # This handles cases where USD prices are not available
            for u, v, data in G_undirected.edges(data=True):
                data['weight'] = max(data.get('tx_count', 1), 1)
            
            total_weight = sum(
                data.get('weight', 0) for _, _, data in G_undirected.edges(data=True)
            )
            
            if total_weight == 0:
                # No edges or all zero tx_counts - skip community detection
                logger.warning("Skipping smurfing detection: graph has no meaningful weights")
                return []
            
            logger.info("Using tx_count as weight for community detection (USD values unavailable)")
        
        communities = list(nx.community.greedy_modularity_communities(
            G_undirected, weight='weight'
        ))
        
        for community in communities:
            community_size = len(community)
            
            if community_size < min_community_size or community_size > max_community_size:
                continue
                
            community_graph = G.subgraph(community)
            
            if self._is_smurfing_network(
                community_graph, small_tx_threshold, 
                small_tx_ratio_threshold, density_threshold
            ):
                sorted_community = sorted(list(community))
                pattern_hash = generate_pattern_hash(
                    PatternType.SMURFING_NETWORK, sorted_community
                )
                
                if pattern_hash in patterns_by_hash:
                    continue
                
                density = nx.density(community_graph)
                total_volume = sum(
                    data.get('amount_usd_sum', 0) 
                    for _, _, data in community_graph.edges(data=True)
                )
                
                hub_addresses = self._identify_hubs_in_network(community_graph)
                coordination_score = self._calculate_coordination_score(community_graph)
                
                address_roles = {
                    addr: 'hub' if addr in hub_addresses else 'participant'
                    for addr in sorted_community
                }
                
                pattern_id = generate_pattern_id(
                    PatternType.SMURFING_NETWORK, pattern_hash
                )
                
                pattern = {
                    'pattern_id': pattern_id,
                    'pattern_type': PatternType.SMURFING_NETWORK,
                    'pattern_hash': pattern_hash,
                    'addresses_involved': sorted_community,
                    'address_roles': address_roles,
                    'transaction_ids': [],
                    'total_amount_usd': total_volume,
                    'detection_method': DetectionMethod.NETWORK_ANALYSIS,
                    'confidence_score': self._calculate_smurfing_confidence(
                        community_graph, small_tx_threshold
                    ),
                    'severity': self._determine_network_severity(
                        sorted_community, total_volume, density
                    ),
                    'evidence': {
                        'network_size': community_size,
                        'hub_addresses': hub_addresses,
                        'spoke_count': community_size - len(hub_addresses),
                        'coordination_score': coordination_score,
                    },
                    'window_days': window_days,
                    'processing_date': processing_date,
                    'network': self.network or '',
                }
                
                patterns_by_hash[pattern_hash] = pattern
                    
        return list(patterns_by_hash.values())

    def _detect_sybil_networks(
        self,
        G: nx.DiGraph,
        window_days: int,
        processing_date: str
    ) -> List[Dict[str, Any]]:
        patterns_by_hash = {}
        
        min_network_size = self._get_config_value('sybil_detection', 'min_network_size', 10)
        similarity_threshold = self._get_config_value('sybil_detection', 'similarity_threshold', 0.8)
        
        outgoing_targets: Dict[str, Set[str]] = {}
        for node in G.nodes():
            successors = set(G.successors(node))
            if len(successors) >= 1:
                outgoing_targets[node] = successors
        
        processed: Set[str] = set()
        
        for addr1, targets1 in outgoing_targets.items():
            if addr1 in processed:
                continue
                
            cluster = {addr1}
            
            for addr2, targets2 in outgoing_targets.items():
                if addr2 == addr1 or addr2 in processed:
                    continue
                    
                if len(targets1 | targets2) > 0:
                    similarity = len(targets1 & targets2) / len(targets1 | targets2)
                    if similarity >= similarity_threshold:
                        cluster.add(addr2)
            
            if len(cluster) >= min_network_size:
                sorted_cluster = sorted(list(cluster))
                pattern_hash = generate_pattern_hash(
                    PatternType.SYBIL_NETWORK, sorted_cluster
                )
                
                if pattern_hash not in patterns_by_hash:
                    cluster_graph = G.subgraph(cluster)
                    total_volume = sum(
                        data.get('amount_usd_sum', 0)
                        for _, _, data in cluster_graph.edges(data=True)
                    )
                    
                    hub_addresses = self._identify_hubs_in_network(cluster_graph)
                    
                    address_roles = {
                        addr: 'hub' if addr in hub_addresses else 'sybil'
                        for addr in sorted_cluster
                    }
                    
                    pattern_id = generate_pattern_id(
                        PatternType.SYBIL_NETWORK, pattern_hash
                    )
                    
                    pattern = {
                        'pattern_id': pattern_id,
                        'pattern_type': PatternType.SYBIL_NETWORK,
                        'pattern_hash': pattern_hash,
                        'addresses_involved': sorted_cluster,
                        'address_roles': address_roles,
                        'transaction_ids': [],
                        'total_amount_usd': total_volume,
                        'detection_method': DetectionMethod.NETWORK_ANALYSIS,
                        'confidence_score': 0.7,
                        'severity': Severity.HIGH,
                        'evidence': {
                            'network_size': len(cluster),
                            'hub_addresses': hub_addresses,
                            'spoke_count': len(cluster) - len(hub_addresses),
                            'coordination_score': 0.9,
                        },
                        'window_days': window_days,
                        'processing_date': processing_date,
                        'network': self.network or '',
                    }
                    
                    patterns_by_hash[pattern_hash] = pattern
                
                processed.update(cluster)
        
        return list(patterns_by_hash.values())

    def _is_smurfing_network(
        self,
        community_graph: nx.DiGraph,
        small_tx_threshold: float,
        small_tx_ratio_threshold: float,
        density_threshold: float
    ) -> bool:
        if community_graph.number_of_nodes() < 5:
            return False
            
        volumes = [
            data.get('amount_usd_sum', 0)
            for _, _, data in community_graph.edges(data=True)
        ]
        
        if not volumes:
            return False
            
        small_tx_ratio = sum(1 for v in volumes if v < small_tx_threshold) / len(volumes)
        density = nx.density(community_graph)
        
        return small_tx_ratio > small_tx_ratio_threshold and density > density_threshold

    def _identify_hubs_in_network(
        self,
        community_graph: nx.DiGraph
    ) -> List[str]:
        if community_graph.number_of_nodes() < 3:
            return []
            
        degrees = [
            (node, community_graph.degree(node)) 
            for node in community_graph.nodes()
        ]
        degrees.sort(key=lambda x: x[1], reverse=True)
        
        num_hubs = max(1, len(degrees) // 5)
        return [node for node, _ in degrees[:num_hubs]]

    def _calculate_coordination_score(
        self,
        network_graph: nx.DiGraph
    ) -> float:
        if network_graph.number_of_nodes() < 2:
            return 0.0
            
        try:
            undirected = network_graph.to_undirected()
            avg_clustering = nx.average_clustering(undirected)
            return avg_clustering
        except Exception:
            return 0.0

    def _calculate_smurfing_confidence(
        self,
        community_graph: nx.DiGraph,
        small_tx_threshold: float
    ) -> float:
        volumes = [
            data.get('amount_usd_sum', 0)
            for _, _, data in community_graph.edges(data=True)
        ]
        
        if not volumes:
            return 0.0
            
        small_tx_ratio = sum(1 for v in volumes if v < small_tx_threshold) / len(volumes)
        density = nx.density(community_graph)
        
        return min((small_tx_ratio * 0.6 + density * 0.4), 1.0)

    def _determine_network_severity(
        self,
        addresses: List[str],
        total_volume: float,
        density: float
    ) -> str:
        has_fraudulent = any(self._is_fraudulent_address(addr) for addr in addresses)
        
        if has_fraudulent:
            return Severity.CRITICAL
        
        if total_volume > 100000 and len(addresses) >= 20:
            return Severity.HIGH
        
        if total_volume > 50000 or len(addresses) >= 30:
            return Severity.HIGH
        
        if total_volume > 10000:
            return Severity.MEDIUM
        
        return Severity.LOW
