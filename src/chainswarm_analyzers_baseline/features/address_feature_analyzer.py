from typing import Dict, List, Any, Optional
from decimal import Decimal
from collections import defaultdict
import math

import networkx as nx
import numpy as np
from loguru import logger
from cdlib import algorithms as cd_algorithms
from scipy.stats import skew, kurtosis


class AddressFeatureAnalyzer:
    def analyze(
        self,
        graph: nx.DiGraph,
        address_labels: Dict[str, Dict[str, Any]],
        transfer_aggregates: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dict[str, Dict[str, Any]]:
        if graph.number_of_nodes() == 0:
            raise ValueError("Empty graph provided - cannot compute features")
        
        if transfer_aggregates is None:
            transfer_aggregates = {}
        
        logger.info(f"Computing features for {graph.number_of_nodes()} addresses")
        
        self._add_node_volume_attributes(graph)
        
        logger.info("Computing global graph analytics")
        graph_analytics = self._compute_all_graph_algorithms(graph)
        
        flows_by_address = self._build_flows_index_from_graph(graph)
        
        results = {}
        for address in graph.nodes():
            flows = flows_by_address.get(address, [])
            aggregates = transfer_aggregates.get(address, {})
            
            base_features = self._compute_base_features(address, flows)
            volume_features = self._compute_volume_features(address, flows)
            statistical_features = self._compute_statistical_features(address, flows, aggregates)
            flow_features = self._compute_flow_features(address, flows, aggregates)
            graph_features = graph_analytics.get(address, self._empty_graph_features())
            directional_features = self._compute_directional_flow_features(address, flows)
            behavioral_features = self._compute_behavioral_features(address, flows, aggregates)
            edge_features = self._compute_edge_features(address, graph)
            
            label_info = address_labels.get(address, {})
            label_features = self._compute_label_features(address, label_info)
            
            all_features = {
                'address': address,
                **base_features,
                **volume_features,
                **statistical_features,
                **flow_features,
                **graph_features,
                **directional_features,
                **behavioral_features,
                **edge_features,
                **label_features,
            }
            
            results[address] = all_features
        
        logger.info(f"Computed features for {len(results)} addresses")
        return results
    
    def _add_node_volume_attributes(self, G: nx.DiGraph) -> None:
        node_volumes = {}
        for node in G.nodes():
            in_volume = sum(
                data.get('amount_usd_sum', 0.0) 
                for _, _, data in G.in_edges(node, data=True)
            )
            out_volume = sum(
                data.get('amount_usd_sum', 0.0) 
                for _, _, data in G.out_edges(node, data=True)
            )
            node_volumes[node] = in_volume + out_volume
        nx.set_node_attributes(G, node_volumes, 'total_volume_usd')
    
    def _build_flows_index_from_graph(
        self,
        G: nx.DiGraph
    ) -> Dict[str, List[Dict[str, Any]]]:
        flows_by_address: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        for u, v, data in G.edges(data=True):
            flow = {
                'from_address': u,
                'to_address': v,
                'amount_usd_sum': float(data.get('amount_usd_sum', 0.0)),
                'tx_count': int(data.get('tx_count', 0))
            }
            flows_by_address[u].append(flow)
            flows_by_address[v].append(flow)
        
        return dict(flows_by_address)
    
    def _compute_base_features(
        self,
        address: str,
        flows: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        tx_in = sum(f['tx_count'] for f in flows if f['to_address'] == address)
        tx_out = sum(f['tx_count'] for f in flows if f['from_address'] == address)
        
        recipients = {f['to_address'] for f in flows if f['from_address'] == address}
        senders = {f['from_address'] for f in flows if f['to_address'] == address}
        
        return {
            'degree_in': len(senders),
            'degree_out': len(recipients),
            'degree_total': len(senders | recipients),
            'unique_counterparties': len(senders | recipients),
            'tx_in_count': tx_in,
            'tx_out_count': tx_out,
            'tx_total_count': tx_in + tx_out,
            'unique_recipients_count': len(recipients),
            'unique_senders_count': len(senders),
        }
    
    def _compute_volume_features(
        self,
        address: str,
        flows: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        total_in = Decimal('0')
        total_out = Decimal('0')
        amounts_in: List[float] = []
        amounts_out: List[float] = []
        
        for f in flows:
            amount = Decimal(str(f['amount_usd_sum']))
            if f['to_address'] == address:
                total_in += amount
                amounts_in.append(float(amount))
            if f['from_address'] == address:
                total_out += amount
                amounts_out.append(float(amount))
        
        all_amounts = amounts_in + amounts_out
        tx_in_count = len(amounts_in)
        tx_out_count = len(amounts_out)
        
        median_in = float(np.median(amounts_in)) if amounts_in else 0.0
        median_out = float(np.median(amounts_out)) if amounts_out else 0.0
        max_tx = float(max(all_amounts)) if all_amounts else 0.0
        min_tx = float(min(all_amounts)) if all_amounts else 0.0
        
        total_volume = total_in + total_out
        
        return {
            'total_in_usd': total_in,
            'total_out_usd': total_out,
            'net_flow_usd': total_in - total_out,
            'total_volume_usd': total_volume,
            'avg_tx_in_usd': total_in / max(tx_in_count, 1),
            'avg_tx_out_usd': total_out / max(tx_out_count, 1),
            'median_tx_in_usd': Decimal(str(median_in)),
            'median_tx_out_usd': Decimal(str(median_out)),
            'max_tx_usd': Decimal(str(max_tx)),
            'min_tx_usd': Decimal(str(min_tx)),
        }
    
    def _compute_statistical_features(
        self,
        address: str,
        flows: List[Dict[str, Any]],
        aggregates: Dict[str, Any]
    ) -> Dict[str, Any]:
        moments = aggregates.get('amount_moments', {})
        n = int(moments.get('n', 0))
        s1 = float(moments.get('s1', 0.0))
        s2 = float(moments.get('s2', 0.0))
        
        if n >= 2 and s1 > 0:
            mean = s1 / n
            variance = max(0, (s2 - s1 ** 2 / n) / (n - 1))
            std = math.sqrt(variance)
            cv = std / max(mean, 1.0)
        else:
            all_amounts = [float(f['amount_usd_sum']) for f in flows]
            
            if len(all_amounts) < 2:
                return {
                    'amount_variance': 0.0,
                    'amount_skewness': 0.0,
                    'amount_kurtosis': 0.0,
                    'volume_std': 0.0,
                    'volume_cv': 0.0,
                }
            
            variance = float(np.var(all_amounts, ddof=1))
            std = float(np.std(all_amounts, ddof=1))
            mean = float(np.mean(all_amounts))
            cv = std / max(mean, 1.0)
        
        all_amounts = [float(f['amount_usd_sum']) for f in flows]
        if len(all_amounts) >= 3:
            skewness = float(skew(all_amounts))
            kurt = float(kurtosis(all_amounts)) if len(all_amounts) >= 4 else 0.0
        else:
            skewness = 0.0
            kurt = 0.0
        
        return {
            'amount_variance': variance,
            'amount_skewness': skewness,
            'amount_kurtosis': kurt,
            'volume_std': std,
            'volume_cv': cv,
        }
    
    def _compute_flow_features(
        self,
        address: str,
        flows: List[Dict[str, Any]],
        aggregates: Dict[str, Any]
    ) -> Dict[str, Any]:
        if not flows:
            return {
                'flow_concentration': 0.0,
                'flow_diversity': 0.0,
                'counterparty_concentration': 0.0,
                'concentration_ratio': 0.0,
                'reciprocity_ratio': 0.0,
                'velocity_score': 0.0,
            }
        
        flow_amounts = [float(f['amount_usd_sum']) for f in flows]
        total_volume = sum(flow_amounts)
        
        if total_volume <= 0:
            return {
                'flow_concentration': 0.0,
                'flow_diversity': 0.0,
                'counterparty_concentration': 0.0,
                'concentration_ratio': 0.0,
                'reciprocity_ratio': 0.0,
                'velocity_score': 0.0,
            }
        
        counterparty_volumes: Dict[str, float] = defaultdict(float)
        for f in flows:
            cp = f['from_address'] if f['to_address'] == address else f['to_address']
            counterparty_volumes[cp] += float(f['amount_usd_sum'])
        
        total_in = sum(f['amount_usd_sum'] for f in flows if f['to_address'] == address)
        total_out = sum(f['amount_usd_sum'] for f in flows if f['from_address'] == address)
        total_vol = total_in + total_out
        
        tx_total = sum(f['tx_count'] for f in flows)
        
        reciprocity_stats = aggregates.get('reciprocity_stats', {})
        if reciprocity_stats.get('total_volume', 0) > 0:
            recip_total = float(reciprocity_stats.get('total_volume', 0))
            recip_vol = float(reciprocity_stats.get('reciprocal_volume', 0))
            reciprocity_ratio = min(recip_vol, recip_total) / max(recip_total, 1)
        else:
            reciprocity_ratio = float(min(total_in, total_out) / max(total_vol, 1))
        
        return {
            'flow_concentration': self._calculate_gini_coefficient(flow_amounts),
            'flow_diversity': self._calculate_normalized_entropy(flow_amounts),
            'counterparty_concentration': self._calculate_gini_coefficient(
                list(counterparty_volumes.values())
            ),
            'concentration_ratio': max(counterparty_volumes.values()) / total_volume
                if counterparty_volumes and total_volume > 0 else 0.0,
            'reciprocity_ratio': reciprocity_ratio,
            'velocity_score': float(min(tx_total / 100.0, 1.0)),
        }
    
    def _compute_directional_flow_features(
        self,
        address: str,
        flows: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        if not flows:
            return {
                'in_out_ratio': 0.5,
                'flow_asymmetry': 0.0,
                'dominant_flow_direction': 'balanced',
                'flow_direction_entropy': 0.0,
                'counterparty_overlap_ratio': 0.0,
            }
        
        total_in = sum(f['amount_usd_sum'] for f in flows if f['to_address'] == address)
        total_out = sum(f['amount_usd_sum'] for f in flows if f['from_address'] == address)
        
        senders = {f['from_address'] for f in flows if f['to_address'] == address}
        recipients = {f['to_address'] for f in flows if f['from_address'] == address}
        
        total_vol = total_in + total_out
        
        in_count = len([f for f in flows if f['to_address'] == address])
        out_count = len([f for f in flows if f['from_address'] == address])
        total_count = in_count + out_count
        
        if total_count > 0:
            p_in = in_count / total_count
            p_out = out_count / total_count
            dir_entropy = self._calculate_shannon_entropy([p_in, p_out])
        else:
            dir_entropy = 0.0
        
        if total_in > 1.5 * total_out:
            dominant = 'incoming'
        elif total_out > 1.5 * total_in:
            dominant = 'outgoing'
        else:
            dominant = 'balanced'
        
        return {
            'in_out_ratio': total_in / total_vol if total_vol > 0 else 0.5,
            'flow_asymmetry': abs(total_in - total_out) / total_vol if total_vol > 0 else 0.0,
            'dominant_flow_direction': dominant,
            'flow_direction_entropy': dir_entropy,
            'counterparty_overlap_ratio': len(senders & recipients) / max(len(senders | recipients), 1),
        }
    
    def _compute_behavioral_features(
        self,
        address: str,
        flows: List[Dict[str, Any]],
        aggregates: Dict[str, Any]
    ) -> Dict[str, Any]:
        temporal_patterns = aggregates.get('temporal_patterns', {})
        temporal_summaries = aggregates.get('temporal_summaries', {})
        behavioral_counters = aggregates.get('behavioral_counters', {})
        hourly_volumes = aggregates.get('hourly_volumes', [0.0] * 24)
        interevent_stats = aggregates.get('interevent_stats', {})
        reciprocity_stats = aggregates.get('reciprocity_stats', {})
        
        hourly_activity = temporal_patterns.get('hourly_activity', [0] * 24)
        daily_activity = temporal_patterns.get('daily_activity', [0] * 7)
        peak_activity_hour = temporal_patterns.get('peak_activity_hour', 0)
        peak_activity_day = temporal_patterns.get('peak_activity_day', 0)
        
        if len(hourly_activity) != 24:
            hourly_activity = (list(hourly_activity) + [0] * 24)[:24]
        if len(daily_activity) != 7:
            daily_activity = (list(daily_activity) + [0] * 7)[:7]
        
        first_timestamp = temporal_summaries.get('first_timestamp', 0)
        last_timestamp = temporal_summaries.get('last_timestamp', 0)
        total_tx_count = temporal_summaries.get('total_tx_count', 0)
        activity_days = temporal_summaries.get('distinct_activity_days', 1)
        total_volume = temporal_summaries.get('total_volume', 0.0)
        weekend_tx_count = temporal_summaries.get('weekend_tx_count', 0)
        night_tx_count = temporal_summaries.get('night_tx_count', 0)
        
        if not isinstance(hourly_volumes, list) or len(hourly_volumes) != 24:
            hourly_volumes = [0.0] * 24
        
        total_txs = sum(hourly_activity)
        hour_entropy = self._calculate_entropy(hourly_activity) if total_txs > 0 else 0.0
        daily_entropy = self._calculate_entropy(daily_activity) if total_txs > 0 else 0.0
        
        max_entropy = math.log2(24) if 24 > 1 else 1.0
        regularity_score = 1.0 - (hour_entropy / max_entropy) if hour_entropy > 0 else 1.0
        
        burst_factor = 0.0
        if total_txs > 0:
            max_hourly = max(hourly_activity)
            expected = total_txs / 24.0
            burst_factor = max_hourly / expected if expected > 0 else 0.0
        
        weekend_ratio = weekend_tx_count / max(total_tx_count, 1)
        night_ratio = night_tx_count / max(total_tx_count, 1)
        
        consistency_score = regularity_score * (1.0 - (hour_entropy / max_entropy)) if hour_entropy > 0 else regularity_score
        
        activity_span_days = 1
        if first_timestamp > 0 and last_timestamp > first_timestamp:
            activity_span_days = max(1, (last_timestamp - first_timestamp) // 86400000)
        
        avg_daily_volume = Decimal(str(total_volume / max(activity_days, 1)))
        
        total_tx_pos = behavioral_counters.get('total_tx_pos_amount', 0)
        round_number_count = behavioral_counters.get('round_number_count', 0)
        small_amount_count = behavioral_counters.get('small_amount_count', 0)
        unusual_tx_count = behavioral_counters.get('unusual_tx_count', 0)
        
        if total_tx_pos > 0:
            round_number_ratio = round_number_count / total_tx_pos
            unusual_timing_score = unusual_tx_count / total_tx_pos
            small_ratio = small_amount_count / total_tx_pos
            structuring_score = min(1.0, small_ratio * 1.5 if small_ratio > 0.5 and small_amount_count >= 3 else small_ratio)
        else:
            round_number_ratio = 0.0
            unusual_timing_score = 0.0
            structuring_score = 0.0
        
        non_zero_volumes = [v for v in hourly_volumes if v > 0]
        hourly_volume_variance = float(np.var(non_zero_volumes)) if len(non_zero_volumes) > 1 else 0.0
        peak_volume_hour = int(np.argmax(hourly_volumes)) if sum(hourly_volumes) > 0 else 0
        
        sum_volumes = sum(non_zero_volumes)
        intraday_volume_ratio = 0.0
        if sum_volumes > 0:
            max_vol = max(hourly_volumes)
            expected_vol = sum_volumes / 24.0
            intraday_volume_ratio = max_vol / expected_vol if expected_vol > 0 else 0.0
        
        hourly_tx_entropy = 0.0
        if total_txs > 0:
            probs = [c / total_txs for c in hourly_activity if c > 0]
            hourly_tx_entropy = self._calculate_shannon_entropy(probs) if probs else 0.0
        
        volume_concentration_score = self._calculate_gini_coefficient(non_zero_volumes)
        
        mean_inter_s = interevent_stats.get('mean_inter_s', 0.0)
        std_inter_s = interevent_stats.get('std_inter_s', 0.0)
        n_inter = interevent_stats.get('n', 0)
        
        flow_burstiness = 0.0
        if n_inter >= 2 and (mean_inter_s + std_inter_s) > 0:
            flow_burstiness = max(0.0, min(1.0, (std_inter_s - mean_inter_s) / (std_inter_s + mean_inter_s)))
        
        from ..adapters.parquet import ParquetAdapter
        all_amounts = [float(f['amount_usd_sum']) for f in flows]
        amount_predictability = 0.0
        if len(all_amounts) > 1:
            mean_amt = np.mean(all_amounts)
            std_amt = np.std(all_amounts)
            cv = std_amt / max(mean_amt, 1.0)
            amount_predictability = max(0.0, 1.0 - min(1.0, cv))
        
        total_vol = float(reciprocity_stats.get('total_volume', 0))
        recip_vol = float(reciprocity_stats.get('reciprocal_volume', 0))
        flow_reciprocity_entropy = 0.0
        if total_vol > 0:
            p_rec = max(0.0, min(1.0, recip_vol / total_vol))
            p_non = 1.0 - p_rec
            ent = 0.0
            if p_rec > 0:
                ent -= p_rec * math.log2(p_rec)
            if p_non > 0:
                ent -= p_non * math.log2(p_non)
            flow_reciprocity_entropy = max(0.0, min(1.0, ent))
        
        return {
            'activity_days': activity_days,
            'activity_span_days': activity_span_days,
            'avg_daily_volume_usd': avg_daily_volume,
            'peak_hour': peak_activity_hour,
            'peak_day': peak_activity_day,
            'regularity_score': regularity_score,
            'burst_factor': burst_factor,
            'hourly_entropy': hour_entropy,
            'daily_entropy': daily_entropy,
            'weekend_transaction_ratio': weekend_ratio,
            'night_transaction_ratio': night_ratio,
            'consistency_score': consistency_score,
            'is_new_address': first_timestamp > 0,
            'is_dormant_reactivated': False,
            'round_number_ratio': round_number_ratio,
            'unusual_timing_score': unusual_timing_score,
            'structuring_score': structuring_score,
            'hourly_volume_variance': hourly_volume_variance,
            'peak_volume_hour': peak_volume_hour,
            'intraday_volume_ratio': intraday_volume_ratio,
            'hourly_transaction_entropy': hourly_tx_entropy,
            'volume_concentration_score': volume_concentration_score,
            'hourly_activity': [int(x) for x in hourly_activity],
            'daily_activity': [int(x) for x in daily_activity],
            'peak_activity_hour': peak_activity_hour,
            'peak_activity_day': peak_activity_day,
            'small_transaction_ratio': structuring_score,
            'flow_reciprocity_entropy': flow_reciprocity_entropy,
            'counterparty_stability': 0.0,
            'flow_burstiness': flow_burstiness,
            'transaction_regularity': regularity_score,
            'amount_predictability': amount_predictability,
            'unique_assets_in': 1,
            'unique_assets_out': 1,
            'dominant_asset_in': 'NATIVE',
            'dominant_asset_out': 'NATIVE',
            'asset_diversity_score': 0.0,
            'first_activity_timestamp': first_timestamp,
            'last_activity_timestamp': last_timestamp,
        }
    
    def _compute_edge_features(self, address: str, graph: nx.DiGraph) -> Dict[str, Any]:
        in_edges = list(graph.in_edges(address, data=True))
        out_edges = list(graph.out_edges(address, data=True))
        all_edges = in_edges + out_edges
        
        if not all_edges:
            return {
                'avg_relationship_age_days': 0,
                'max_relationship_age_days': 0,
                'bidirectional_relationship_ratio': 0.0,
                'avg_edge_reciprocity': 0.0,
                'multi_asset_edge_ratio': 0.0,
                'edge_hourly_entropy': 0.0,
                'edge_weekly_entropy': 0.0,
            }
        
        relationship_ages = []
        bidirectional_count = 0
        reciprocity_scores = []
        multi_asset_count = 0
        aggregate_hourly = [0] * 24
        aggregate_weekly = [0] * 7
        
        for _, _, data in all_edges:
            first_seen = data['first_seen_timestamp']
            last_seen = data['last_seen_timestamp']
            if first_seen > 0:
                age_days = (last_seen - first_seen) / 86400000
                relationship_ages.append(age_days)
            
            if data.get('is_bidirectional', False):
                bidirectional_count += 1
            
            reciprocity_scores.append(data.get('reciprocity_ratio', 0.0))
            
            if data.get('unique_assets', 0) > 1:
                multi_asset_count += 1
            
            hourly = data.get('hourly_pattern', [0] * 24)
            weekly = data.get('weekly_pattern', [0] * 7)
            for i in range(24):
                aggregate_hourly[i] += hourly[i]
            for i in range(7):
                aggregate_weekly[i] += weekly[i]
        
        avg_relationship_age = sum(relationship_ages) / len(relationship_ages) if relationship_ages else 0
        max_relationship_age = max(relationship_ages) if relationship_ages else 0
        bidirectional_ratio = bidirectional_count / len(all_edges)
        avg_reciprocity = sum(reciprocity_scores) / len(reciprocity_scores)
        multi_asset_ratio = multi_asset_count / len(all_edges)
        
        hourly_entropy = self._calculate_entropy(aggregate_hourly)
        weekly_entropy = self._calculate_entropy(aggregate_weekly)
        
        return {
            'avg_relationship_age_days': int(avg_relationship_age),
            'max_relationship_age_days': int(max_relationship_age),
            'bidirectional_relationship_ratio': float(bidirectional_ratio),
            'avg_edge_reciprocity': float(avg_reciprocity),
            'multi_asset_edge_ratio': float(multi_asset_ratio),
            'edge_hourly_entropy': float(hourly_entropy),
            'edge_weekly_entropy': float(weekly_entropy),
        }
    
    def _compute_label_features(
        self,
        address: str,
        label_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        address_type = label_info.get('address_type', 'UNKNOWN')
        trust_level = label_info.get('trust_level', 'UNKNOWN')
        
        return {
            'is_whale': address_type == 'WHALE',
            'is_exchange_like': address_type in ('EXCHANGE', 'DEX'),
            'is_mixer_like': address_type == 'MIXER',
            'is_hub': False,
            'is_dust_collector': False,
            'is_dormant_activated': False,
            'has_cycle_involvement': False,
            'has_layering_involvement': False,
            'data_quality_score': 1.0 if trust_level == 'VERIFIED' else 0.5,
            'feature_confidence_score': 1.0,
            'risk_proximity_1hop': 0.0,
            'risk_proximity_2hop': 0.0,
        }
    
    def _compute_all_graph_algorithms(
        self,
        G: nx.DiGraph
    ) -> Dict[str, Dict[str, Any]]:
        results: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: self._empty_graph_features()
        )
        
        pagerank_scores = self._compute_pagerank(G)
        betweenness_scores = self._compute_betweenness_centrality(G)
        closeness_scores = self._compute_closeness_centrality(G)
        kcore_scores = self._compute_kcore(G)
        clustering_scores = self._compute_clustering_coefficient(G)
        communities = self._compute_community_detection(G)
        khop_features = self._compute_khop_features(G)
        
        for address in G.nodes():
            pagerank_val = pagerank_scores.get(address, 0.0)
            betweenness_val = betweenness_scores.get(address, 0.0)
            clustering_val = clustering_scores.get(address, 0.0)
            
            results[address].update({
                'pagerank': pagerank_val,
                'betweenness': betweenness_val,
                'closeness': closeness_scores.get(address, 0.0),
                'clustering_coefficient': clustering_val,
                'kcore': kcore_scores.get(address, 0),
                'community_id': communities.get(address, -1),
                'centrality_score': (
                    pagerank_val * 0.4 + 
                    betweenness_val * 0.3 + 
                    clustering_val * 0.3
                ),
                'degree': G.degree(address),
            })
            
            khop = khop_features.get(address, {})
            results[address].update({
                'khop1_count': khop.get('khop1_count', 0),
                'khop2_count': khop.get('khop2_count', 0),
                'khop3_count': khop.get('khop3_count', 0),
                'khop1_volume_usd': Decimal(str(khop.get('khop1_volume_usd', 0))),
                'khop2_volume_usd': Decimal(str(khop.get('khop2_volume_usd', 0))),
                'khop3_volume_usd': Decimal(str(khop.get('khop3_volume_usd', 0))),
            })
        
        logger.info(f"Graph analytics computed for {len(results)} addresses")
        return dict(results)
    
    def _compute_pagerank(
        self,
        G: nx.DiGraph
    ) -> Dict[str, float]:
        try:
            # Check if graph has meaningful weights
            total_weight = sum(
                data.get('weight', 0) for _, _, data in G.edges(data=True)
            )
            if total_weight == 0:
                # Fallback to unweighted PageRank when USD prices unavailable
                logger.info("PageRank: zero weights detected, using unweighted computation")
                return nx.pagerank(G, alpha=0.85)  # No weight parameter
            return nx.pagerank(G, weight='weight', alpha=0.85)
        except (nx.PowerIterationFailedConvergence, ZeroDivisionError) as e:
            logger.warning(f"PageRank failed: {e}, using default values")
            return {node: 0.0 for node in G.nodes()}
    
    def _compute_betweenness_centrality(
        self,
        G: nx.DiGraph
    ) -> Dict[str, float]:
        n_nodes = G.number_of_nodes()
        if n_nodes <= 1:
            return {node: 0.0 for node in G.nodes()}
        
        k = min(1000, n_nodes - 1) if n_nodes > 2 else None
        
        try:
            return nx.betweenness_centrality(
                G, k=k, weight='weight', normalized=True
            )
        except Exception as e:
            logger.warning(f"Betweenness centrality failed: {e}")
            return {node: 0.0 for node in G.nodes()}
    
    def _compute_closeness_centrality(
        self,
        G: nx.DiGraph
    ) -> Dict[str, float]:
        try:
            # Check if graph has meaningful weights for distance calculation
            total_weight = sum(
                data.get('weight', 0) for _, _, data in G.edges(data=True)
            )
            if total_weight == 0:
                # Fallback to unweighted closeness (hop-based distance)
                logger.info("Closeness centrality: zero weights, using hop-based distance")
                return nx.closeness_centrality(G)  # No distance parameter
            return nx.closeness_centrality(G, distance='weight')
        except Exception as e:
            logger.warning(f"Closeness centrality failed: {e}")
            return {node: 0.0 for node in G.nodes()}
    
    def _compute_kcore(
        self,
        G: nx.DiGraph
    ) -> Dict[str, int]:
        try:
            return nx.core_number(G.to_undirected())
        except Exception as e:
            logger.warning(f"K-core computation failed: {e}")
            return {node: 0 for node in G.nodes()}
    
    def _compute_clustering_coefficient(
        self,
        G: nx.DiGraph
    ) -> Dict[str, float]:
        try:
            G_undirected = G.to_undirected()
            # Check if graph has meaningful weights
            total_weight = sum(
                data.get('weight', 0) for _, _, data in G_undirected.edges(data=True)
            )
            if total_weight == 0:
                # Fallback to unweighted clustering
                logger.info("Clustering coefficient: zero weights, using unweighted")
                return nx.clustering(G_undirected)  # No weight parameter
            return nx.clustering(G_undirected, weight='weight')
        except Exception as e:
            logger.warning(f"Clustering coefficient failed: {e}")
            return {node: 0.0 for node in G.nodes()}
    
    def _compute_community_detection(
        self,
        G: nx.DiGraph
    ) -> Dict[str, int]:
        if G.number_of_nodes() < 2:
            return {node: 0 for node in G.nodes()}
        
        try:
            G_undirected = G.to_undirected()
            
            # Check if graph has meaningful weights
            total_weight = sum(
                data.get('weight', 0) for _, _, data in G_undirected.edges(data=True)
            )
            
            if total_weight == 0:
                # Fallback: use tx_count as weight when USD values unavailable
                logger.info("Community detection: zero USD weights, using tx_count")
                for u, v, data in G_undirected.edges(data=True):
                    data['weight'] = max(data.get('tx_count', 1), 1)
            
            coms = cd_algorithms.leiden(G_undirected, weights='weight')
            return {
                node: i
                for i, com in enumerate(coms.communities)
                for node in com
            }
        except Exception as e:
            logger.warning(f"Community detection failed: {e}")
            return {node: -1 for node in G.nodes()}
    
    def _compute_khop_features(
        self,
        G: nx.DiGraph,
        max_k: int = 3
    ) -> Dict[str, Dict[str, Any]]:
        khop_results: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        for node in G.nodes():
            for k in range(1, max_k + 1):
                try:
                    neighbors = set(
                        nx.single_source_shortest_path_length(G, node, cutoff=k).keys()
                    ) - {node}
                    
                    neighbor_volume = sum(
                        G.nodes[n].get('total_volume_usd', 0.0) 
                        for n in neighbors
                    )
                    
                    khop_results[node][f'khop{k}_count'] = len(neighbors)
                    khop_results[node][f'khop{k}_volume_usd'] = neighbor_volume
                except Exception:
                    khop_results[node][f'khop{k}_count'] = 0
                    khop_results[node][f'khop{k}_volume_usd'] = 0.0
        
        return dict(khop_results)
    
    def _empty_graph_features(self) -> Dict[str, Any]:
        return {
            'pagerank': 0.0,
            'betweenness': 0.0,
            'closeness': 0.0,
            'clustering_coefficient': 0.0,
            'kcore': 0,
            'community_id': -1,
            'centrality_score': 0.0,
            'degree': 0,
            'khop1_count': 0,
            'khop2_count': 0,
            'khop3_count': 0,
            'khop1_volume_usd': Decimal('0'),
            'khop2_volume_usd': Decimal('0'),
            'khop3_volume_usd': Decimal('0'),
        }
    
    def _calculate_entropy(self, values: List[int]) -> float:
        total = sum(values)
        if total == 0:
            return 0.0
        return -sum(
            (v / total) * math.log2(v / total) 
            for v in values if v > 0
        )
    
    def _calculate_shannon_entropy(self, probabilities: List[float]) -> float:
        return -sum(
            p * math.log2(p) 
            for p in probabilities if p > 0
        )
    
    def _calculate_normalized_entropy(self, values: List[float]) -> float:
        if len(values) <= 1:
            return 0.0
        
        total = sum(values)
        if total <= 0:
            return 0.0
        
        probs = [v / total for v in values if v > 0]
        if not probs:
            return 0.0
        
        entropy = -sum(p * math.log2(p) for p in probs if p > 0)
        max_entropy = math.log2(len(probs)) if len(probs) > 1 else 1.0
        
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def _calculate_gini_coefficient(self, values: List[float]) -> float:
        if not values:
            return 0.0
        
        values = sorted([v for v in values if v > 0])
        n = len(values)
        if n <= 1:
            return 0.0
        
        total = sum(values)
        if total <= 0:
            return 0.0
        
        cumsum = 0.0
        for i, v in enumerate(values):
            cumsum += (i + 1) * v
        
        return 1.0 - (2.0 * cumsum) / (n * total) + (n + 1) / n
