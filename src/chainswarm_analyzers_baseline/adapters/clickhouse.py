from typing import Dict, List, Any, Optional
from datetime import datetime
from decimal import Decimal
from collections import defaultdict
import time

from clickhouse_connect.driver import Client
from chainswarm_core.db import row_to_dict
from chainswarm_core.constants import PatternTypes
from loguru import logger


def _generate_version() -> int:
    return int(time.time() * 1000)


class ClickHouseAdapter:
    PATTERN_TYPE_TABLES = {
        PatternTypes.CYCLE: 'analyzers_patterns_cycle',
        PatternTypes.LAYERING_PATH: 'analyzers_patterns_layering',
        PatternTypes.SMURFING_NETWORK: 'analyzers_patterns_network',
        PatternTypes.PROXIMITY_RISK: 'analyzers_patterns_proximity',
        PatternTypes.MOTIF_FANIN: 'analyzers_patterns_motif',
        PatternTypes.MOTIF_FANOUT: 'analyzers_patterns_motif',
        PatternTypes.TEMPORAL_BURST: 'analyzers_patterns_burst',
        PatternTypes.THRESHOLD_EVASION: 'analyzers_patterns_threshold',
    }
    
    def __init__(self, client: Client, network: Optional[str] = None):
        self.client = client
        self.network = network
    
    def read_transfers(
        self,
        start_timestamp_ms: int,
        end_timestamp_ms: int
    ) -> List[Dict[str, Any]]:
        query = """
            SELECT
                tx_id,
                event_index,
                edge_index,
                block_height,
                block_timestamp,
                from_address,
                to_address,
                asset_symbol,
                asset_contract,
                amount,
                fee,
                amount_usd
            FROM core_transfers FINAL
            WHERE block_timestamp >= %(start_ts)s
              AND block_timestamp < %(end_ts)s
            ORDER BY block_timestamp, tx_id, event_index, edge_index
        """
        
        params = {
            'start_ts': int(start_timestamp_ms),
            'end_ts': int(end_timestamp_ms),
        }
        
        result = self.client.query(query, parameters=params)
        transfers = [row_to_dict(row, result.column_names) for row in result.result_rows]
        
        logger.info(f"Read {len(transfers)} transfers from ClickHouse")
        
        return transfers
    
    def read_money_flows(
        self,
        start_timestamp_ms: int,
        end_timestamp_ms: int
    ) -> List[Dict[str, Any]]:
        query = """
            SELECT
                from_address,
                to_address,
                tx_count,
                amount_usd_sum,
                first_seen_timestamp,
                last_seen_timestamp,
                active_days,
                avg_tx_size_usd,
                unique_assets,
                dominant_asset,
                hourly_pattern,
                weekly_pattern,
                reciprocity_ratio,
                is_bidirectional
            FROM core_money_flows_view FINAL
            WHERE last_seen_timestamp >= %(start_ts)s
              AND last_seen_timestamp < %(end_ts)s
        """
        
        params = {
            'start_ts': int(start_timestamp_ms),
            'end_ts': int(end_timestamp_ms),
        }
        
        result = self.client.query(query, parameters=params)
        money_flows = [row_to_dict(row, result.column_names) for row in result.result_rows]
        
        logger.info(f"Read {len(money_flows)} money flows from MV")
        
        return money_flows
    
    def read_transfer_timestamps(
        self,
        start_timestamp_ms: int,
        end_timestamp_ms: int,
        addresses: List[str]
    ) -> Dict[str, List[Dict[str, Any]]]:
        if not addresses:
            return {}
        
        address_list = ", ".join([f"'{addr}'" for addr in addresses])
        
        query = f"""
            SELECT
                multiIf(from_address IN ({address_list}), from_address, to_address) AS address,
                block_timestamp,
                amount_usd,
                multiIf(from_address IN ({address_list}), to_address, from_address) AS counterparty
            FROM core_transfers FINAL
            WHERE block_timestamp >= %(start_ts)s
              AND block_timestamp < %(end_ts)s
              AND (from_address IN ({address_list}) OR to_address IN ({address_list}))
            ORDER BY address, block_timestamp
        """
        
        params = {
            'start_ts': int(start_timestamp_ms),
            'end_ts': int(end_timestamp_ms),
        }
        
        result = self.client.query(query, parameters=params)
        
        address_timestamps = defaultdict(list)
        for row in result.result_rows:
            row_dict = row_to_dict(row, result.column_names)
            address = row_dict['address']
            address_timestamps[address].append({
                'timestamp': row_dict['block_timestamp'],
                'volume': float(row_dict['amount_usd']),
                'counterparty': row_dict['counterparty']
            })
        
        logger.info(f"Read timestamps for {len(address_timestamps)} addresses")
        
        return dict(address_timestamps)
    
    def read_assets(self, network: Optional[str] = None) -> List[Dict[str, Any]]:
        if network:
            query = """
                SELECT
                    asset_symbol,
                    asset_contract,
                    network,
                    verified,
                    verification_source,
                    first_seen_timestamp
                FROM core_assets FINAL
                WHERE network = %(network)s
            """
            params = {'network': network}
            result = self.client.query(query, parameters=params)
        else:
            query = """
                SELECT
                    asset_symbol,
                    asset_contract,
                    network,
                    verified,
                    verification_source,
                    first_seen_timestamp
                FROM core_assets FINAL
            """
            result = self.client.query(query)
        
        assets = [row_to_dict(row, result.column_names) for row in result.result_rows]
        
        logger.info(f"Read {len(assets)} assets from ClickHouse")
        
        return assets
    
    def read_asset_prices(
        self,
        start_timestamp_ms: int,
        end_timestamp_ms: int
    ) -> List[Dict[str, Any]]:
        from datetime import date
        start_date = date.fromtimestamp(start_timestamp_ms / 1000)
        end_date = date.fromtimestamp(end_timestamp_ms / 1000)
        
        query = """
            SELECT
                asset_symbol,
                asset_contract,
                price_date,
                price_usd,
                source
            FROM core_asset_prices FINAL
            WHERE price_date >= %(start_date)s
              AND price_date < %(end_date)s
        """
        
        params = {
            'start_date': start_date,
            'end_date': end_date,
        }
        
        result = self.client.query(query, parameters=params)
        prices = [row_to_dict(row, result.column_names) for row in result.result_rows]
        
        logger.info(f"Read {len(prices)} asset prices from ClickHouse")
        
        return prices
    
    def read_address_labels(
        self,
        addresses: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        if not addresses:
            return {}
        
        address_list = ", ".join([f"'{addr}'" for addr in addresses])
        
        query = f"""
            SELECT
                address,
                label,
                address_type,
                trust_level,
                source
            FROM core_address_labels FINAL
            WHERE address IN ({address_list})
        """
        
        result = self.client.query(query)
        
        labels = {}
        for row in result.result_rows:
            row_dict = row_to_dict(row, result.column_names)
            address = row_dict['address']
            labels[address] = {
                'label': row_dict.get('label', ''),
                'address_type': row_dict.get('address_type', ''),
                'trust_level': row_dict.get('trust_level', ''),
                'source': row_dict.get('source', ''),
            }
        
        logger.info(f"Found labels for {len(labels)} of {len(addresses)} addresses")
        
        return labels
    
    def write_features(
        self,
        features: List[Dict[str, Any]],
        window_days: int,
        processing_date: str
    ) -> None:
        if not features:
            raise ValueError("No features to write")
        
        batch_size = 1000
        date_obj = datetime.strptime(processing_date, '%Y-%m-%d').date()
        
        logger.info(f"Inserting {len(features)} features into analyzers_features")
        
        column_names = [
            'window_days', 'processing_date',
            'address', 'degree_in', 'degree_out', 'degree_total', 'unique_counterparties',
            'total_in_usd', 'total_out_usd', 'net_flow_usd', 'total_volume_usd',
            'avg_tx_in_usd', 'avg_tx_out_usd', 'median_tx_in_usd', 'median_tx_out_usd',
            'max_tx_usd', 'min_tx_usd',
            'amount_variance', 'amount_skewness', 'amount_kurtosis',
            'volume_std', 'volume_cv', 'flow_concentration',
            'tx_in_count', 'tx_out_count', 'tx_total_count',
            'activity_days', 'activity_span_days', 'avg_daily_volume_usd',
            'peak_hour', 'peak_day', 'regularity_score', 'burst_factor',
            'reciprocity_ratio', 'flow_diversity', 'counterparty_concentration',
            'velocity_score', 'structuring_score',
            'hourly_entropy', 'daily_entropy', 'weekend_transaction_ratio',
            'night_transaction_ratio', 'consistency_score',
            'is_new_address', 'is_dormant_reactivated',
            'unique_recipients_count', 'unique_senders_count',
            'pagerank', 'betweenness', 'closeness', 'clustering_coefficient',
            'kcore', 'community_id', 'centrality_score',
            'khop1_count', 'khop2_count', 'khop3_count',
            'khop1_volume_usd', 'khop2_volume_usd', 'khop3_volume_usd',
            'flow_reciprocity_entropy', 'counterparty_stability', 'flow_burstiness',
            'transaction_regularity', 'amount_predictability',
            'first_activity_timestamp', 'last_activity_timestamp',
            'unique_assets_in', 'unique_assets_out',
            'dominant_asset_in', 'dominant_asset_out', 'asset_diversity_score',
            'hourly_activity', 'daily_activity',
            'peak_activity_hour', 'peak_activity_day',
            'small_transaction_ratio', 'concentration_ratio',
            'avg_relationship_age_days',
            'max_relationship_age_days',
            'bidirectional_relationship_ratio',
            'avg_edge_reciprocity',
            'multi_asset_edge_ratio',
            'edge_hourly_entropy',
            'edge_weekly_entropy',
            '_version'
        ]
        
        for i in range(0, len(features), batch_size):
            batch = features[i:i + batch_size]
            batch_data = []
            
            for feature in batch:
                row = [
                    window_days,
                    date_obj,
                    feature['address'],
                    int(feature.get('degree_in', 0)),
                    int(feature.get('degree_out', 0)),
                    int(feature.get('degree_total', 0)),
                    int(feature.get('unique_counterparties', 0)),
                    str(feature.get('total_in_usd', 0)),
                    str(feature.get('total_out_usd', 0)),
                    str(feature.get('net_flow_usd', 0)),
                    str(feature.get('total_volume_usd', 0)),
                    str(feature.get('avg_tx_in_usd', 0)),
                    str(feature.get('avg_tx_out_usd', 0)),
                    str(feature.get('median_tx_in_usd', 0)),
                    str(feature.get('median_tx_out_usd', 0)),
                    str(feature.get('max_tx_usd', 0)),
                    str(feature.get('min_tx_usd', 0)),
                    float(feature.get('amount_variance', 0.0)),
                    float(feature.get('amount_skewness', 0.0)),
                    float(feature.get('amount_kurtosis', 0.0)),
                    float(feature.get('volume_std', 0.0)),
                    float(feature.get('volume_cv', 0.0)),
                    float(feature.get('flow_concentration', 0.0)),
                    int(feature.get('tx_in_count', 0)),
                    int(feature.get('tx_out_count', 0)),
                    int(feature.get('tx_total_count', 0)),
                    int(feature.get('activity_days', 0)),
                    int(feature.get('activity_span_days', 0)),
                    str(feature.get('avg_daily_volume_usd', 0)),
                    int(feature.get('peak_hour', 0)),
                    int(feature.get('peak_day', 0)),
                    float(feature.get('regularity_score', 0.0)),
                    float(feature.get('burst_factor', 0.0)),
                    float(feature.get('reciprocity_ratio', 0.0)),
                    float(feature.get('flow_diversity', 0.0)),
                    float(feature.get('counterparty_concentration', 0.0)),
                    float(feature.get('velocity_score', 0.0)),
                    float(feature.get('structuring_score', 0.0)),
                    float(feature.get('hourly_entropy', 0.0)),
                    float(feature.get('daily_entropy', 0.0)),
                    float(feature.get('weekend_transaction_ratio', 0.0)),
                    float(feature.get('night_transaction_ratio', 0.0)),
                    float(feature.get('consistency_score', 0.0)),
                    bool(feature.get('is_new_address', False)),
                    bool(feature.get('is_dormant_reactivated', False)),
                    int(feature.get('unique_recipients_count', 0)),
                    int(feature.get('unique_senders_count', 0)),
                    float(feature.get('pagerank', 0.0)),
                    float(feature.get('betweenness', 0.0)),
                    float(feature.get('closeness', 0.0)),
                    float(feature.get('clustering_coefficient', 0.0)),
                    int(feature.get('kcore', 0)),
                    int(feature.get('community_id', 0)),
                    float(feature.get('centrality_score', 0.0)),
                    int(feature.get('khop1_count', 0)),
                    int(feature.get('khop2_count', 0)),
                    int(feature.get('khop3_count', 0)),
                    str(feature.get('khop1_volume_usd', 0)),
                    str(feature.get('khop2_volume_usd', 0)),
                    str(feature.get('khop3_volume_usd', 0)),
                    float(feature.get('flow_reciprocity_entropy', 0.0)),
                    float(feature.get('counterparty_stability', 0.0)),
                    float(feature.get('flow_burstiness', 0.0)),
                    float(feature.get('transaction_regularity', 0.0)),
                    float(feature.get('amount_predictability', 0.0)),
                    int(feature.get('first_activity_timestamp', 0)),
                    int(feature.get('last_activity_timestamp', 0)),
                    int(feature.get('unique_assets_in', 0)),
                    int(feature.get('unique_assets_out', 0)),
                    str(feature.get('dominant_asset_in', '')),
                    str(feature.get('dominant_asset_out', '')),
                    float(feature.get('asset_diversity_score', 0.0)),
                    [int(x) for x in feature.get('hourly_activity', [])],
                    [int(x) for x in feature.get('daily_activity', [])],
                    int(feature.get('peak_activity_hour', 0)),
                    int(feature.get('peak_activity_day', 0)),
                    float(feature.get('small_transaction_ratio', 0.0)),
                    float(feature.get('concentration_ratio', 0.0)),
                    int(feature.get('avg_relationship_age_days', 0)),
                    int(feature.get('max_relationship_age_days', 0)),
                    float(feature.get('bidirectional_relationship_ratio', 0.0)),
                    float(feature.get('avg_edge_reciprocity', 0.0)),
                    float(feature.get('multi_asset_edge_ratio', 0.0)),
                    float(feature.get('edge_hourly_entropy', 0.0)),
                    float(feature.get('edge_weekly_entropy', 0.0)),
                    _generate_version(),
                ]
                batch_data.append(row)
            
            self.client.insert(
                'analyzers_features',
                batch_data,
                column_names=column_names
            )
        
        logger.info(f"Inserted {len(features)} features into analyzers_features")
    
    def write_patterns(
        self,
        patterns: List[Dict[str, Any]],
        window_days: int,
        processing_date: str
    ) -> None:
        if not patterns:
            logger.info("No patterns to write")
            return
        
        batch_size = 1000
        date_obj = datetime.strptime(processing_date, '%Y-%m-%d').date()
        
        logger.info(f"Inserting {len(patterns)} patterns into type-specific tables")
        
        patterns_by_type: Dict[str, List[Dict]] = defaultdict(list)
        for pattern in patterns:
            pattern_type = pattern.get('pattern_type', 'UNKNOWN')
            patterns_by_type[pattern_type].append(pattern)
        
        for pattern_type, type_patterns in patterns_by_type.items():
            if pattern_type not in self.PATTERN_TYPE_TABLES:
                logger.warning(f"Unknown pattern type '{pattern_type}', skipping {len(type_patterns)} patterns")
                continue
            
            table_name = self.PATTERN_TYPE_TABLES[pattern_type]
            logger.info(f"Inserting {len(type_patterns)} patterns of type '{pattern_type}' into {table_name}")
            
            column_names = self._get_pattern_columns(pattern_type)
            
            for i in range(0, len(type_patterns), batch_size):
                batch = type_patterns[i:i + batch_size]
                batch_data = []
                
                for pattern in batch:
                    row = self._build_pattern_row(pattern, pattern_type, window_days, date_obj)
                    batch_data.append(row)
                
                if batch_data:
                    self.client.insert(
                        table_name,
                        batch_data,
                        column_names=column_names
                    )
        
        total_patterns = sum(len(p) for p in patterns_by_type.values())
        logger.info(f"Inserted {total_patterns} patterns across {len(patterns_by_type)} types")
    
    def _get_pattern_columns(self, pattern_type: str) -> List[str]:
        common_columns = [
            'window_days', 'processing_date',
            'pattern_id', 'pattern_type', 'pattern_hash',
            'addresses_involved', 'address_roles',
        ]
        
        if pattern_type == PatternTypes.CYCLE:
            type_columns = ['cycle_path', 'cycle_length', 'cycle_volume_usd']
        elif pattern_type == PatternTypes.LAYERING_PATH:
            type_columns = ['layering_path', 'path_depth', 'path_volume_usd',
                          'source_address', 'destination_address']
        elif pattern_type == PatternTypes.SMURFING_NETWORK:
            type_columns = ['network_members', 'network_size', 'network_density', 'hub_addresses']
        elif pattern_type == PatternTypes.PROXIMITY_RISK:
            type_columns = ['risk_source_address', 'distance_to_risk']
        elif pattern_type in [PatternTypes.MOTIF_FANIN, PatternTypes.MOTIF_FANOUT]:
            type_columns = ['motif_type', 'motif_center_address', 'motif_participant_count']
        elif pattern_type == PatternTypes.TEMPORAL_BURST:
            type_columns = [
                'burst_address', 'burst_start_timestamp', 'burst_end_timestamp',
                'burst_duration_seconds', 'burst_transaction_count', 'burst_volume_usd',
                'normal_tx_rate', 'burst_tx_rate', 'burst_intensity', 'z_score',
                'hourly_distribution', 'peak_hours',
            ]
        elif pattern_type == PatternTypes.THRESHOLD_EVASION:
            type_columns = [
                'primary_address', 'threshold_value', 'threshold_type',
                'transactions_near_threshold', 'avg_transaction_size', 'max_transaction_size',
                'size_consistency', 'clustering_score',
                'unique_days', 'avg_daily_transactions', 'temporal_spread_score',
                'threshold_avoidance_score',
            ]
        else:
            type_columns = []
        
        trailing_columns = [
            'detection_timestamp', 'pattern_start_time', 'pattern_end_time',
            'pattern_duration_hours', 'evidence_transaction_count', 'evidence_volume_usd',
            'detection_method', '_version'
        ]
        
        return common_columns + type_columns + trailing_columns
    
    def _build_pattern_row(
        self,
        pattern: Dict[str, Any],
        pattern_type: str,
        window_days: int,
        date_obj
    ) -> List[Any]:
        row = [
            window_days,
            date_obj,
            pattern['pattern_id'],
            pattern['pattern_type'],
            pattern['pattern_hash'],
            pattern['addresses_involved'],
            pattern['address_roles'],
        ]
        
        if pattern_type == PatternTypes.CYCLE:
            row.extend([
                pattern.get('cycle_path', []),
                int(pattern.get('cycle_length', 0)),
                str(pattern.get('cycle_volume_usd', 0)),
            ])
        elif pattern_type == PatternTypes.LAYERING_PATH:
            row.extend([
                pattern.get('layering_path', []),
                int(pattern.get('path_depth', 0)),
                str(pattern.get('path_volume_usd', 0)),
                pattern.get('source_address', ''),
                pattern.get('destination_address', ''),
            ])
        elif pattern_type == PatternTypes.SMURFING_NETWORK:
            row.extend([
                pattern.get('network_members', []),
                int(pattern.get('network_size', 0)),
                float(pattern.get('network_density', 0.0)),
                pattern.get('hub_addresses', []),
            ])
        elif pattern_type == PatternTypes.PROXIMITY_RISK:
            row.extend([
                pattern.get('risk_source_address', ''),
                int(pattern.get('distance_to_risk', 0)),
            ])
        elif pattern_type in [PatternTypes.MOTIF_FANIN, PatternTypes.MOTIF_FANOUT]:
            row.extend([
                pattern.get('motif_type', ''),
                pattern.get('motif_center_address', ''),
                int(pattern.get('motif_participant_count', 0)),
            ])
        elif pattern_type == PatternTypes.TEMPORAL_BURST:
            row.extend([
                pattern.get('burst_address', ''),
                int(pattern.get('burst_start_timestamp', 0)),
                int(pattern.get('burst_end_timestamp', 0)),
                int(pattern.get('burst_duration_seconds', 0)),
                int(pattern.get('burst_transaction_count', 0)),
                str(pattern.get('burst_volume_usd', 0)),
                float(pattern.get('normal_tx_rate', 0.0)),
                float(pattern.get('burst_tx_rate', 0.0)),
                float(pattern.get('burst_intensity', 0.0)),
                float(pattern.get('z_score', 0.0)),
                pattern.get('hourly_distribution', []),
                pattern.get('peak_hours', []),
            ])
        elif pattern_type == PatternTypes.THRESHOLD_EVASION:
            row.extend([
                pattern.get('primary_address', ''),
                str(pattern.get('threshold_value', 0)),
                pattern.get('threshold_type', ''),
                int(pattern.get('transactions_near_threshold', 0)),
                str(pattern.get('avg_transaction_size', 0)),
                str(pattern.get('max_transaction_size', 0)),
                float(pattern.get('size_consistency', 0.0)),
                float(pattern.get('clustering_score', 0.0)),
                int(pattern.get('unique_days', 0)),
                float(pattern.get('avg_daily_transactions', 0.0)),
                float(pattern.get('temporal_spread_score', 0.0)),
                float(pattern.get('threshold_avoidance_score', 0.0)),
            ])
        
        row.extend([
            int(pattern.get('detection_timestamp', int(time.time()))),
            int(pattern.get('pattern_start_time', 0)),
            int(pattern.get('pattern_end_time', 0)),
            int(pattern.get('pattern_duration_hours', 0)),
            int(pattern.get('evidence_transaction_count', 0)),
            str(pattern.get('evidence_volume_usd', 0)),
            pattern.get('detection_method', 'SCC_ANALYSIS'),
            _generate_version(),
        ])
        
        return row
    
    def delete_features_partition(self, window_days: int, processing_date: str) -> None:
        """Delete only features for given window_days and processing_date."""
        date_obj = datetime.strptime(processing_date, '%Y-%m-%d').date()
        
        params = {
            'window_days': window_days,
            'processing_date': date_obj
        }
        
        query = """
        ALTER TABLE analyzers_features
        DELETE WHERE window_days = %(window_days)s AND processing_date = %(processing_date)s
        """
        self.client.command(query, parameters=params)
        logger.info(f"Deleted features partition for window_days={window_days}, processing_date={processing_date}")
    
    def delete_patterns_partition(self, window_days: int, processing_date: str) -> None:
        """Delete only patterns for given window_days and processing_date."""
        date_obj = datetime.strptime(processing_date, '%Y-%m-%d').date()
        
        params = {
            'window_days': window_days,
            'processing_date': date_obj
        }
        
        unique_tables = set(self.PATTERN_TYPE_TABLES.values())
        for table_name in unique_tables:
            query = f"""
            ALTER TABLE {table_name}
            DELETE WHERE window_days = %(window_days)s AND processing_date = %(processing_date)s
            """
            self.client.command(query, parameters=params)
            logger.info(f"Deleted patterns partition from {table_name}")
