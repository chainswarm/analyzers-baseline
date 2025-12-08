import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from decimal import Decimal
from collections import defaultdict

import pandas as pd
from loguru import logger

from chainswarm_core.constants import PatternTypes


class ParquetAdapter:
    PATTERN_TYPE_FILES = {
        PatternTypes.CYCLE: 'patterns_cycle.parquet',
        PatternTypes.LAYERING_PATH: 'patterns_layering.parquet',
        PatternTypes.SMURFING_NETWORK: 'patterns_network.parquet',
        PatternTypes.PROXIMITY_RISK: 'patterns_proximity.parquet',
        PatternTypes.MOTIF_FANIN: 'patterns_motif.parquet',
        PatternTypes.MOTIF_FANOUT: 'patterns_motif.parquet',
        PatternTypes.TEMPORAL_BURST: 'patterns_burst.parquet',
        PatternTypes.THRESHOLD_EVASION: 'patterns_threshold.parquet',
    }
    
    def __init__(self, input_path: Path, output_path: Path):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        
        # Create output directory if it doesn't exist
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Cache for loaded dataframes
        self._transfers_df: Optional[pd.DataFrame] = None
        self._labels_df: Optional[pd.DataFrame] = None
        self._assets_df: Optional[pd.DataFrame] = None
        self._prices_df: Optional[pd.DataFrame] = None
    
    def read_transfers(
        self,
        start_timestamp_ms: int,
        end_timestamp_ms: int
    ) -> List[Dict[str, Any]]:
        transfers_file = self.input_path / "transfers.parquet"
        
        if not transfers_file.exists():
            raise ValueError(f"Transfers file not found: {transfers_file}")
        
        if self._transfers_df is None:
            logger.info(f"Loading transfers from {transfers_file}")
            self._transfers_df = pd.read_parquet(transfers_file, engine='pyarrow')
            logger.info(f"Loaded {len(self._transfers_df)} transfers")
        
        df = self._transfers_df
        
        mask = (df['block_timestamp'] >= start_timestamp_ms) & \
               (df['block_timestamp'] < end_timestamp_ms)
        filtered_df = df[mask]
        
        logger.info(f"Filtered to {len(filtered_df)} transfers in time range")
        
        return filtered_df.to_dict(orient='records')
    
    def read_money_flows(
        self,
        start_timestamp_ms: int,
        end_timestamp_ms: int
    ) -> List[Dict[str, Any]]:
        transfers_file = self.input_path / "transfers.parquet"
        
        if not transfers_file.exists():
            raise ValueError(f"Transfers file not found: {transfers_file}")
        
        if self._transfers_df is None:
            logger.info(f"Loading transfers from {transfers_file}")
            self._transfers_df = pd.read_parquet(transfers_file, engine='pyarrow')
            logger.info(f"Loaded {len(self._transfers_df)} transfers")
        
        df = self._transfers_df
        
        mask = (df['block_timestamp'] >= start_timestamp_ms) & \
               (df['block_timestamp'] < end_timestamp_ms)
        filtered_df = df[mask]
        
        if filtered_df.empty:
            raise ValueError(f"No transfers in time range {start_timestamp_ms} to {end_timestamp_ms}")
        
        grouped = filtered_df.groupby(
            ['from_address', 'to_address'],
            as_index=False
        ).agg({
            'amount_usd': 'sum',
            'tx_id': 'count'
        })
        
        grouped = grouped.rename(columns={
            'amount_usd': 'amount_usd_sum',
            'tx_id': 'tx_count'
        })
        
        logger.info(f"Aggregated to {len(grouped)} money flows")
        
        return grouped.to_dict(orient='records')
    
    def read_assets(self, network: Optional[str] = None) -> List[Dict[str, Any]]:
        assets_file = self.input_path / "assets.parquet"
        
        if not assets_file.exists():
            raise ValueError(f"Assets file not found: {assets_file}")
        
        if self._assets_df is None:
            logger.info(f"Loading assets from {assets_file}")
            self._assets_df = pd.read_parquet(assets_file, engine='pyarrow')
            logger.info(f"Loaded {len(self._assets_df)} assets")
        
        df = self._assets_df
        
        if network:
            df = df[df['network'] == network]
        
        logger.info(f"Returning {len(df)} assets")
        
        return df.to_dict(orient='records')
    
    def read_asset_prices(
        self,
        start_timestamp_ms: int,
        end_timestamp_ms: int
    ) -> List[Dict[str, Any]]:
        prices_file = self.input_path / "asset_prices.parquet"
        
        if not prices_file.exists():
            raise ValueError(f"Asset prices file not found: {prices_file}")
        
        if self._prices_df is None:
            logger.info(f"Loading asset prices from {prices_file}")
            self._prices_df = pd.read_parquet(prices_file, engine='pyarrow')
            logger.info(f"Loaded {len(self._prices_df)} asset price records")
        
        df = self._prices_df
        
        start_date = pd.Timestamp(start_timestamp_ms, unit='ms')
        end_date = pd.Timestamp(end_timestamp_ms, unit='ms')
        
        if 'price_date' in df.columns:
            price_dates = pd.to_datetime(df['price_date'])
            mask = (price_dates >= start_date) & (price_dates < end_date)
            filtered_df = df[mask]
        else:
            filtered_df = df
        
        logger.info(f"Returning {len(filtered_df)} asset prices in time range")
        
        return filtered_df.to_dict(orient='records')
    
    def read_address_labels(
        self,
        addresses: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        labels_file = self.input_path / "address_labels.parquet"
        
        if not labels_file.exists():
            logger.info(f"Address labels file not found: {labels_file}, returning empty labels")
            return {}
        
        if self._labels_df is None:
            logger.info(f"Loading address labels from {labels_file}")
            self._labels_df = pd.read_parquet(labels_file, engine='pyarrow')
            logger.info(f"Loaded {len(self._labels_df)} address labels")
        
        df = self._labels_df
        
        mask = df['address'].isin(addresses)
        filtered_df = df[mask]
        
        result = {}
        for _, row in filtered_df.iterrows():
            address = row['address']
            result[address] = {
                'label': row.get('label', ''),
                'address_type': row.get('address_type', ''),
                'trust_level': row.get('trust_level', ''),
                'source': row.get('source', ''),
            }
        
        logger.info(f"Found labels for {len(result)} of {len(addresses)} addresses")
        
        return result
    
    LIST_COLUMNS = ['hourly_activity', 'daily_activity']
    
    def _convert_value(self, value: Any, as_json: bool = False) -> Any:
        if isinstance(value, Decimal):
            return float(value)
        if isinstance(value, list):
            if as_json:
                return json.dumps(value)
            return [self._convert_value(v) for v in value]
        if isinstance(value, dict):
            return {k: self._convert_value(v) for k, v in value.items()}
        return value
    
    def _normalize_features(self, features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized = []
        for feature in features:
            normalized_feature = {}
            for key, value in feature.items():
                is_list_col = key in self.LIST_COLUMNS
                normalized_feature[key] = self._convert_value(value, as_json=is_list_col)
            normalized.append(normalized_feature)
        return normalized
    
    def write_features(
        self,
        features: List[Dict[str, Any]],
        window_days: int,
        processing_date: str
    ) -> None:
        if not features:
            raise ValueError("No features to write")
        
        date_obj = datetime.strptime(processing_date, '%Y-%m-%d').date()
        
        for feature in features:
            feature['window_days'] = window_days
            feature['processing_date'] = date_obj
        
        features = self._normalize_features(features)
        
        output_file = self.output_path / "features.parquet"
        
        df = pd.DataFrame(features)
        df.to_parquet(output_file, engine='pyarrow', index=False)
        
        logger.info(f"Wrote {len(features)} features to {output_file}")
    
    def write_patterns(
        self,
        patterns: List[Dict[str, Any]],
        window_days: int,
        processing_date: str
    ) -> None:
        if not patterns:
            logger.info("No patterns to write")
            return
        
        date_obj = datetime.strptime(processing_date, '%Y-%m-%d').date()
        
        for pattern in patterns:
            pattern['window_days'] = window_days
            pattern['processing_date'] = date_obj
        
        patterns_by_type: Dict[str, List[Dict]] = defaultdict(list)
        for pattern in patterns:
            pattern_type = pattern.get('pattern_type', 'UNKNOWN')
            patterns_by_type[pattern_type].append(pattern)
        
        for pattern_type, type_patterns in patterns_by_type.items():
            if pattern_type not in self.PATTERN_TYPE_FILES:
                logger.warning(f"Unknown pattern type: {pattern_type}, skipping {len(type_patterns)} patterns")
                continue
            
            output_filename = self.PATTERN_TYPE_FILES[pattern_type]
            output_file = self.output_path / output_filename
            
            df = pd.DataFrame(type_patterns)
            
            if output_file.exists() and pattern_type in [PatternTypes.MOTIF_FANIN, PatternTypes.MOTIF_FANOUT]:
                existing_df = pd.read_parquet(output_file, engine='pyarrow')
                df = pd.concat([existing_df, df], ignore_index=True)
            
            df.to_parquet(output_file, engine='pyarrow', index=False)
            
            logger.info(f"Wrote {len(type_patterns)} {pattern_type} patterns to {output_file}")
        
        total_patterns = sum(len(p) for p in patterns_by_type.values())
        logger.info(f"Total patterns written: {total_patterns} across {len(patterns_by_type)} types")
    
    def clear_cache(self) -> None:
        self._transfers_df = None
        self._labels_df = None
        self._assets_df = None
        self._prices_df = None
        logger.debug("Cleared Parquet adapter cache")