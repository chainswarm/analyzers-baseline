from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, List, Any


@dataclass
class Transfer:
    tx_id: str
    event_index: str
    edge_index: str
    block_height: int
    block_timestamp: int
    from_address: str
    to_address: str
    asset_symbol: str
    asset_contract: str
    amount: Decimal
    amount_usd: Decimal
    fee: Decimal


@dataclass
class MoneyFlow:
    from_address: str
    to_address: str
    amount_usd_sum: Decimal
    tx_count: int
    first_seen_timestamp: int
    last_seen_timestamp: int
    active_days: int
    avg_tx_size_usd: Decimal
    unique_assets: int
    dominant_asset: str
    hourly_pattern: List[int]
    weekly_pattern: List[int]
    reciprocity_ratio: float
    is_bidirectional: bool


@dataclass
class AddressLabel:
    address: str
    label: str
    address_type: str
    trust_level: str
    source: str


FeatureDict = Dict[str, Any]
AddressFeatures = Dict[str, FeatureDict]
PatternDict = Dict[str, Any]
PatternList = List[PatternDict]
