from typing import Protocol, Dict, List, Any, Optional


class InputAdapter(Protocol):
    
    def read_transfers(
        self,
        start_timestamp_ms: int,
        end_timestamp_ms: int
    ) -> List[Dict[str, Any]]:
        ...
    
    def read_money_flows(
        self,
        start_timestamp_ms: int,
        end_timestamp_ms: int
    ) -> List[Dict[str, Any]]:
        ...
    
    def read_assets(self, network: Optional[str] = None) -> List[Dict[str, Any]]:
        ...
    
    def read_asset_prices(
        self,
        start_timestamp_ms: int,
        end_timestamp_ms: int
    ) -> List[Dict[str, Any]]:
        ...
    
    def read_address_labels(
        self,
        addresses: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        ...


class OutputAdapter(Protocol):
    
    def write_features(
        self,
        features: List[Dict[str, Any]],
        window_days: int,
        processing_date: str
    ) -> None:
        ...
    
    def write_patterns(
        self,
        patterns: List[Dict[str, Any]],
        window_days: int,
        processing_date: str
    ) -> None:
        ...