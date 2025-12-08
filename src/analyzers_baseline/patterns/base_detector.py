from abc import ABC, abstractmethod
from typing import Dict, List, Any
import hashlib
import uuid

import networkx as nx
from loguru import logger

from chainswarm_core.constants import (
    PatternTypes,
    DetectionMethods,
    AddressTypes,
    TrustLevels,
)
from chainswarm_core.constants.risk import Severities

PatternType = PatternTypes
DetectionMethod = DetectionMethods
Severity = Severities
AddressType = AddressTypes
TrustLevel = TrustLevels


def generate_pattern_hash(pattern_type: str, addresses: List[str]) -> str:
    sorted_addresses = sorted(addresses)
    content = f"{pattern_type}:{','.join(sorted_addresses)}"
    return hashlib.md5(content.encode()).hexdigest()


def generate_pattern_id(pattern_type: str, pattern_hash: str) -> str:
    return f"{pattern_type}_{pattern_hash[:12]}_{uuid.uuid4().hex[:8]}"


class BasePatternDetector(ABC):

    def __init__(
        self,
        config: Dict[str, Any] = None,
        address_labels_cache: Dict[str, Dict[str, Any]] = None,
        network: str = None
    ):
        self.config = config or {}
        self._address_labels_cache = address_labels_cache or {}
        self.network = network
        logger.debug(f"Initialized {self.__class__.__name__} for network={network}")

    @property
    @abstractmethod
    def pattern_type(self) -> str:
        ...

    @abstractmethod
    def detect(
        self,
        G: nx.DiGraph,
        address_labels: Dict[str, Dict[str, Any]],
        window_days: int,
        processing_date: str
    ) -> List[Dict[str, Any]]:
        ...

    def _get_config_value(
        self,
        section: str,
        key: str,
        default: Any = None
    ) -> Any:
        if section not in self.config:
            return default
        
        section_config = self.config[section]
        
        if self.network and 'network_overrides' in section_config:
            network_overrides = section_config['network_overrides']
            if self.network in network_overrides:
                network_config = network_overrides[self.network]
                if key in network_config:
                    return network_config[key]
        
        return section_config.get(key, default)

    def _is_trusted_address(self, address: str) -> bool:
        label_info = self._address_labels_cache.get(address)
        if not label_info:
            return False
            
        trust_level = label_info.get('trust_level')
        address_type = label_info.get('address_type')
        
        safe_trust_levels = [TrustLevel.VERIFIED, TrustLevel.OFFICIAL]
        safe_address_types = [
            AddressType.EXCHANGE,
            AddressType.INSTITUTIONAL,
            AddressType.STAKING,
            AddressType.VALIDATOR,
        ]
        
        return (trust_level in safe_trust_levels and
                address_type in safe_address_types)

    def _is_fraudulent_address(self, address: str) -> bool:
        label_info = self._address_labels_cache.get(address)
        if not label_info:
            return False
            
        trust_level = label_info.get('trust_level')
        address_type = label_info.get('address_type')
        
        fraudulent_address_types = [
            AddressType.MIXER,
            AddressType.SCAM,
            AddressType.DARK_MARKET,
            AddressType.SANCTIONED
        ]
        
        return (address_type in fraudulent_address_types or
                trust_level == TrustLevel.BLACKLISTED)

    def _get_address_context(self, address: str) -> Dict[str, Any]:
        label_info = self._address_labels_cache.get(address, {})
        
        return {
            'trust_level': label_info.get('trust_level', TrustLevel.UNVERIFIED),
            'address_type': label_info.get('address_type', AddressType.UNKNOWN),
            'is_trusted': self._is_trusted_address(address),
            'is_fraudulent': self._is_fraudulent_address(address)
        }

    def _create_base_pattern(
        self,
        addresses_involved: List[str],
        address_roles: Dict[str, str],
        evidence_volume_usd: float,
        evidence_tx_count: int,
        detection_method: str,
        window_days: int,
        processing_date: str,
        confidence_score: float = 0.5,
        severity: str = Severity.MEDIUM
    ) -> Dict[str, Any]:
        sorted_addresses = sorted(addresses_involved)
        pattern_hash = generate_pattern_hash(self.pattern_type, sorted_addresses)
        pattern_id = generate_pattern_id(self.pattern_type, pattern_hash)
        
        return {
            'pattern_id': pattern_id,
            'pattern_type': self.pattern_type,
            'pattern_hash': pattern_hash,
            'addresses_involved': sorted_addresses,
            'address_roles': address_roles,
            'transaction_ids': [],
            'total_amount_usd': evidence_volume_usd,
            'detection_method': detection_method,
            'confidence_score': confidence_score,
            'severity': severity,
            'evidence': {},
            'window_days': window_days,
            'processing_date': processing_date,
            'network': self.network or '',
        }