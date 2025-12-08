from typing import Protocol, Dict, List, Any

import networkx as nx


class FeatureAnalyzer(Protocol):
    
    def analyze(
        self,
        graph: nx.DiGraph,
        address_labels: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        ...


class PatternAnalyzer(Protocol):
    
    def analyze(
        self,
        graph: nx.DiGraph,
        address_labels: Dict[str, Dict[str, Any]],
        config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        ...