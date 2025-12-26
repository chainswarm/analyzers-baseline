from typing import List, Dict, Any
from collections import defaultdict

import networkx as nx
from loguru import logger


def build_money_flow_graph(money_flows: List[Dict[str, Any]]) -> nx.DiGraph:
    G = nx.DiGraph()
    
    for flow in money_flows:
        from_addr = flow['from_address']
        to_addr = flow['to_address']
        amount_usd_sum = float(flow['amount_usd_sum'])
        tx_count = int(flow['tx_count'])
        first_seen = int(flow['first_seen_timestamp'])
        last_seen = int(flow['last_seen_timestamp'])
        active_days = int(flow['active_days'])
        avg_tx_size = float(flow['avg_tx_size_usd'])
        unique_assets = int(flow['unique_assets'])
        dominant_asset = flow['dominant_asset']
        hourly_pattern = flow['hourly_pattern']
        weekly_pattern = flow['weekly_pattern']
        reciprocity_ratio = float(flow['reciprocity_ratio'])
        is_bidirectional = bool(flow['is_bidirectional'])
        
        G.add_edge(
            from_addr,
            to_addr,
            weight=amount_usd_sum,
            amount_usd_sum=amount_usd_sum,
            tx_count=tx_count,
            first_seen_timestamp=first_seen,
            last_seen_timestamp=last_seen,
            active_days=active_days,
            relationship_age_days=(last_seen - first_seen) / 86400000 if first_seen > 0 else 0,
            avg_tx_size_usd=avg_tx_size,
            unique_assets=unique_assets,
            dominant_asset=dominant_asset,
            hourly_pattern=hourly_pattern,
            weekly_pattern=weekly_pattern,
            reciprocity_ratio=reciprocity_ratio,
            is_bidirectional=is_bidirectional
        )
    
    logger.info(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def add_node_volume_attributes(G: nx.DiGraph) -> None:
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


def extract_addresses_from_flows(flows: List[Dict[str, Any]]) -> List[str]:
    addresses_set = set()
    for flow in flows:
        addresses_set.add(flow['from_address'])
        addresses_set.add(flow['to_address'])
    return sorted(list(addresses_set))


def build_flows_index_by_address(G: nx.DiGraph) -> Dict[str, List[Dict[str, Any]]]:
    flows_by_address: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    
    for u, v, data in G.edges(data=True):
        flow = {
            'from_address': u,
            'to_address': v,
            'amount_usd_sum': data.get('amount_usd_sum', 0.0),
            'tx_count': int(data.get('tx_count', 0))
        }
        flows_by_address[u].append(flow)
        flows_by_address[v].append(flow)
    
    return dict(flows_by_address)
