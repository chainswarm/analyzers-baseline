from typing import Dict, List, Any, Set
from collections import defaultdict
from datetime import datetime, timezone
import numpy as np


def compute_transfer_aggregates(
    transfers: List[Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    if not transfers:
        return {}
    
    address_transfers: Dict[str, List[Dict]] = defaultdict(list)
    for tx in transfers:
        from_addr = tx.get('from_address')
        to_addr = tx.get('to_address')
        if from_addr:
            address_transfers[from_addr].append(tx)
        if to_addr and to_addr != from_addr:
            address_transfers[to_addr].append(tx)
    
    result = {}
    for address, txs in address_transfers.items():
        result[address] = _compute_address_aggregates(address, txs)
    
    return result


def _compute_address_aggregates(
    address: str,
    transfers: List[Dict[str, Any]]
) -> Dict[str, Any]:
    temporal_patterns = _compute_temporal_patterns(address, transfers)
    temporal_summaries = _compute_temporal_summaries(address, transfers)
    behavioral_counters = _compute_behavioral_counters(address, transfers)
    hourly_volumes = _compute_hourly_volumes(address, transfers)
    interevent_stats = _compute_interevent_stats(address, transfers)
    amount_moments = _compute_amount_moments(address, transfers)
    reciprocity_stats = _compute_reciprocity_stats(address, transfers)
    
    return {
        'temporal_patterns': temporal_patterns,
        'temporal_summaries': temporal_summaries,
        'behavioral_counters': behavioral_counters,
        'hourly_volumes': hourly_volumes,
        'interevent_stats': interevent_stats,
        'amount_moments': amount_moments,
        'reciprocity_stats': reciprocity_stats,
    }


def _compute_temporal_patterns(
    address: str,
    transfers: List[Dict[str, Any]]
) -> Dict[str, Any]:
    hourly_activity = [0] * 24
    daily_activity = [0] * 7
    
    for tx in transfers:
        ts_ms = tx.get('block_timestamp', 0)
        if ts_ms <= 0:
            continue
        
        dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
        hour = dt.hour
        weekday = dt.weekday()
        
        hourly_activity[hour] += 1
        daily_activity[weekday] += 1
    
    peak_activity_hour = int(np.argmax(hourly_activity)) if sum(hourly_activity) > 0 else 0
    peak_activity_day = int(np.argmax(daily_activity)) if sum(daily_activity) > 0 else 0
    
    return {
        'hourly_activity': hourly_activity,
        'daily_activity': daily_activity,
        'peak_activity_hour': peak_activity_hour,
        'peak_activity_day': peak_activity_day,
    }


def _compute_temporal_summaries(
    address: str,
    transfers: List[Dict[str, Any]]
) -> Dict[str, Any]:
    if not transfers:
        return {
            'first_timestamp': 0,
            'last_timestamp': 0,
            'total_tx_count': 0,
            'distinct_activity_days': 0,
            'total_volume': 0.0,
            'weekend_tx_count': 0,
            'night_tx_count': 0,
        }
    
    timestamps = []
    activity_dates: Set[str] = set()
    total_volume = 0.0
    weekend_tx_count = 0
    night_tx_count = 0
    
    for tx in transfers:
        ts_ms = tx.get('block_timestamp', 0)
        if ts_ms <= 0:
            continue
        
        timestamps.append(ts_ms)
        
        dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
        activity_dates.add(dt.strftime('%Y-%m-%d'))
        
        hour = dt.hour
        weekday = dt.weekday()
        
        if weekday >= 5:
            weekend_tx_count += 1
        if hour <= 5:
            night_tx_count += 1
        
        amount = float(tx.get('amount', 0) or tx.get('amount_usd', 0) or 0)
        total_volume += abs(amount)
    
    return {
        'first_timestamp': min(timestamps) if timestamps else 0,
        'last_timestamp': max(timestamps) if timestamps else 0,
        'total_tx_count': len(transfers),
        'distinct_activity_days': len(activity_dates),
        'total_volume': total_volume,
        'weekend_tx_count': weekend_tx_count,
        'night_tx_count': night_tx_count,
    }


def _compute_behavioral_counters(
    address: str,
    transfers: List[Dict[str, Any]]
) -> Dict[str, Any]:
    total_tx_pos_amount = 0
    round_number_count = 0
    small_amount_count = 0
    unusual_tx_count = 0
    
    for tx in transfers:
        amount = float(tx.get('amount', 0) or tx.get('amount_usd', 0) or 0)
        
        if amount <= 0:
            continue
        
        total_tx_pos_amount += 1
        
        if int(amount) % 100 == 0 and amount >= 100:
            round_number_count += 1
        
        if amount < 1000:
            small_amount_count += 1
        
        ts_ms = tx.get('block_timestamp', 0)
        if ts_ms > 0:
            dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
            hour = dt.hour
            weekday = dt.weekday()
            
            if hour <= 5 or weekday >= 5:
                unusual_tx_count += 1
    
    return {
        'total_tx_pos_amount': total_tx_pos_amount,
        'round_number_count': round_number_count,
        'small_amount_count': small_amount_count,
        'unusual_tx_count': unusual_tx_count,
    }


def _compute_hourly_volumes(
    address: str,
    transfers: List[Dict[str, Any]]
) -> List[float]:
    hourly_volumes = [0.0] * 24
    
    for tx in transfers:
        amount = float(tx.get('amount', 0) or tx.get('amount_usd', 0) or 0)
        if amount <= 0:
            continue
        
        ts_ms = tx.get('block_timestamp', 0)
        if ts_ms <= 0:
            continue
        
        dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
        hour = dt.hour
        hourly_volumes[hour] += abs(amount)
    
    return hourly_volumes


def _compute_interevent_stats(
    address: str,
    transfers: List[Dict[str, Any]]
) -> Dict[str, Any]:
    timestamps = []
    for tx in transfers:
        ts_ms = tx.get('block_timestamp', 0)
        if ts_ms > 0:
            timestamps.append(ts_ms)
    
    if len(timestamps) < 2:
        return {
            'mean_inter_s': 0.0,
            'std_inter_s': 0.0,
            'n': 0,
        }
    
    timestamps.sort()
    diffs_seconds = []
    for i in range(1, len(timestamps)):
        diff_s = (timestamps[i] - timestamps[i-1]) / 1000.0
        diffs_seconds.append(diff_s)
    
    n = len(diffs_seconds)
    mean_inter = float(np.mean(diffs_seconds)) if n > 0 else 0.0
    std_inter = float(np.std(diffs_seconds, ddof=1)) if n > 1 else 0.0
    
    return {
        'mean_inter_s': mean_inter,
        'std_inter_s': std_inter,
        'n': n,
    }


def _compute_amount_moments(
    address: str,
    transfers: List[Dict[str, Any]]
) -> Dict[str, Any]:
    amounts = []
    for tx in transfers:
        amount = float(tx.get('amount', 0) or tx.get('amount_usd', 0) or 0)
        if amount > 0:
            amounts.append(amount)
    
    n = len(amounts)
    if n == 0:
        return {'n': 0, 's1': 0.0, 's2': 0.0, 's3': 0.0, 's4': 0.0}
    
    s1 = sum(amounts)
    s2 = sum(a * a for a in amounts)
    s3 = sum(a ** 3 for a in amounts)
    s4 = sum(a ** 4 for a in amounts)
    
    return {
        'n': n,
        's1': s1,
        's2': s2,
        's3': s3,
        's4': s4,
    }


def _compute_reciprocity_stats(
    address: str,
    transfers: List[Dict[str, Any]]
) -> Dict[str, Any]:
    counterparties_in: Dict[str, float] = defaultdict(float)
    counterparties_out: Dict[str, float] = defaultdict(float)
    
    for tx in transfers:
        from_addr = tx.get('from_address')
        to_addr = tx.get('to_address')
        amount = float(tx.get('amount', 0) or tx.get('amount_usd', 0) or 0)
        
        if amount <= 0:
            continue
        
        if to_addr == address and from_addr:
            counterparties_in[from_addr] += amount
        elif from_addr == address and to_addr:
            counterparties_out[to_addr] += amount
    
    in_set = set(counterparties_in.keys())
    out_set = set(counterparties_out.keys())
    reciprocal_counterparties = in_set & out_set
    
    total_volume = sum(counterparties_in.values()) + sum(counterparties_out.values())
    reciprocal_volume = 0.0
    for cp in reciprocal_counterparties:
        reciprocal_volume += counterparties_in.get(cp, 0) + counterparties_out.get(cp, 0)
    
    return {
        'total_volume': total_volume,
        'reciprocal_volume': reciprocal_volume,
    }