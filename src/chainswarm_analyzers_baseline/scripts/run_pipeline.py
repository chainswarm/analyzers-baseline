import argparse
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)
os.chdir(project_root)

from dotenv import load_dotenv
from loguru import logger
from chainswarm_core.observability import setup_logger

from chainswarm_analyzers_baseline.adapters.parquet import ParquetAdapter
from chainswarm_analyzers_baseline.config import SettingsLoader
from chainswarm_analyzers_baseline.pipeline import create_pipeline


def parse_path_metadata(input_path: Path) -> dict:
    path_str = str(input_path.resolve())
    
    pattern = r"[/\\]input[/\\]([^/\\]+)[/\\](\d{4}-\d{2}-\d{2})[/\\](\d+)/?$"
    match = re.search(pattern, path_str)
    
    if not match:
        raise ValueError(
            f"Cannot parse metadata from path: {input_path}. "
            f"Expected format: .../input/{{network}}/{{YYYY-MM-DD}}/{{window_days}}/"
        )
    
    return {
        "network": match.group(1),
        "processing_date": match.group(2),
        "window_days": int(match.group(3)),
    }


def construct_output_path(input_path: Path) -> Path:
    path_str = str(input_path.resolve())
    output_str = re.sub(r"[/\\]input[/\\]", "/output/", path_str)
    return Path(output_str)


def main():
    load_dotenv()
    
    parser = argparse.ArgumentParser(
        description="Run full analysis pipeline on blockchain transaction data"
    )
    
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input directory (e.g., data/input/torus/2025-11-20/300)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (auto-constructed from input if not provided)"
    )
    
    parser.add_argument(
        "--window-days",
        type=int,
        default=None,
        help="Analysis window in days (extracted from path if not provided)"
    )
    parser.add_argument(
        "--processing-date",
        type=str,
        default=None,
        help="Processing date YYYY-MM-DD (extracted from path if not provided)"
    )
    parser.add_argument(
        "--network",
        type=str,
        default=None,
        help="Network name (extracted from path if not provided)"
    )
    
    parser.add_argument(
        "--features-only",
        action="store_true",
        help="Run only feature analysis"
    )
    parser.add_argument(
        "--patterns-only",
        action="store_true",
        help="Run only pattern detection"
    )
    
    parser.add_argument(
        "--clickhouse",
        action="store_true",
        help="Use ClickHouse instead of Parquet (uses CLICKHOUSE_* env vars)"
    )
    parser.add_argument(
        "--database-prefix",
        type=str,
        default="analytics",
        help="Database prefix for ClickHouse (default: analytics -> analytics_{network})"
    )
    
    args = parser.parse_args()
    
    network_hint = args.network or "baseline"
    setup_logger(f"analyzers-baseline-{network_hint}")
    
    logger.info("Starting analysis pipeline")
    
    if args.features_only and args.patterns_only:
        raise ValueError("Cannot use both --features-only and --patterns-only")
    
    run_features = not args.patterns_only
    run_patterns = not args.features_only
    
    mode_desc = "full pipeline"
    if args.features_only:
        mode_desc = "features only"
    elif args.patterns_only:
        mode_desc = "patterns only"
    
    logger.info(f"Running {mode_desc}")
    
    network: str
    processing_date: str
    window_days: int
    
    if args.input and not args.clickhouse:
        path_metadata = parse_path_metadata(args.input)
        network = args.network or path_metadata["network"]
        processing_date = args.processing_date or path_metadata["processing_date"]
        window_days = args.window_days if args.window_days is not None else path_metadata["window_days"]
        logger.info(f"Extracted from path: network={network}, date={processing_date}, window={window_days}")
    else:
        network = args.network or "unknown"
        if args.processing_date:
            processing_date = args.processing_date
        else:
            processing_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        window_days = args.window_days if args.window_days is not None else 30
    
    processing_dt = datetime.strptime(processing_date, "%Y-%m-%d")
    
    logger.info(f"Processing date: {processing_date}, window: {window_days} days, network: {network}")
    
    end_timestamp_ms = int(processing_dt.timestamp() * 1000)
    start_timestamp_ms = int(
        (processing_dt - timedelta(days=window_days)).timestamp() * 1000
    )
    
    logger.info(
        f"Time window: {window_days} days "
        f"({start_timestamp_ms} to {end_timestamp_ms})"
    )
    
    settings_loader = SettingsLoader()
    
    if args.clickhouse:
        adapter = _create_clickhouse_adapter(args, network)
    else:
        adapter = _create_parquet_adapter(args)
    
    pipeline = create_pipeline(
        adapter=adapter,
        network=network,
        settings_loader=settings_loader,
    )
    
    result = pipeline.run(
        start_timestamp_ms=start_timestamp_ms,
        end_timestamp_ms=end_timestamp_ms,
        window_days=window_days,
        processing_date=processing_date,
        run_features=run_features,
        run_patterns=run_patterns,
    )
    
    logger.info(
        f"Pipeline complete: "
        f"{result['features_count']} features, "
        f"{result['patterns_count']} patterns, "
        f"{result['duration_seconds']:.2f}s"
    )


def _create_parquet_adapter(args):
    if not args.input:
        raise ValueError("--input is required for Parquet mode")
    if not args.input.exists():
        raise ValueError(f"Input path does not exist: {args.input}")
    
    output_path = args.output if args.output else construct_output_path(args.input)
    
    logger.info(f"Using Parquet adapter: input={args.input}, output={output_path}")
    
    return ParquetAdapter(
        input_path=args.input,
        output_path=output_path
    )


def _create_clickhouse_adapter(args, network: str):
    from chainswarm_core.db import ClientFactory, get_connection_params
    from chainswarm_analyzers_baseline.adapters.clickhouse import ClickHouseAdapter
    
    connection_params = get_connection_params(
        network=network,
        database_prefix=args.database_prefix
    )
    
    logger.info(
        f"Using ClickHouse adapter via chainswarm-core: "
        f"host={connection_params.get('host')}, "
        f"database={connection_params.get('database')}"
    )
    
    factory = ClientFactory(connection_params)
    client = factory.create_client()
    
    return ClickHouseAdapter(
        client=client,
        network=network
    )


if __name__ == "__main__":
    main()
