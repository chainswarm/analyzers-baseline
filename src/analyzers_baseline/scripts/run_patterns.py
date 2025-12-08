import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to python path and change working directory
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)
os.chdir(project_root)

from dotenv import load_dotenv
from loguru import logger
from chainswarm_core.observability import setup_logger

from analyzers_baseline.adapters.parquet import ParquetAdapter
from analyzers_baseline.pipeline import create_pipeline
from analyzers_baseline.scripts.run_pipeline import parse_path_metadata, construct_output_path


def main():
    load_dotenv()
    
    parser = argparse.ArgumentParser(
        description="Run pattern detection on blockchain transaction data"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
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
    
    args = parser.parse_args()
    
    network_hint = args.network or "baseline"
    setup_logger(f"analyzers-baseline-{network_hint}-patterns")
    
    logger.info("Starting pattern detection")
    
    if not args.input.exists():
        raise ValueError(f"Input path does not exist: {args.input}")
    
    path_metadata = parse_path_metadata(args.input)
    network = args.network or path_metadata["network"]
    processing_date = args.processing_date or path_metadata["processing_date"]
    window_days = args.window_days if args.window_days is not None else path_metadata["window_days"]
    
    logger.info(f"Extracted from path: network={network}, date={processing_date}, window={window_days}")
    
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
    
    output_path = args.output if args.output else construct_output_path(args.input)
    
    adapter = ParquetAdapter(
        input_path=args.input,
        output_path=output_path
    )
    
    pipeline = create_pipeline(adapter=adapter)
    
    result = pipeline.run(
        start_timestamp_ms=start_timestamp_ms,
        end_timestamp_ms=end_timestamp_ms,
        window_days=window_days,
        processing_date=processing_date,
        network=network,
        run_features=False,
        run_patterns=True,
    )
    
    logger.info(f"Pattern detection complete: {result['patterns_count']} patterns detected")


if __name__ == "__main__":
    main()