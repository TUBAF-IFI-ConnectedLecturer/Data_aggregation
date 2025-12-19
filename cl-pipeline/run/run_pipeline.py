import yaml
from pathlib import Path
import logging
import argparse
import os

from pipeline.logger import setup_logger
from pipeline.pipeline import Pipeline
from pipeline.config import join
from pipeline.taskfactory import import_stages

from dotenv import load_dotenv

def _get_log_file_path(config_file_path: Path) -> str:
    """
    Determine log file path based on config file location.

    If config is in pipelines/*/config/, logs go to pipelines/*/logs/
    Otherwise, use default app.log

    Args:
        config_file_path: Path to config file

    Returns:
        Path to log file
    """
    config_path = Path(config_file_path)

    # Check if config is in pipelines structure
    if 'pipelines' in config_path.parts:
        parts = list(config_path.parts)
        try:
            pipelines_idx = parts.index('pipelines')
            if pipelines_idx + 1 < len(parts):
                pipeline_name = parts[pipelines_idx + 1]
                config_name = config_path.stem  # e.g., "test" or "full"

                # Construct log path: pipelines/<pipeline_name>/logs/<config_name>.log
                log_dir = Path('pipelines') / pipeline_name / 'logs'
                log_file = log_dir / f"{config_name}.log"
                return str(log_file)
        except (ValueError, IndexError):
            pass

    # Fallback to default
    return "app.log"

def run_pipeline(config_file_path: Path):
    # Determine log file path based on config file location
    log_file_path = _get_log_file_path(config_file_path)

    setup_logger(log_file_path)
    logging.info("Starting pipeline")
    logging.info(f"Config file: {config_file_path}")

    yaml.add_constructor('!join', join)
    with open(config_file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    import_stages(config["stages_module_path"])
    myPipeline = Pipeline(config["stages"], config["folder_structure"])
    myPipeline.run_pipline()

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description="Run the pipeline.")
    parser.add_argument("-c", "--config_path", required=False, 
                        default="identification_opal.yaml",
                        help="path to the config file")
    args=parser.parse_args()

    load_dotenv(Path(__file__).parent / '.env')
    print(Path(__file__).parent )
    # Only print GITHUB_API_KEY if it exists (not needed for local PDF pipeline)
    if "GITHUB_API_KEY" in os.environ:
        print(f"GITHUB_API_KEY: {os.environ['GITHUB_API_KEY']}")
    print("----------------------------------------------------")
    run_pipeline(Path(args.config_path))
