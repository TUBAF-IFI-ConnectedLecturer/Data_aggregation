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

def run_pipeline(config_file_path: Path):
    setup_logger()
    logging.info("Starting pipeline")
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

    load_dotenv()

    run_pipeline(Path(args.config_path))
