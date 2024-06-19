import yaml
from pathlib import Path
import logging

from pipeline.logger import setup_logger
from pipeline.pipeline import Pipeline
from pipeline.config import join
from pipeline.taskfactory import import_stages

if __name__ == '__main__':
    setup_logger()
    logging.info("Starting pipeline")
    yaml.add_constructor('!join', join)
    with open('identification_opal.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    import_stages(config["stages_module_path"])
    myPipeline = Pipeline(config["stages"], config["folder_structure"])
    myPipeline.run_pipline()