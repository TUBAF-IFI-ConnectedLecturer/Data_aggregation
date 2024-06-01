import unittest
from pipeline.taskfactory import import_stages
from pipeline.pipeline import Pipeline
from pipeline.logger import setup_logger
import logging
import yaml

yaml_content = """
stages_module_path:
    - ./test_packages/

folder_structure:
  data_root_folder: &BASE ../data

stages:
  - name: Runner1
    class: TestTask1
  - class: TestTask2  # Missing state name
  - name: Runner3
    class: TestTask3  # Missing class
  - name: Runner4
    class: TestTask4
    df_file_name_input : ./test_packages/myTextFile.txt
  - name: Runner5
    class: TestTask4
    df_file_name_input : myTextFile_missing.txt
"""

class TestConfig(unittest.TestCase):
    def test_taskfactory(self):
        setup_logger()
        config = yaml.load(yaml_content, Loader=yaml.FullLoader)
        import_stages(config["stages_module_path"])

        pipeline = Pipeline(config["stages"], config["folder_structure"])
        pipeline.run_pipline()

if __name__ == '__main__':
    setup_logger()
    unittest.main()