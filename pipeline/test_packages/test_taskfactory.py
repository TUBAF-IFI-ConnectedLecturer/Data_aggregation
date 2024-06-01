import unittest
from pipeline.taskfactory import Task, TaskWithInputFileMonitor, task_builder, loggedExecution, import_stages
from pipeline.logger import setup_logger
import logging
import yaml

class TestTask1(Task):

    def execute_task(self):
        logging.info(f"TestTask1 is executed as {self.name}")

class TestTask2(Task):
    def __init__(self, config_stage, config_global):
        self.df_file_name_input = ""
        super().__init__(config_stage, config_global)

    def execute_task(self):
        logging.info(f"TestTask2 is executed.")

class TestTask4(TaskWithInputFileMonitor):
    def __init__(self, config_stage, config_global):
        self.df_file_name_input = ""
        super().__init__(config_stage, config_global)

    def execute_task(self):
        logging.info(f"TestTask4 is executed as {self.name}")

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
        for stage in config["stages"]:
            task = task_builder(stage, config["folder_structure"])
            if task is not None:
                task.run()

if __name__ == '__main__':
    setup_logger()
    unittest.main()