from pipeline.taskfactory import Task, loggedExecution
from pathlib import Path

class ProvideDataFolders(Task):
    def __init__(self, config_stage, config_global):
        super().__init__(config_stage, config_global)

    @loggedExecution
    def execute_task(self):
        for key in self.__dict__:
            if "folder" in key:
                path = Path( getattr(self, key))
                path.mkdir(parents=True, exist_ok=True)