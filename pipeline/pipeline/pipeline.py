from pipeline.taskfactory import task_builder,Task

class Pipeline:
    def __init__(self, config_stages, config_global):
        self.tasks = []
        for config_stage in config_stages:
            stage = task_builder(config_stage, config_global)
            if stage is not None:
                self.add_task(stage)

    def add_task(self, task: Task):
        self.tasks.append(task)

    def run_pipline(self):
        for task in self.tasks:
            task.run()