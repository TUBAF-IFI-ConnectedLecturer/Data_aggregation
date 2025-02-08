from abc import ABC, abstractmethod
from pathlib import Path
import sys
import logging
import os
import subprocess
from inspect import isclass
from pkgutil import iter_modules
from pathlib import Path
from importlib import import_module
from typing import Union
import importlib.util
import importlib.machinery

from pipeline.logger import setup_logger

# Decorator to log the execution of a task
def loggedExecution(func):
    def wrapper(self, *args):
        # Check if the task has a name attribute
        if not hasattr(self, "name"):
            name = "?"
        else:
            name = self.name
        # Actual logging "enveloping" the execution of the task
        logging.info("Stage {} ({}) started", name, self.__class__.__name__) 
        res = func(self, *args)
        logging.info("Stage {} ({}) finished", name, self.__class__.__name__) 
        return res
    return wrapper

class Task(ABC):

    def __init__(self, config_stage, config_global):
        if "name" in config_stage:
            self.name = config_stage["name"]
        else:
            logging.error(f"No stage name in config.")
        self.set_parameters(config_stage)
        self.set_parameters(config_global)

    def set_parameters(self, config):
        for k, v in config.items():
            if "file_name" in k:
                v = Path(v)
            setattr(self, k, v)
    
    def run(self):
        self.execute_task()

    def execute_task(self):
        pass

class TaskWithInputFileMonitor(Task):

    def __init__(self, config_stage, config_global):
        super().__init__(config_stage, config_global)
        if "force_run" not in self.parameters:
            self.parameters['force_run'] = False

    # Check if all input files exist
    def all_input_files_exist(self):
        for attribute in vars(self):
            if "_input" in attribute:
                path = getattr(self, attribute)
                
                if path.exists() == False:
                    print (path)
                    return False
                logging.debug(f"Input file at {path} exists.")
        return True

    def changes_in_input_files(self):
        for attribute in vars(self):
            if "_input" in attribute:
                path = getattr(self, attribute)
                
                md5sum_result = subprocess.run(['md5sum', path], capture_output=True, text=True)

                if md5sum_result.returncode != 0:
                    logging.error(f"Error computing MD5 checksum: {md5sum_result.stderr}")
                    return False
                
                current_file_hash = md5sum_result.stdout.split(" ")[0]
                path_stored_date = path.parent / f".{path.name}.hash"

                if not path_stored_date.exists():
                    with open(path_stored_date, 'w') as file:
                        file.write(current_file_hash)
                    return True
                else:
                    with open(path_stored_date, "r") as file:
                        old_file_hash = file.read()
                    if old_file_hash != current_file_hash:
                        with open(path_stored_date, 'w') as file:
                            file.write(current_file_hash)
                        return True
        return False

    @loggedExecution
    def run(self):
        if self.all_input_files_exist():
            if self.changes_in_input_files() or self.parameters['force_run'] == True:
                if self.parameters['force_run'] == True:
                    logging.info(f"Forcing task to run.")
                logging.info(f"Input files for task have changed. Running task.")
                self.execute_task()
            else:
                logging.info(f"Input files are present but have not changed.")
        else:
            print(self.parameters)
            logging.error(f"Input files are missing!")

# Converting a string to a class name
def str_to_class(class_name, module_name=None):
    try:
        return getattr(sys.modules[__name__], class_name)
    except AttributeError:
        return None

def find_modules(folders):
    python_files_dict = {}
    for folder_path in folders:
        path = Path(folder_path)
        python_files = list(path.rglob('*.py'))
        # Add entries to the dictionary with file names as keys and file paths as values
        for file_path in python_files:
            python_files_dict[file_path.name] = str(file_path)
    return python_files_dict

def import_module_from_path(module_name, file_path):
    # Create a module spec
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    # Create a new module based on the spec
    module = importlib.util.module_from_spec(spec)
    # Add the module to sys.modules
    sys.modules[module_name] = module
    # Execute the module in its own namespace
    spec.loader.exec_module(module)
    return module

def import_stages(package_paths):
    if package_paths is None:
        logging.debug("No package paths defined in config.")
        return
    modules = find_modules(package_paths)
    for module_name, file_path in modules.items():
        module_name = module_name.replace(".py", "")
        module = import_module_from_path(module_name, file_path)

        for attribute_name in dir(module):
            attribute = getattr(module, attribute_name)
            if isclass(attribute):            
                # Add the class to this package's variables
                if globals().get(attribute_name) is None:
                    logging.debug(f"Import {attribute_name} to globals()")
                    globals()[attribute_name] = attribute

# Factory method to create a task
def task_builder(config_stage, config_global) -> Union[Task, None]:
    if "class" not in config_stage:
        logging.error(f"No task class defined for state {config_stage['name']}")
        return None
    class_ = str_to_class(config_stage["class"])
    if class_ is None: 
        logging.error(f"Class {config_stage['class']} does not exist")
        return None
    return class_(config_stage, config_global)