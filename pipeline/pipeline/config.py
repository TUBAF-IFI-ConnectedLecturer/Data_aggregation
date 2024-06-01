# parse yaml file with custom tag implementing path variables and 
# merge operations

import yaml

def join(loader, node):
    seq = loader.construct_sequence(node)
    return ''.join([str(i) for i in seq])

if __name__ == '__main__':
    yaml_content = """
general:
  - data_root_folder: &BASE ./data
  - raw_data_folder: &RAW !join [*BASE, /raw]
  - processed_data_folder: &PREPROCESSED !join [*BASE, /processed]
    """
    
    yaml.add_constructor('!join', join)
    data = yaml.load(yaml_content, Loader=yaml.FullLoader)
    print(data)