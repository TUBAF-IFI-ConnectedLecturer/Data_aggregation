import unittest
from pipeline.config import join
import yaml

class TestConfig(unittest.TestCase):
    def test_join(self):
        yaml_content = """
        general:
          - data_root_folder: &BASE ./data
          - raw_data_folder: &RAW !join [*BASE, /raw]
          - processed_data_folder: &PREPROCESSED !join [*BASE, /processed]    
        """
        yaml.add_constructor('!join', join)
        data = yaml.load(yaml_content, Loader=yaml.FullLoader)
        self.assertEqual(data['general'][1]['raw_data_folder'], './data/raw')
        self.assertEqual(data['general'][2]['processed_data_folder'], './data/processed')

if __name__ == '__main__':
    unittest.main()
