import configparser
import os
config = configparser.ConfigParser()
if os.path.exists('../env.cfg'):
    config.read('../env.cfg')
else:
    raise FileNotFoundError('Configuration file not found!')

DATASET_PATH = config['DATASET']['MAIN_PATH']
TRAIN_PATH = config['DATASET']['TRAIN_PATH']
VALID_PATH = config['DATASET']['VALID_PATH']
TEST_PATH = config['DATASET']['TEST_PATH']