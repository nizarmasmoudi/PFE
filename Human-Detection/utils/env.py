import configparser

config = configparser.ConfigParser()
config.read('../../env.cfg')

DATASET_PATH = config['DATASET']['MAIN_PATH']
TRAIN_PATH = config['DATASET']['TRAIN_PATH']
VALID_PATH = config['DATASET']['VALID_PATH']
TEST_PATH = config['DATASET']['TEST_PATH']