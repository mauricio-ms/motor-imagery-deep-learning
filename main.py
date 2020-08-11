import os
from logger.log_factory import get_logger

PROJECT_NAME = "motor-imagery-convolutional-recurrent-neural-network"
ROOT_DIR = os.getcwd()[:os.getcwd().index(PROJECT_NAME) + len(PROJECT_NAME)]

LOGGER = get_logger("logger")
