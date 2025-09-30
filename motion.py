import sys
#the uArm application is inside of the parent directory, this assumes that you are runing this notebook in the `notebooks` directory
sys.path.append('..')

from uarm.wrapper import SwiftAPI
from uarm.utils.log import logger
logger.setLevel(logger.VERBOSE)