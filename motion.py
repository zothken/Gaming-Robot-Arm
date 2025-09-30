import sys

sys.path.append(r'C:\Users\nando\OneDrive\Documents\Uni\PP BA\gaming-robot-arm\uArm-Python-SDK-2.0')

from uarm.wrapper import SwiftAPI
from uarm.utils.log import logger
logger.setLevel(logger.VERBOSE)

swift = SwiftAPI(port="COM5", callback_thread_pool_size=1)

#swift.reset()

#swift.set_position(x=100, y=100, z=100)

swift.disconnect()