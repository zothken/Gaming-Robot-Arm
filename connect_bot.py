import sys

sys.path.append('uArm-Python-SDK-2.0')

from uarm.wrapper import SwiftAPI
from uarm.utils.log import logger
logger.setLevel(logger.VERBOSE)




swift = SwiftAPI(port="COM5", callback_thread_pool_size=1)

#swift.reset()

#swift.set_position(x=346, y=0, z=10)

#swift.disconnect()