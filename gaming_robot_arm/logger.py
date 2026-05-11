# Zentrales Logging-Setup fuer alle gaming-robot-arm Module.

import logging

from gaming_robot_arm.config import LOG_LEVEL

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("gaming-robot-arm")
