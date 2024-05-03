import math
from enum import Enum
import time
from queue import Queue
import numpy as np


# State Variables 
status_server = 0 # 0: idle, 1: busy
packet_in_queue = 0