import time
import os
import random
import functools
from collections import OrderedDict

import numpy as np
import torch

def get_current_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def output_to_file(buf, fout):
    print(buf)
    fout.write(buf + '\n')
    fout.flush()

