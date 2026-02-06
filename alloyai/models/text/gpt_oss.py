import logging
import gc
from typing import Mapping, Union

import torch

from .. import Model

logging.getLogger(__name__).debug("torch version: %s", torch.__version__)
