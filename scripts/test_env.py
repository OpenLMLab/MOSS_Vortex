import os
import time
import json
import logging
from typing import Any, List
import pathlib
import tempfile
import threading
import queue
import json
import time
try:
    import thread
except ImportError:
    import _thread as thread

from prometheus_client import (  # type: ignore
    CollectorRegistry,
    Counter,
    multiprocess,
    start_http_server,
)

import torch  # type: ignore
import onnxruntime as ort
from transformers import (  # type: ignore
    AutoTokenizer, 
    AutoModelForCausalLM, 
    CodeGenForCausalLM, 
    StoppingCriteriaList,StoppingCriteria
)
import numpy as np

from mosec import Server, Worker
import websocket
from websocket import create_connection

from mosec import Server, Worker
from mosec.errors import EncodingError, DecodingError, ValidationError
from onnx_infer import Inferer
from onnx_batch_infer import Inferer as Batch_Inferer