import os, sys, logging
from typing import List, Dict

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(BASE_DIR)


# 文件日志和控制台日志输出级别
FILE_LOG_LEVEL = logging.DEBUG
STREAM_LOG_LEVEL = logging.DEBUG


MAX_CALL_TOOLS_TIMES = 3


MAX_COMPOSE_TOKENS = 1000
COMPOSE_TEMPERATURE = 0.1


TYPES = {
    "string": (str,),
    "number": (int, float),
    "integer": (int,),
    "boolean": (bool,),
    "array": (List,),
    "object": (Dict,)
}


LOG_DIR = os.path.join(BASE_DIR, 'logs')
PROMPTS_PATH = os.path.join(BASE_DIR, "prompts.json")
INTENT_PATH = os.path.join(BASE_DIR, "intent.json")
