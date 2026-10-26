# skills
MAX_SKILLS_CHARS = 5000

# tool
MAX_CALL_TOOLS_TIMES = 3

# composer
MAX_COMPOSE_TOKENS = 1000
COMPOSE_TEMPERATURE = 0.1

# trace
MAX_TOOL_TRACES = 200

# agent
AGENT_MODEL = {
  "GENERAL": "",
  "TECHNICAL": "",
  "ESCALATION": ""
}

# memory
MAX_WORKING_MEMORY = 20
COMPRESS_THS = 15
RETAIN_NUM = 5
HISTORY_TOP_K = 5
MAX_SUMMARY_CHARS = 800
PORTRAIT_PREFIX = "user_profile"

# redis
REDIS_URL = "redis://redis:6379/0"
REDIS_EXPIRE = 24 * 60 * 60

# chromadb
CHROMA_HOST = "chromadb"
CHROMA_PORT = 8000
CHROMA_PERSIST_DIRECTORY = "/app/data/chroma"

# monitor
PROMETHEUS_PORT = 0
MONITOR_INTERVAL = 10
ALERT_WEBHOOK_URL = None

# evaluator
EVALUATE_PASS_THRESHOLD = 0.75

# run
ASSISTANT_HOST = "0.0.0.0"
ASSISTANT_PORT = 8000
ASSISTANT_ENV = "development"


import os as _os
from json import load as _load

__base_dir__ = _os.path.dirname(_os.path.dirname(__file__))
with open(_os.path.join(__base_dir__, "config/config.json"), 'r', encoding='utf-8') as _f:
    _config = _load(_f)

for _var, _value in _config.items():
    if _var not in globals(): continue # 防止名称污染
    globals()[_var] = _value


from sys import path as _sys_path
from logging import DEBUG as _DEBUG


BASE_DIR = __base_dir__
_sys_path.append(BASE_DIR)


LOG_DIR = _os.path.join(BASE_DIR, 'logs')
SKILLS_DIR = _os.path.join(BASE_DIR, 'skills')

ERROR_CODE_PATH = _os.path.join(BASE_DIR, 'config/error_code.json')
PROMPTS_PATH = _os.path.join(BASE_DIR, "config/prompts.json")
INTENT_PATH = _os.path.join(BASE_DIR, "config/intent.json")
AGENT_TYPE_INFO_PATH = _os.path.join(BASE_DIR, "config/agent_type_info.json")
ALERT_PATH = _os.path.join(BASE_DIR, "config/alert.json")


# 文件日志和控制台日志输出级别
FILE_LOG_LEVEL = _DEBUG
STREAM_LOG_LEVEL = _DEBUG


# 类型映射
TYPES = {
    "string": (str,),
    "number": (int, float),
    "integer": (int,),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,)
}
for _k, _v in (_config.get("TYPES") or {}).items():
    if not isinstance(_v, str): continue
    try:
        _types = tuple(eval(_v))
        for _t in _types:
            if not isinstance(_t, type):
                raise TypeError(f"unexpected type ({_t}) from TYPES[{_k}]")
        TYPES[_k] = _types
    except Exception as _e:
        print(_e)


print("********* config *********")
for _var, _value in globals().items():
    if _var.startswith("_"): continue
    print(f"{_var}: {_value}")
print("********* config *********\n")



if __name__ == "__main__":
    # print(MAX_WORKING_MEMORY)

    ...
