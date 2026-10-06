import json

from AssIstant.common.config import PROMPTS_PATH

with open(PROMPTS_PATH, 'r', encoding='utf-8') as f:
    PROMPTS = json.load(f)
