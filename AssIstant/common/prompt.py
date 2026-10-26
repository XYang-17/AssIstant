import json
from typing import Dict

from AssIstant.common.config import PROMPTS_PATH

with open(PROMPTS_PATH, 'r', encoding='utf-8') as f:
    PROMPTS: Dict[str, str] = json.load(f)
