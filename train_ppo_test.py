import os
import subprocess
import sys
from enum import Enum, unique

from llamafactory import launcher
from llamafactory.api.app import run_api
from llamafactory.chat.chat_model import run_chat
from llamafactory.eval.evaluator import run_eval
from llamafactory.extras import logging
from llamafactory.extras.env import VERSION, print_env
from llamafactory.extras.misc import (
    find_available_port,
    get_device_count,
    is_env_enabled,
    use_ray,
)
from llamafactory.train.tuner import export_model, run_exp
from llamafactory.webui.interface import run_web_demo, run_web_ui

if __name__ == "__main__":
    run_exp()
