# 临时脚本：跑 SR-DKT 30 个 epoch
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from train import *

# 将 epoch 数改为 30
CONFIG["epochs"] = 30

if __name__ == "__main__":
    main()