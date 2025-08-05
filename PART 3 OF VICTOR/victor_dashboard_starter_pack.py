# victor_dashboard_starter_pack.py
# Version: 1.0.0 — Initial Nodes for Victor Forensic Dashboard
# Drop into ComfyUI/custom_nodes/

import os
import re
import psutil
from datetime import datetime
from nodes import Node, NodeOutput, NodeCategory
from victor_logger import LOG_FILE, log_event

# ======================
# 📜 Real-Time Log Stream
# ======================
class VictorLogStreamNode(Node):
    def __init__(self):
        super().__init__()
        self.name = "Victor Log Stream"
        self.category = NodeCategory.CUSTOM
        self.description = "Returns the most recent N lines from victor_node.log"

        self.add_output("Latest Logs", str)

    def execute(self):
        if not os.path.exists(LOG_FILE):
            return ["[VictorLogStream] Log file not found."]
        try:
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                lines = f.readlines()[-100:]
            return ["".join(lines)]
        except Exception as e:
            return [f"[VictorLogStream][ERROR] {e}"]


# =============================
# ⚠️ Error Tracker / Extractor
# =============================
class VictorErrorTrackerNode(Node):
    def __init__(self):
        super().__init__()
        self.name = "Victor Error Tracker"
        self.category = NodeCategory.CUSTOM
        self.description = "Extracts all log lines containing '[ERROR]'"
        self.add_output("Error Lines", str)

    def execute(self):
        if not os.path.exists(LOG_FILE):
            return ["[VictorErrors] No log file found."]

        try:
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                errors = [line for line in f if "[ERROR]" in line or "[EXCEPTION]" in line]
            return ["".join(errors)]
        except Exception as e:
            return [f"[VictorErrors][ERROR] {e}"]


# ==================================
# 📊 Victor System Stats Snapshot
# ==================================
class VictorSystemStatusNode(Node):
    def __init__(self):
        super().__init__()
        self.name = "Victor System Status"
        self.category = NodeCategory.CUSTOM
        self.description = "Displays current CPU/RAM/log stats"
        self.add_output("System Status", str)

    def execute(se

# === AUTO-EXPAND HOOK ===
def expand():
    print(f'[AUTO_EXPAND] Module {__file__} has no custom logic. Placeholder activated.')
