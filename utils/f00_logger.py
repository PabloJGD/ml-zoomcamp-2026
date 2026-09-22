import os
import sys
import logging as lg
from datetime import datetime
from contextlib import contextmanager

os.makedirs("logs", exist_ok=True)

_GREEN = '\033[32m'
_YELLOW = '\033[33m'
_RED = '\033[31m'
_RESET = '\033[0m'

_LEVEL_COLOR = {
    lg.INFO: _GREEN,
    lg.WARNING: _YELLOW,
    lg.ERROR: _RED,
    lg.CRITICAL: _RED,
}


class _ColorFormatter(lg.Formatter):
    def format(self, record):
        color = _LEVEL_COLOR.get(record.levelno, '')
        if not color:
            return super().format(record)
        asctime = self.formatTime(record, self.datefmt)
        prefix  = f"{color}{asctime} - {record.levelname}{_RESET}"
        result = f"{prefix} - {record.getMessage()}"
        if record.exc_info:
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            result += "\n" + record.exc_text
        if record.stack_info:
            result += "\n" + self.formatStack(record.stack_info)
        return result

_plain_fmt = '%(asctime)s - %(levelname)s - %(message)s'
_date_fmt  = '%Y-%m-%d %H:%M:%S'

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

_file_handler = lg.FileHandler("logs/execution_logs.log", mode="a", encoding="utf-8")
_file_handler.setFormatter(lg.Formatter(_plain_fmt, datefmt=_date_fmt))

_console_handler = lg.StreamHandler()
_console_handler.setFormatter(_ColorFormatter(_plain_fmt, _date_fmt))

_root = lg.getLogger()
_root.setLevel(lg.INFO)
_root.handlers.clear()
_root.addHandler(_file_handler)
_root.addHandler(_console_handler)

logger = _root
