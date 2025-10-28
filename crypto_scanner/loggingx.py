# -*- coding: utf-8 -*-
import os
import sys
import logging
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime, timezone
from .config import DEBUG
from typing import Optional


# === 日志目录 ===
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# === 日志文件路径 ===
LOG_FILE = os.path.join(LOG_DIR, "detect.log")

# === 全局 logger ===
_logger = logging.getLogger("detector")
_logger.setLevel(logging.DEBUG if DEBUG else logging.INFO)

# === 格式器 ===
_fmt_console = logging.Formatter(
    "[%(asctime)s][%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"
)
_fmt_file = logging.Formatter(
    "%(asctime)s | %(levelname)-5s | %(filename)s:%(lineno)d | %(message)s",
    "%Y-%m-%d %H:%M:%S",
)

# === 控制台 Handler ===
_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setLevel(logging.DEBUG if DEBUG else logging.INFO)
_console_handler.setFormatter(_fmt_console)

# === 文件 Handler（每日轮转） ===
_file_handler = TimedRotatingFileHandler(
    LOG_FILE, when="midnight", interval=1, backupCount=10, encoding="utf-8"
)
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(_fmt_file)
_file_handler.suffix = "%Y%m%d"  # 文件名自动加日期后缀

# === 去重绑定 ===
if not _logger.handlers:
    _logger.addHandler(_console_handler)
    _logger.addHandler(_file_handler)


# === 时间工具 ===
def ts_now() -> str:
    """返回 UTC 时间字符串"""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


# === 简易日志函数（兼容旧版） ===
def dbg(msg: str):
    """兼容旧 dbg()，仅在 DEBUG=True 时打印"""
    if DEBUG:
        _logger.debug(msg)


# === 新推荐接口 ===
def info(msg: str):
    _logger.info(msg)


def warn(msg: str):
    _logger.warning(msg)


def error(msg: str):
    _logger.error(msg)


def exception(msg: str, exc: Optional[Exception] = None):
    """打印带异常堆栈"""
    if exc:
        _logger.exception(f"{msg}: {exc}")
    else:
        _logger.exception(msg)


# === 统一接口导出 ===
__all__ = ["dbg", "info", "warn", "error", "exception", "ts_now", "_logger"]
