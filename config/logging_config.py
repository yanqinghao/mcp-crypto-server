# config/logging_config.py

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(
    log_level="INFO",
    log_dir="logs",
    max_file_size=10 * 1024 * 1024,  # 10MB
    backup_count=5,
):
    """
    设置简单的日志系统，带有文件大小限制

    Args:
        log_level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: 日志文件目录
        max_file_size: 单个日志文件最大大小（字节）
        backup_count: 保留的备份日志文件数量
    """
    # 确保日志目录存在
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # 创建日志文件名
    log_file = log_path / "mcp_crypto.log"

    # 配置根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # 创建格式化器
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 创建文件处理器（带大小限制和自动轮转）
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_file_size,  # 文件大小限制
        backupCount=backup_count,  # 备份文件数量
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # 添加控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # 降低一些第三方库的日志级别，以减少噪音
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("ccxt").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    logger = logging.getLogger("config")
    logger.info(
        f"日志系统已初始化，级别：{log_level}，文件：{log_file}，大小限制：{max_file_size / 1024 / 1024:.1f}MB"
    )


class ServiceLogger:
    """简单的服务日志记录器"""

    def __init__(self, service_name):
        """初始化服务日志记录器"""
        self.logger = logging.getLogger(service_name)

    def debug(self, message, **kwargs):
        """记录调试信息"""
        self._log(logging.DEBUG, message, **kwargs)

    def info(self, message, **kwargs):
        """记录一般信息"""
        self._log(logging.INFO, message, **kwargs)

    def warning(self, message, **kwargs):
        """记录警告信息"""
        self._log(logging.WARNING, message, **kwargs)

    def error(self, message, **kwargs):
        """记录错误信息"""
        self._log(logging.ERROR, message, **kwargs)

    def critical(self, message, **kwargs):
        """记录严重错误"""
        self._log(logging.CRITICAL, message, **kwargs)

    def _log(self, level, message, **kwargs):
        """处理额外参数的内部日志方法"""
        if kwargs:
            # 将额外的键值对添加到消息中
            extra_info = ", ".join(f"{k}={v}" for k, v in kwargs.items())
            full_message = f"{message} [{extra_info}]"
        else:
            full_message = message

        self.logger.log(level, full_message)
