import logging
import asyncio
import threading
import schedule
import time
from datetime import datetime

# Import the stock analysis function
from utils.recommend_stocks import run_analysis_and_save, SimpleStockDB

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# --- Stock Analysis Scheduler ---
class StockAnalysisScheduler:
    def __init__(self):
        self.db = SimpleStockDB()
        self.scheduler_thread = None
        self.running = False

    def start_scheduler(self):
        """启动定时任务调度器"""
        if self.running:
            logger.warning("调度器已经在运行中")
            return

        self.running = True

        # 设置定时任务 - 每天凌晨2点运行
        schedule.every().day.at("03:20").do(self._run_analysis_job)

        # 启动调度器线程
        self.scheduler_thread = threading.Thread(
            target=self._scheduler_loop, daemon=True
        )
        self.scheduler_thread.start()

        logger.info("股票分析定时任务已启动 - 每天02:00执行")

    def stop_scheduler(self):
        """停止定时任务调度器"""
        self.running = False
        schedule.clear()
        logger.info("股票分析定时任务已停止")

    def _scheduler_loop(self):
        """调度器主循环"""
        while self.running:
            try:
                schedule.run_pending()
                time.sleep(60)  # 每分钟检查一次
            except Exception as e:
                logger.error(f"调度器运行出错: {e}")
                time.sleep(60)

    def _run_analysis_job(self):
        """运行股票分析任务"""
        logger.info("开始执行定时股票分析任务")
        start_time = datetime.now()

        try:
            # 在新的事件循环中运行异步任务
            def run_async():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    saved_count = loop.run_until_complete(
                        run_analysis_and_save(self.db, verbose=False)
                    )
                    end_time = datetime.now()
                    duration = (end_time - start_time).total_seconds()
                    logger.info(
                        f"定时股票分析完成 - 保存{saved_count}条记录，用时{duration:.1f}秒"
                    )
                except Exception as e:
                    logger.error(f"定时股票分析失败: {e}")
                finally:
                    loop.close()

            # 在线程中运行异步任务
            analysis_thread = threading.Thread(target=run_async)
            analysis_thread.start()
            analysis_thread.join(timeout=3600)  # 最多等待1小时

            if analysis_thread.is_alive():
                logger.warning("股票分析任务超时，可能仍在后台运行")

        except Exception as e:
            logger.error(f"股票分析任务执行出错: {e}")


# --- Global Scheduler Instance ---
stock_scheduler = StockAnalysisScheduler()
