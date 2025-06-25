import logging
import atexit
from fastmcp import FastMCP  # Context is used in tools
from config import settings

# Import tools from their respective modules
from tools.crypto_tools import mcp as crypto_tools
from tools.ak_stock_tools import mcp as ak_stock_tools
from tools.etf_tools import mcp as etf_tools
from tools.us_stock_tools import mcp as us_stock_tools
from prompts.analysis_prompts import mcp as analysis_prompts
from resources.market_info_resources import mcp as market_info_resources

from utils.stock_analysis_scheduler import stock_scheduler


# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# 注册清理函数
def cleanup_scheduler():
    """程序退出时清理调度器"""
    stock_scheduler.stop_scheduler()


atexit.register(cleanup_scheduler)


# --- Main MCP Server Instance ---
mcp_server = FastMCP(
    name="Crypto Analysis MCP Server",
    instructions="Provides cryptocurrency price data and technical analysis indicators (SMA, RSI, MACD) using CCXT and TA-Lib.",
    host=settings.SERVER_HOST,
    # port="8888",
    port=settings.SERVER_PORT,
)

# --- Import Tools into the Main Server ---
mcp_server.mount("crypto_tools", crypto_tools)
mcp_server.mount("a_hk_stock_tools", ak_stock_tools)
mcp_server.mount("etf_tools", etf_tools)
mcp_server.mount("us_stock_tools", us_stock_tools)
mcp_server.mount("analysis_prompts", analysis_prompts)
mcp_server.mount("market_info_resources", market_info_resources)
# logger.info(f"Imported tools: {list(mcp_server.get_tools())}")


@mcp_server.tool()
def ping():
    return "Composite OK"


# --- Run the Server ---
def main():
    """Main function to run the MCP server."""
    logger.info("Attempting to start server...")
    # 启动股票分析定时任务
    try:
        stock_scheduler.start_scheduler()
        logger.info("股票分析定时任务启动成功")
    except Exception as e:
        logger.error(f"启动股票分析定时任务失败: {e}")

    try:
        mcp_server.run(transport="sse", uvicorn_config={"reload": True})
    finally:
        # 确保在服务器停止时清理资源
        stock_scheduler.stop_scheduler()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nServer interrupted by user. Shutting down...")
    except Exception:
        logger.exception("An error occurred during server execution:")
