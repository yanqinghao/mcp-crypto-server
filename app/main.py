import logging

from fastmcp import FastMCP  # Context is used in tools
from config import settings

# Import tools from their respective modules
from tools.price_data import mcp as price_tools
from tools.indicator_calculator import mcp as indicator_tools
from tools.report_generator import mcp as report_tools

from prompts.analysis_prompts import mcp as analysis_prompts

from resources.market_info_resources import mcp as market_info_resources


# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Main MCP Server Instance ---
mcp_server = FastMCP(
    name="Crypto Analysis MCP Server",
    instructions="Provides cryptocurrency price data and technical analysis indicators (SMA, RSI, MACD) using CCXT and TA-Lib.",
    host=settings.SERVER_HOST,
    port="8888",
    # port=settings.SERVER_PORT,
)

# --- Import Tools into the Main Server ---
mcp_server.mount("price", price_tools)

mcp_server.mount("indicator", indicator_tools)

mcp_server.mount("report", report_tools)

mcp_server.mount("analysis_prompts", analysis_prompts)

mcp_server.mount("market_info", market_info_resources)
# logger.info(f"Imported tools: {list(mcp_server.get_tools())}")


@mcp_server.tool()
def ping():
    return "Composite OK"


# --- Run the Server ---
def main():
    """Main function to run the MCP server."""
    logger.info("Attempting to start server...")
    mcp_server.run(transport="sse", uvicorn_config={"reload": True})  # [2]


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nServer interrupted by user. Shutting down...")
    except Exception:
        logger.exception("An error occurred during server execution:")
