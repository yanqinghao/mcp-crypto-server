#!/usr/bin/env python
from fastmcp import FastMCP

# Create the MCP server
mcp = FastMCP(
    "Crypto Data Analysis",
    dependencies=["pandas", "numpy", "ccxt", "matplotlib", "influxdb-client", "langchain"],
)

# Import resources, tools, and prompts
from prompts.analysis_prompts import *
from resources.market_data import *
from resources.schemas import *
from resources.symbols import *
from tools.llm_analysis import *
from tools.market_analysis import *
from tools.price_data import *

# Run the server if executed directly
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:mcp", host="0.0.0.0", port=8000)
