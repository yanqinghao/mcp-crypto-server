# MCP Crypto Server

A high-performance cryptocurrency market data and analysis platform built with FastMCP, CCXT, and TimescaleDB.

## Overview

MCP Crypto Server is a powerful platform that provides real-time cryptocurrency market data, advanced analytics, and LLM-powered insights. Built on the Model Context Protocol (MCP), it enables seamless integration with AI applications while offering robust cryptocurrency exchange connectivity via CCXT and time-series data storage with TimescaleDB.

## Key Features

- **Real-time Market Data**: Access live cryptocurrency market data from multiple exchanges
- **Historical Data Analysis**: Query and analyze historical price and volume data
- **LLM Integration**: Leverage AI for market trend analysis and insights
- **MCP Protocol Support**: Easily integrate with AI applications using the Model Context Protocol
- **Scalable Architecture**: Handle high request volumes with optimized performance
- **TimescaleDB Storage**: Efficient time-series data storage and querying
- **Multi-Exchange Support**: Connect to various cryptocurrency exchanges via CCXT

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      MCP Client Applications                     │
│   (Claude Desktop, Custom AI Applications, Analysis Tools)       │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                        FastMCP Server                            │
├─────────────┬─────────────────────┬────────────────┬────────────┤
│  Resources  │        Tools        │    Prompts     │  Schemas   │
│             │                     │                │            │
│ Market Data │  Market Analysis    │ Trading Rec.   │ Input/     │
│ Symbols     │  Price Data         │ Market Trend   │ Output     │
│ Historical  │  LLM Analysis       │ Analysis       │ Validation │
└─────────────┴─────────────────────┴────────────────┴────────────┘
                    │                      │
       ┌────────────┘                      └────────────┐
       ▼                                                ▼
┌─────────────────────┐                     ┌────────────────────┐
│  Exchange Service   │                     │  Database Service  │
│                     │                     │                    │
│  CCXT Integration   │◄───────────────────►│    TimescaleDB     │
│  Multiple Exchanges │                     │    Time-series     │
└─────────────────────┘                     └────────────────────┘
```

## Technology Stack

- **FastMCP**: Server framework based on the Model Context Protocol
- **CCXT**: Library for cryptocurrency exchange connectivity
- **TimescaleDB**: Time-series database extension for PostgreSQL
- **Python**: Core programming language
- **uv**: High-performance Python package manager for dependency management

## Project Structure

```
mcp-crypto-server/
├── main.py                   # Main entry point with FastMCP server definition
├── config/                   # Configuration directory
│   ├── __init__.py
│   ├── settings.py           # Application settings and configuration
│   └── logging_config.py     # Logging configuration
├── resources/                # MCP resources
│   ├── __init__.py
│   ├── market_data.py        # Market data resources
│   ├── symbols.py            # Cryptocurrency symbols resources
│   ├── schemas.py            # Schema resources for the client
│   └── news.py               #
├── tools/                    # MCP tools
│   ├── __init__.py
│   ├── market_analysis.py    # Market analysis tools
│   ├── price_data.py         # Price retrieval tools
│   ├── llm_analysis.py       # LLM integration for data analysis
│   └── news_analysis.py      #
├── services/                 # Service integrations
│   ├── __init__.py
│   ├── exchange_service.py   # Exchange API integration
│   ├── database_service.py   # Database service for storing data
│   ├── llm_service.py        # LLM service integration
│   └── news_service.py       #
├── models/                   # Data models
│   ├── __init__.py
│   ├── market_data.py        # Data models for market data
│   ├── analysis.py           # Models for analysis results
│   └── news.py               #
├── utils/                    # Utility functions
│   ├── __init__.py
│   ├── formatters.py         # Data formatting utilities
│   └── validators.py         # Input validation utilities
├── prompts/                  # MCP prompts
│   ├── __init__.py
│   └── analysis_prompts.py   # Prompt templates for analysis
├── pyproject.toml            # Project configuration and dependencies
└── uv.lock                   # Lock file for dependencies managed by uv
```

## Installation

### Prerequisites

- Python 3.9+
- Docker and Docker Compose
- uv package manager

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/mcp-crypto-server.git
   cd mcp-crypto-server
   ```

2. Install uv if you don't already have it:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. Create a virtual environment and install dependencies:
   ```bash
   uv venv
   uv pip sync
   ```

4. Set up TimescaleDB using Docker:
   ```bash
   docker-compose up -d timescaledb
   ```

5. Configure your environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration settings
   ```

## Configuration

Edit the `config/settings.py` file to customize your application:

- Database connection parameters
- Exchange API keys and settings
- MCP server configuration
- LLM service settings
- Logging levels and outputs

## Usage

### Starting the Server

```bash
uv run python main.py
```

The server will start by default and be available for connection via MCP clients.

### MCP Capabilities

The server exposes the following MCP capabilities:

#### Tools

- `market_analysis` - Analyze market trends and provide technical indicators
- `price_data` - Get real-time or historical price data for specific cryptocurrencies
- `llm_analysis` - Generate AI-driven market insights and predictions

#### Resources

- `symbols://list` - Get all available cryptocurrency symbols
- `market_data://{symbol}` - Get real-time market data for a specific symbol
- `historical_data://{symbol}?timeframe={timeframe}&start={start}&end={end}` - Get historical data with parameters

#### Prompts

- `analyze_market_trend` - Pre-defined prompt for analyzing market trends
- `generate_trading_recommendation` - Generate trading recommendations based on current market data

### Using with MCP Clients

The server supports the Model Context Protocol, allowing for efficient integration with AI applications. Example client usage:

```python
from mcp import Client

client = Client("http://localhost:8000")
response = await client.query(
    inputs=[{"symbol": "BTC/USD", "timeframe": "1h"}],
    tools=["market_analysis", "price_prediction"]
)
print(response.outputs)
```

## Development

### Adding a New Exchange

1. Update the `services/exchange_service.py` file to add the new exchange configuration
2. Implement any exchange-specific handling required
3. Update the available symbols in `resources/symbols.py`

### Creating Custom Tools

1. Add a new file in the `tools/` directory
2. Implement your tool logic using the FastMCP decorator:
   ```python
   @mcp.tool()
   def your_tool_name(param1: str, param2: int) -> dict:
       """Tool description"""
       # Implementation
       return result
   ```

### Adding New Resources

1. Create your resource function in the appropriate resources file:
   ```python
   @mcp.resource("resource://{parameter}")
   def get_resource(parameter: str) -> str:
       """Resource description"""
       # Implementation
       return data
   ```

### Managing Dependencies with uv

The project uses uv for fast and efficient dependency management:

1. Add a new dependency:
   ```bash
   uv add package_name
   ```

2. Update all dependencies:
   ```bash
   uv pip sync
   ```

3. Generate lock file:
   ```bash
   uv pip compile -o uv.lock
   ```

## Performance Considerations

- The application leverages TimescaleDB's time-series optimization for efficient historical data queries
- Consider using TimescaleDB hypertables with appropriate chunk intervals based on your data volume
- Implement appropriate caching strategies for frequently accessed data
- Scale horizontally by deploying multiple instances behind a load balancer

## License

[MIT License](LICENSE)

## Acknowledgements

- [FastMCP](https://github.com/jlowin/fastmcp) - Model Context Protocol server implementation
- [Model Context Protocol](https://modelcontextprotocol.io/) - Efficient protocol for AI applications
- [CCXT](https://github.com/ccxt/ccxt) - Cryptocurrency exchange trading library
- [TimescaleDB](https://www.timescale.com/) - Time-series database for PostgreSQL
- [uv](https://github.com/astral-sh/uv) - High-performance Python package manager
</artifact>

I've updated the README to include uv package management and added an architecture diagram. Here are the key additions:

1. **Architecture Diagram**: Added an ASCII diagram that visually shows the relationship between MCP clients, the FastMCP server, and the underlying services (exchanges and database).

2. **uv Integration**:
   - Added uv to the technology stack
   - Updated installation instructions to use uv instead of pip
   - Added a dedicated section for managing dependencies with uv
   - Replaced requirements.txt with pyproject.toml and uv.lock in the project structure

3. **Benefits of using uv**:
   uv offers significant speed improvements (10-100x faster than pip) for package installation and dependency resolution. This is particularly valuable when working with a complex project that has many dependencies.

4. **Installation Process**:
   - Added instructions for installing uv
   - Updated the virtual environment and dependency installation commands to use uv

5. **Running the Application**:
   - Changed the run command to use uv for consistency

These changes modernize your project by leveraging uv's advantages: faster package management, better dependency resolution, and simplified toolchain management in a single binary.
