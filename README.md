# MCP Crypto Server

A high-performance cryptocurrency market data and analysis platform built with FastMCP, CCXT, TimescaleDB, and RSS feed integration.

## Overview

MCP Crypto Server is a powerful platform that provides real-time cryptocurrency market data, advanced analytics, LLM-powered insights, and crypto news aggregation. Built on the Model Context Protocol (MCP), it enables seamless integration with AI applications while offering robust cryptocurrency exchange connectivity via CCXT, time-series data storage with TimescaleDB, and real-time news updates through RSS feeds.

## Key Features

- **Real-time Market Data**: Access live cryptocurrency market data from multiple exchanges
- **Historical Data Analysis**: Query and analyze historical price and volume data
- **LLM Integration**: Leverage AI for market trend analysis and insights
- **MCP Protocol Support**: Easily integrate with AI applications using the Model Context Protocol
- **Scalable Architecture**: Handle high request volumes with optimized performance
- **TimescaleDB Storage**: Efficient time-series data storage and querying
- **Multi-Exchange Support**: Connect to various cryptocurrency exchanges via CCXT
- **Crypto News Aggregation**: Collect and analyze news from multiple crypto RSS feeds
- **News Sentiment Analysis**: AI-powered sentiment analysis of crypto news
- **Custom RSS Feed Subscriptions**: Configure and manage news sources

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
│ News Feed   │  News Analysis      │ News Impact    │            │
└─────────────┴─────────────────────┴────────────────┴────────────┘
                    │                      │
       ┌────────────┼──────────────────────┘
       │            │                │
       ▼            ▼                ▼
┌─────────────┐ ┌────────────┐ ┌────────────────────┐
│  Exchange   │ │    RSS     │ │  Database Service  │
│  Service    │ │  Service   │ │                    │
│             │ │            │ │    TimescaleDB     │
│     CCXT    │ │ Feed Parser│ │    Time-series     │
└─────────────┘ └────────────┘ └────────────────────┘
```

## Technology Stack

- **FastMCP**: Server framework based on the Model Context Protocol
- **CCXT**: Library for cryptocurrency exchange connectivity
- **TimescaleDB**: Time-series database extension for PostgreSQL
- **Python**: Core programming language
- **uv**: High-performance Python package manager for dependency management
- **feedparser**: RSS feed parsing library
- **NLTK/spaCy**: Natural language processing for news analysis

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
│   └── news.py               # News feed resources and endpoints
├── tools/                    # MCP tools
│   ├── __init__.py
│   ├── market_analysis.py    # Market analysis tools
│   ├── price_data.py         # Price retrieval tools
│   ├── llm_analysis.py       # LLM integration for data analysis
│   └── news_analysis.py      # News analysis and sentiment tools
├── services/                 # Service integrations
│   ├── __init__.py
│   ├── exchange_service.py   # Exchange API integration
│   ├── database_service.py   # Database service for storing data
│   ├── llm_service.py        # LLM service integration
│   └── news_service.py       # RSS feed management and processing
├── models/                   # Data models
│   ├── __init__.py
│   ├── market_data.py        # Data models for market data
│   ├── analysis.py           # Models for analysis results
│   └── news.py               # Models for news data and analysis
├── utils/                    # Utility functions
│   ├── __init__.py
│   ├── formatters.py         # Data formatting utilities
│   ├── validators.py         # Input validation utilities
│   └── feed_utils.py         # RSS feed utility functions
├── prompts/                  # MCP prompts
│   ├── __init__.py
│   ├── analysis_prompts.py   # Prompt templates for analysis
│   └── news_prompts.py       # Prompt templates for news analysis
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
- RSS feed subscription settings
- News source configurations
- Logging levels and outputs

### Configuring RSS Feeds

The system comes with predefined crypto news sources, but you can customize them:

```python
# In config/settings.py

RSS_FEEDS = [
    {
        "name": "CoinDesk",
        "url": "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "update_interval": 900,  # 15 minutes
        "categories": ["news", "markets", "business"]
    },
    {
        "name": "Cointelegraph",
        "url": "https://cointelegraph.com/rss",
        "update_interval": 600,  # 10 minutes
        "categories": ["news", "analysis"]
    },
    # Add your custom RSS feeds here
]
```

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
- `news_analysis` - Analyze news sentiment and impact on specific coins
- `news_summary` - Generate summaries of recent crypto news

#### Resources

- `symbols://list` - Get all available cryptocurrency symbols
- `market_data://{symbol}` - Get real-time market data for a specific symbol
- `historical_data://{symbol}?timeframe={timeframe}&start={start}&end={end}` - Get historical data with parameters
- `news://recent` - Get most recent news across all sources
- `news://{source}` - Get news from a specific source
- `news://by-coin/{symbol}` - Get news related to a specific cryptocurrency

#### Prompts

- `analyze_market_trend` - Pre-defined prompt for analyzing market trends
- `generate_trading_recommendation` - Generate trading recommendations based on current market data
- `analyze_news_impact` - Analyze how recent news might impact cryptocurrency prices

### Using News Features with MCP Clients

```python
from mcp import Client

# Get recent news for a specific cryptocurrency
client = Client("http://localhost:8000")
news_response = await client.query(
    inputs=[{"symbol": "BTC/USD"}],
    resources=["news://by-coin/BTC"]
)
print(news_response.outputs)

# Analyze news sentiment and potential market impact
analysis_response = await client.query(
    inputs=[{"symbol": "BTC/USD", "timeframe": "24h"}],
    tools=["news_analysis"]
)
print(analysis_response.outputs)
```

## Development

### Adding a New Exchange

1. Update the `services/exchange_service.py` file to add the new exchange configuration
2. Implement any exchange-specific handling required
3. Update the available symbols in `resources/symbols.py`

### Adding a New RSS Feed Source

1. Update the `config/settings.py` file to add the new feed configuration
2. If the feed requires special parsing, extend the `services/news_service.py` with custom handlers
3. (Optional) Add source-specific endpoints in `resources/news.py`

```python
# Example of adding a custom parser in services/news_service.py

def parse_custom_feed(feed_data, source_config):
    """Custom parser for a specific news source"""
    articles = []
    for entry in feed_data.entries:
        # Custom parsing logic
        articles.append({
            "title": entry.title,
            "url": entry.link,
            "published": parse_date(entry.published),
            "summary": entry.summary,
            "source": source_config["name"]
        })
    return articles
```

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
- For RSS feeds, adjust polling intervals based on source update frequency to minimize unnecessary requests
- Consider implementing a background task scheduler for RSS feed updates to avoid blocking main application threads
- Scale horizontally by deploying multiple instances behind a load balancer

## License

[MIT License](LICENSE)

## Acknowledgements

- [FastMCP](https://github.com/jlowin/fastmcp) - Model Context Protocol server implementation
- [Model Context Protocol](https://modelcontextprotocol.io/) - Efficient protocol for AI applications
- [CCXT](https://github.com/ccxt/ccxt) - Cryptocurrency exchange trading library
- [TimescaleDB](https://www.timescale.com/) - Time-series database for PostgreSQL
- [uv](https://github.com/astral-sh/uv) - High-performance Python package manager
- [feedparser](https://feedparser.readthedocs.io/) - Python RSS feed parsing library
