# tools/price_data.py
from fastmcp import FastMCP, Context
from models.market_data import PriceInput, PriceOutput  # Uses corrected Pydantic models
from services.exchange_service import exchange_service
import ccxt

mcp = FastMCP()


@mcp.tool()
async def get_current_price(
    ctx: Context, inputs: PriceInput
) -> PriceOutput:  # Correctly uses PriceInput
    """
    Fetches the current market price for a given trading pair symbol.
    Uses the last traded price from the exchange ticker.
    """
    await ctx.info(f"Executing get_current_price for symbol: {inputs.symbol}")
    try:
        ticker = await exchange_service.get_ticker(inputs.symbol)
        if ticker and "last" in ticker and ticker["last"] is not None:
            await ctx.info(
                f"Successfully fetched price for {inputs.symbol}: {ticker['last']}"
            )
            return PriceOutput(symbol=inputs.symbol, price=ticker["last"])
        elif ticker:
            await ctx.warning(
                f"'last' price not found or is null in ticker for {inputs.symbol}. Ticker: {ticker}"
            )
            return PriceOutput(
                symbol=inputs.symbol,
                price=None,
                error="Last price not available or null in ticker data.",
            )
        else:
            return PriceOutput(
                symbol=inputs.symbol,
                price=None,
                error=f"Could not fetch ticker data for {inputs.symbol} from {exchange_service.exchange_id}.",
            )

    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        await ctx.error(f"CCXT Error fetching price for {inputs.symbol}: {e}")
        # Provide a more specific error message if possible
        error_msg = (
            f"Exchange/Network Error ({type(e).__name__}): Check symbol and connection."
        )
        return PriceOutput(symbol=inputs.symbol, price=None, error=error_msg)
    except Exception as e:
        await ctx.error(
            f"Unexpected error in get_current_price for {inputs.symbol}: {e}",
            exc_info=True,
        )
        return PriceOutput(
            symbol=inputs.symbol,
            price=None,
            error="An unexpected server error occurred.",
        )
