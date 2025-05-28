from fastmcp import FastMCP, Context
from models.market_data import (
    CandlesInput,
    CandlesOutput,
    PriceInput,
    PriceOutput,
    TickerInput,
    TickerOutput,
    OrderBookInput,
    OrderBookOutput,
    TradesInput,
    TradesOutput,
    OHLCVCandle,
    OrderBookLevel,
    TradeData,
)
from services.exchange_service import (
    fetch_ohlcv_data,
    fetch_ticker_data,
    fetch_order_book,
    fetch_trades,
)
import ccxt

mcp = FastMCP()


@mcp.tool()
async def get_candles(ctx: Context, inputs: CandlesInput) -> CandlesOutput:
    """
    Fetches OHLCV candle data for a given trading pair with specified timeframe and limit.
    """
    await ctx.info(
        f"Fetching {inputs.limit} candles for {inputs.symbol} ({inputs.timeframe.value})"
    )

    try:
        # Fetch OHLCV data from exchange
        ohlcv_data = await fetch_ohlcv_data(
            ctx=ctx,
            symbol=inputs.symbol,
            timeframe=inputs.timeframe.value,
            limit=inputs.limit,
            since=inputs.since,
        )

        if ohlcv_data:
            candles = [OHLCVCandle.from_list(candle) for candle in ohlcv_data]
            await ctx.info(
                f"Successfully fetched {len(candles)} candles for {inputs.symbol}"
            )

            return CandlesOutput(
                symbol=inputs.symbol,
                timeframe=inputs.timeframe.value,
                candles=candles,
                count=len(candles),
            )
        else:
            return CandlesOutput(
                symbol=inputs.symbol,
                timeframe=inputs.timeframe.value,
                error=f"No candle data available for {inputs.symbol}",
            )

    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        await ctx.error(f"CCXT Error fetching candles for {inputs.symbol}: {e}")
        return CandlesOutput(
            symbol=inputs.symbol,
            timeframe=inputs.timeframe.value,
            error=f"Exchange/Network Error: {str(e)}",
        )
    except Exception as e:
        await ctx.error(
            f"Unexpected error fetching candles for {inputs.symbol}: {e}", exc_info=True
        )
        return CandlesOutput(
            symbol=inputs.symbol,
            timeframe=inputs.timeframe.value,
            error="An unexpected server error occurred.",
        )


@mcp.tool()
async def get_current_price(ctx: Context, inputs: PriceInput) -> PriceOutput:
    """
    Fetches the current market price for a given trading pair symbol.
    """
    await ctx.info(f"Fetching current price for {inputs.symbol}")

    try:
        ticker = await fetch_ticker_data(ctx, inputs.symbol)
        if ticker and "last" in ticker and ticker["last"] is not None:
            await ctx.info(f"Current price for {inputs.symbol}: {ticker['last']}")
            return PriceOutput(
                symbol=inputs.symbol,
                price=ticker["last"],
                timestamp=ticker.get("timestamp"),
            )
        else:
            return PriceOutput(symbol=inputs.symbol, error="Price data not available")

    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        await ctx.error(f"CCXT Error fetching price for {inputs.symbol}: {e}")
        return PriceOutput(
            symbol=inputs.symbol, error=f"Exchange/Network Error: {str(e)}"
        )
    except Exception as e:
        await ctx.error(
            f"Unexpected error fetching price for {inputs.symbol}: {e}", exc_info=True
        )
        return PriceOutput(
            symbol=inputs.symbol, error="An unexpected server error occurred."
        )


@mcp.tool()
async def get_ticker(ctx: Context, inputs: TickerInput) -> TickerOutput:
    """
    Fetches comprehensive ticker data for a given trading pair.
    """
    await ctx.info(f"Fetching ticker data for {inputs.symbol}")

    try:
        ticker = await fetch_ticker_data(ctx, inputs.symbol)
        if ticker:
            await ctx.info(f"Successfully fetched ticker for {inputs.symbol}")
            return TickerOutput(
                symbol=inputs.symbol,
                bid=ticker.get("bid"),
                ask=ticker.get("ask"),
                last=ticker.get("last"),
                open=ticker.get("open"),
                high=ticker.get("high"),
                low=ticker.get("low"),
                close=ticker.get("close"),
                volume=ticker.get("baseVolume"),
                percentage=ticker.get("percentage"),
                change=ticker.get("change"),
                timestamp=ticker.get("timestamp"),
            )
        else:
            return TickerOutput(symbol=inputs.symbol, error="Ticker data not available")

    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        await ctx.error(f"CCXT Error fetching ticker for {inputs.symbol}: {e}")
        return TickerOutput(
            symbol=inputs.symbol, error=f"Exchange/Network Error: {str(e)}"
        )
    except Exception as e:
        await ctx.error(
            f"Unexpected error fetching ticker for {inputs.symbol}: {e}", exc_info=True
        )
        return TickerOutput(
            symbol=inputs.symbol, error="An unexpected server error occurred."
        )


@mcp.tool()
async def get_order_book(ctx: Context, inputs: OrderBookInput) -> OrderBookOutput:
    """
    Fetches order book data for a given trading pair.
    """
    await ctx.info(f"Fetching order book for {inputs.symbol} (limit: {inputs.limit})")

    try:
        order_book = await fetch_order_book(ctx, inputs.symbol, inputs.limit)
        if order_book:
            bids = [
                OrderBookLevel(price=bid[0], amount=bid[1])
                for bid in order_book.get("bids", [])
            ]
            asks = [
                OrderBookLevel(price=ask[0], amount=ask[1])
                for ask in order_book.get("asks", [])
            ]

            await ctx.info(f"Successfully fetched order book for {inputs.symbol}")
            return OrderBookOutput(
                symbol=inputs.symbol,
                bids=bids,
                asks=asks,
                timestamp=order_book.get("timestamp"),
            )
        else:
            return OrderBookOutput(
                symbol=inputs.symbol, error="Order book data not available"
            )

    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        await ctx.error(f"CCXT Error fetching order book for {inputs.symbol}: {e}")
        return OrderBookOutput(
            symbol=inputs.symbol, error=f"Exchange/Network Error: {str(e)}"
        )
    except Exception as e:
        await ctx.error(
            f"Unexpected error fetching order book for {inputs.symbol}: {e}",
            exc_info=True,
        )
        return OrderBookOutput(
            symbol=inputs.symbol, error="An unexpected server error occurred."
        )


@mcp.tool()
async def get_recent_trades(ctx: Context, inputs: TradesInput) -> TradesOutput:
    """
    Fetches recent trades data for a given trading pair.
    """
    await ctx.info(f"Fetching {inputs.limit} recent trades for {inputs.symbol}")

    try:
        trades_data = await fetch_trades(ctx, inputs.symbol, inputs.limit)
        if trades_data:
            trades = [
                TradeData(
                    id=trade.get("id"),
                    timestamp=trade["timestamp"],
                    price=trade["price"],
                    amount=trade["amount"],
                    side=trade["side"],
                )
                for trade in trades_data
            ]

            await ctx.info(
                f"Successfully fetched {len(trades)} trades for {inputs.symbol}"
            )
            return TradesOutput(symbol=inputs.symbol, trades=trades, count=len(trades))
        else:
            return TradesOutput(
                symbol=inputs.symbol, error="Recent trades data not available"
            )

    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        await ctx.error(f"CCXT Error fetching trades for {inputs.symbol}: {e}")
        return TradesOutput(
            symbol=inputs.symbol, error=f"Exchange/Network Error: {str(e)}"
        )
    except Exception as e:
        await ctx.error(
            f"Unexpected error fetching trades for {inputs.symbol}: {e}", exc_info=True
        )
        return TradesOutput(
            symbol=inputs.symbol, error="An unexpected server error occurred."
        )
