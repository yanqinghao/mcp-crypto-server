from typing import List, Dict, Any
from config import settings

from models.analysis import (
    ComprehensiveAnalysisInput,
    ComprehensiveAnalysisOutput,
    SmaInput,
    RsiInput,
    MacdInput,
    BbandsInput,
    AtrInput,
    AdxInput,
    ObvInput,
)

# Import individual calculator tools
from .indicator_calculator import (
    calculate_sma,
    calculate_rsi,
    calculate_macd,
    calculate_bbands,
    calculate_atr,
    calculate_adx,
    calculate_obv,
)
from fastmcp import FastMCP, Context

mcp = FastMCP()


@mcp.tool()
async def generate_comprehensive_market_report(
    ctx: Context, inputs: ComprehensiveAnalysisInput
) -> ComprehensiveAnalysisOutput:
    """
    Generates a comprehensive market analysis report by combining multiple technical indicators.
    Returns historical data for each indicator based on history_len parameter.
    """
    await ctx.info(
        f"Generating comprehensive report for {inputs.symbol} ({inputs.timeframe}) with {inputs.history_len} data points."
    )
    output_base = {"symbol": inputs.symbol, "timeframe": inputs.timeframe}

    indicator_results_structured: Dict[str, Any] = {}
    report_sections: List[str] = []

    # Determine which indicators to run
    default_indicators = ["SMA", "RSI", "MACD", "BBANDS", "ATR", "ADX", "OBV"]
    indicators_to_run = (
        inputs.indicators_to_include
        if inputs.indicators_to_include is not None
        else default_indicators
    )

    try:
        # --- SMA ---
        if "SMA" in indicators_to_run:
            sma_period = inputs.sma_period or settings.DEFAULT_SMA_PERIOD
            sma_input = SmaInput(
                symbol=inputs.symbol,
                timeframe=inputs.timeframe,
                history_len=inputs.history_len,
                period=sma_period,
            )
            sma_output = await calculate_sma(ctx, sma_input)
            indicator_results_structured["sma"] = sma_output.model_dump()

            if sma_output.sma is not None and len(sma_output.sma) > 0:
                latest_sma = sma_output.sma[-1]
                report_sections.append(
                    f"- SMA({sma_output.period}): {latest_sma:.4f} (Latest)"
                )
                if len(sma_output.sma) > 1:
                    trend = "↗" if sma_output.sma[-1] > sma_output.sma[-2] else "↘"
                    report_sections.append(
                        f"  - Trend: {trend} ({len(sma_output.sma)} data points)"
                    )
            elif sma_output.error:
                report_sections.append(f"- SMA: Error - {sma_output.error}")

        # --- RSI ---
        if "RSI" in indicators_to_run:
            rsi_period = inputs.rsi_period or settings.DEFAULT_RSI_PERIOD
            rsi_input = RsiInput(
                symbol=inputs.symbol,
                timeframe=inputs.timeframe,
                history_len=inputs.history_len,
                period=rsi_period,
            )
            rsi_output = await calculate_rsi(ctx, rsi_input)
            indicator_results_structured["rsi"] = rsi_output.model_dump()

            if rsi_output.rsi is not None and len(rsi_output.rsi) > 0:
                latest_rsi = rsi_output.rsi[-1]
                report_sections.append(f"- RSI({rsi_output.period}): {latest_rsi:.2f}")
                if latest_rsi > 70:
                    report_sections.append(
                        "  - Note: RSI suggests overbought conditions (>70)."
                    )
                elif latest_rsi < 30:
                    report_sections.append(
                        "  - Note: RSI suggests oversold conditions (<30)."
                    )

                # Add trend analysis if we have multiple data points
                if len(rsi_output.rsi) > 1:
                    trend = "↗" if rsi_output.rsi[-1] > rsi_output.rsi[-2] else "↘"
                    report_sections.append(
                        f"  - Trend: {trend} ({len(rsi_output.rsi)} data points)"
                    )
            elif rsi_output.error:
                report_sections.append(f"- RSI: Error - {rsi_output.error}")

        # --- MACD ---
        if "MACD" in indicators_to_run:
            macd_input = MacdInput(
                symbol=inputs.symbol,
                timeframe=inputs.timeframe,
                history_len=inputs.history_len,
                fast_period=inputs.macd_fast_period or settings.DEFAULT_MACD_FAST,
                slow_period=inputs.macd_slow_period or settings.DEFAULT_MACD_SLOW,
                signal_period=inputs.macd_signal_period or settings.DEFAULT_MACD_SIGNAL,
            )
            macd_output = await calculate_macd(ctx, macd_input)
            indicator_results_structured["macd"] = macd_output.model_dump()

            if (
                macd_output.macd is not None
                and len(macd_output.macd) > 0
                and macd_output.signal is not None
                and len(macd_output.signal) > 0
                and macd_output.histogram is not None
                and len(macd_output.histogram) > 0
            ):
                latest_macd = macd_output.macd[-1]
                latest_signal = macd_output.signal[-1]
                latest_hist = macd_output.histogram[-1]

                report_sections.append(
                    f"- MACD({macd_output.fast_period},{macd_output.slow_period},{macd_output.signal_period}): "
                    f"MACD: {latest_macd:.4f}, Signal: {latest_signal:.4f}, Hist: {latest_hist:.4f}"
                )

                if latest_hist > 0 and latest_macd > latest_signal:
                    report_sections.append(
                        "  - Note: MACD histogram positive, potentially bullish momentum."
                    )
                elif latest_hist < 0 and latest_macd < latest_signal:
                    report_sections.append(
                        "  - Note: MACD histogram negative, potentially bearish momentum."
                    )

                # Add trend analysis
                if len(macd_output.histogram) > 1:
                    hist_trend = (
                        "↗"
                        if macd_output.histogram[-1] > macd_output.histogram[-2]
                        else "↘"
                    )
                    report_sections.append(
                        f"  - Histogram Trend: {hist_trend} ({len(macd_output.histogram)} data points)"
                    )

            elif macd_output.error:
                report_sections.append(f"- MACD: Error - {macd_output.error}")

        # --- Bollinger Bands (BBANDS) ---
        if "BBANDS" in indicators_to_run:
            bbands_input = BbandsInput(
                symbol=inputs.symbol,
                timeframe=inputs.timeframe,
                history_len=inputs.history_len,
                period=inputs.bbands_period or settings.DEFAULT_BBANDS_PERIOD,
            )
            bbands_output = await calculate_bbands(ctx, bbands_input)
            indicator_results_structured["bbands"] = bbands_output.model_dump()

            if (
                bbands_output.upper_band is not None
                and len(bbands_output.upper_band) > 0
                and bbands_output.middle_band is not None
                and len(bbands_output.middle_band) > 0
                and bbands_output.lower_band is not None
                and len(bbands_output.lower_band) > 0
            ):
                latest_upper = bbands_output.upper_band[-1]
                latest_middle = bbands_output.middle_band[-1]
                latest_lower = bbands_output.lower_band[-1]

                report_sections.append(
                    f"- Bollinger Bands({bbands_output.period}, {bbands_output.nbdevup}dev): "
                    f"Upper: {latest_upper:.4f}, Middle: {latest_middle:.4f}, Lower: {latest_lower:.4f}"
                )
                report_sections.append(
                    f"  - Band Width: {latest_upper - latest_lower:.4f}"
                )

                # Add trend analysis
                if len(bbands_output.middle_band) > 1:
                    trend = (
                        "↗"
                        if bbands_output.middle_band[-1] > bbands_output.middle_band[-2]
                        else "↘"
                    )
                    report_sections.append(
                        f"  - Middle Band Trend: {trend} ({len(bbands_output.middle_band)} data points)"
                    )

            elif bbands_output.error:
                report_sections.append(f"- BBANDS: Error - {bbands_output.error}")

        # --- Average True Range (ATR) ---
        if "ATR" in indicators_to_run:
            atr_input = AtrInput(
                symbol=inputs.symbol,
                timeframe=inputs.timeframe,
                history_len=inputs.history_len,
                period=inputs.atr_period or settings.DEFAULT_ATR_PERIOD,
            )
            atr_output = await calculate_atr(ctx, atr_input)
            indicator_results_structured["atr"] = atr_output.model_dump()

            if atr_output.atr is not None and len(atr_output.atr) > 0:
                latest_atr = atr_output.atr[-1]
                report_sections.append(
                    f"- ATR({atr_output.period}): {latest_atr:.4f} (Volatility Measure)"
                )

                # Add trend analysis for volatility
                if len(atr_output.atr) > 1:
                    vol_trend = "↗" if atr_output.atr[-1] > atr_output.atr[-2] else "↘"
                    report_sections.append(
                        f"  - Volatility Trend: {vol_trend} ({len(atr_output.atr)} data points)"
                    )

            elif atr_output.error:
                report_sections.append(f"- ATR: Error - {atr_output.error}")

        # --- Average Directional Index (ADX) ---
        if "ADX" in indicators_to_run:
            adx_input = AdxInput(
                symbol=inputs.symbol,
                timeframe=inputs.timeframe,
                history_len=inputs.history_len,
                period=inputs.adx_period or settings.DEFAULT_ADX_PERIOD,
            )
            adx_output = await calculate_adx(ctx, adx_input)
            indicator_results_structured["adx"] = adx_output.model_dump()

            if (
                adx_output.adx is not None
                and len(adx_output.adx) > 0
                and adx_output.plus_di is not None
                and len(adx_output.plus_di) > 0
                and adx_output.minus_di is not None
                and len(adx_output.minus_di) > 0
            ):
                latest_adx = adx_output.adx[-1]
                latest_plus_di = adx_output.plus_di[-1]
                latest_minus_di = adx_output.minus_di[-1]

                report_sections.append(
                    f"- ADX({adx_output.period}): {latest_adx:.2f}, +DI: {latest_plus_di:.2f}, -DI: {latest_minus_di:.2f}"
                )

                if latest_adx > 25:
                    report_sections.append(
                        f"  - Note: ADX ({latest_adx:.2f}) suggests a trending market."
                    )
                else:
                    report_sections.append(
                        f"  - Note: ADX ({latest_adx:.2f}) suggests a weak or non-trending market."
                    )

                # Directional analysis
                if latest_plus_di > latest_minus_di:
                    report_sections.append("  - Direction: Bullish (+DI > -DI)")
                else:
                    report_sections.append("  - Direction: Bearish (-DI > +DI)")

                # Add trend analysis
                if len(adx_output.adx) > 1:
                    trend_strength = (
                        "↗" if adx_output.adx[-1] > adx_output.adx[-2] else "↘"
                    )
                    report_sections.append(
                        f"  - Trend Strength: {trend_strength} ({len(adx_output.adx)} data points)"
                    )

            elif adx_output.error:
                report_sections.append(f"- ADX: Error - {adx_output.error}")

        # --- On-Balance Volume (OBV) ---
        if "OBV" in indicators_to_run:
            obv_input = ObvInput(
                symbol=inputs.symbol,
                timeframe=inputs.timeframe,
                history_len=inputs.history_len,
                data_points=inputs.obv_data_points or settings.DEFAULT_OBV_DATA_POINTS,
            )
            obv_output = await calculate_obv(ctx, obv_input)
            indicator_results_structured["obv"] = obv_output.model_dump()

            if obv_output.obv is not None and len(obv_output.obv) > 0:
                latest_obv = obv_output.obv[-1]
                report_sections.append(
                    f"- OBV (using {obv_output.data_points} points): {latest_obv:.2f}"
                )

                # Add trend analysis for volume flow
                if len(obv_output.obv) > 1:
                    volume_trend = (
                        "↗" if obv_output.obv[-1] > obv_output.obv[-2] else "↘"
                    )
                    report_sections.append(
                        f"  - Volume Flow: {volume_trend} ({len(obv_output.obv)} data points)"
                    )

            elif obv_output.error:
                report_sections.append(f"- OBV: Error - {obv_output.error}")

        # --- Synthesize the report ---
        if not report_sections:
            return ComprehensiveAnalysisOutput(
                **output_base,
                error="No indicator data could be calculated or selected.",
            )

        report_title = f"Comprehensive Technical Analysis for {inputs.symbol} ({inputs.timeframe}) - {inputs.history_len} Data Points:\n"
        report_text = report_title + "\n".join(report_sections)

        # Add summary section with overall trend analysis
        summary_sections = []
        trend_indicators = []

        # Collect trend information from indicators that have it
        for indicator_name, indicator_data in indicator_results_structured.items():
            if isinstance(indicator_data, dict) and not indicator_data.get("error"):
                if indicator_name == "sma" and indicator_data.get("sma"):
                    sma_data = indicator_data["sma"]
                    if len(sma_data) > 1:
                        trend_indicators.append(
                            f"SMA: {'↗' if sma_data[-1] > sma_data[-2] else '↘'}"
                        )

                elif indicator_name == "rsi" and indicator_data.get("rsi"):
                    rsi_data = indicator_data["rsi"]
                    if len(rsi_data) > 1:
                        trend_indicators.append(
                            f"RSI: {'↗' if rsi_data[-1] > rsi_data[-2] else '↘'}"
                        )

        if trend_indicators:
            summary_sections.append("\nTrend Summary:")
            summary_sections.extend([f"  {trend}" for trend in trend_indicators])
            report_text += "\n" + "\n".join(summary_sections)

        await ctx.info(
            f"Successfully generated comprehensive report for {inputs.symbol} with {inputs.history_len} data points."
        )
        return ComprehensiveAnalysisOutput(
            **output_base,
            report_text=report_text,
            structured_data=indicator_results_structured,
        )

    except Exception as e:
        await ctx.error(
            f"Unexpected error in generate_comprehensive_market_report for {inputs.symbol}: {e}",
            exc_info=True,
        )
        return ComprehensiveAnalysisOutput(
            **output_base, error=f"An unexpected server error occurred: {str(e)}"
        )
