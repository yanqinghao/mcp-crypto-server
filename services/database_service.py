# services/database_service.py

import logging
from datetime import datetime
from typing import Any

from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

from config.settings import settings


class DatabaseService:
    """Service for interacting with InfluxDB database for cryptocurrency data"""

    def __init__(self, database_url: str):
        """
        Initialize database service with connection URL

        Args:
            database_url: InfluxDB connection URL
        """
        # Parse the URL to extract components
        # Format: influxdb://username:password@hostname:port/database
        try:
            if database_url.startswith("influxdb://"):
                # Remove the protocol prefix
                url_parts = database_url[len("influxdb://") :].split("@")

                if len(url_parts) == 2:
                    # Extract credentials and host parts
                    credentials, host_parts = url_parts
                    username, password = credentials.split(":")

                    # Extract host, port, and database
                    host_db_parts = host_parts.split("/")
                    host_port = host_db_parts[0].split(":")
                    host = host_port[0]
                    port = int(host_port[1]) if len(host_port) > 1 else 8086
                    bucket = host_db_parts[1] if len(host_db_parts) > 1 else "crypto_data"
                else:
                    # No credentials provided
                    host_parts = url_parts[0]
                    host_db_parts = host_parts.split("/")
                    host_port = host_db_parts[0].split(":")
                    host = host_port[0]
                    port = int(host_port[1]) if len(host_port) > 1 else 8086
                    bucket = host_db_parts[1] if len(host_db_parts) > 1 else "crypto_data"
                    username, password = None, None

                # Configure InfluxDB client
                self.url = f"http://{host}:{port}"
                self.token = f"{username}:{password}" if username and password else ""
                self.org = settings.influx_org if hasattr(settings, "influx_org") else "my-org"
                self.bucket = bucket
            else:
                # Use provided URL directly
                self.url = database_url
                self.token = settings.influx_token if hasattr(settings, "influx_token") else ""
                self.org = settings.influx_org if hasattr(settings, "influx_org") else "my-org"
                self.bucket = (
                    settings.influx_bucket if hasattr(settings, "influx_bucket") else "crypto_data"
                )

        except Exception as e:
            logging.error(f"Error parsing InfluxDB URL: {e}")
            # Use default values
            self.url = "http://localhost:8086"
            self.token = ""
            self.org = "my-org"
            self.bucket = "crypto_data"

        # Initialize InfluxDB client
        self.client = InfluxDBClient(url=self.url, token=self.token, org=self.org)
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
        self.query_api = self.client.query_api()

    def save_ticker_data(self, data: dict[str, Any]) -> bool:
        """
        Save ticker data to InfluxDB

        Args:
            data: Ticker data dictionary with exchange, symbol, price, etc.

        Returns:
            True if successful, False otherwise
        """
        try:
            point = (
                Point("ticker")
                .tag("exchange", data["exchange"])
                .tag("symbol", data["symbol"])
                .field("price", float(data["price"]))
                .field("volume", float(data.get("volume", 0)))
                .field("bid", float(data.get("bid", 0)))
                .field("ask", float(data.get("ask", 0)))
            )

            # Set timestamp if provided
            if "timestamp" in data:
                if isinstance(data["timestamp"], str):
                    timestamp = datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))
                    point = point.time(timestamp)
                elif isinstance(data["timestamp"], (int, float)):
                    # Assuming milliseconds timestamp
                    timestamp = datetime.fromtimestamp(data["timestamp"] / 1000)
                    point = point.time(timestamp)

            self.write_api.write(bucket=self.bucket, record=point)
            return True
        except Exception as e:
            logging.error(f"Error saving ticker data: {e}")
            return False

    def save_ohlcv_data(
        self, data: list[dict[str, Any]], exchange: str, symbol: str, timeframe: str
    ) -> bool:
        """
        Save OHLCV data to InfluxDB

        Args:
            data: List of OHLCV data points
            exchange: Exchange name
            symbol: Trading pair symbol
            timeframe: Timeframe (e.g., '1m', '1h', '1d')

        Returns:
            True if successful, False otherwise
        """
        try:
            points = []

            for candle in data:
                # Convert timestamp to datetime if it's string
                if isinstance(candle.get("timestamp"), str):
                    timestamp = datetime.fromisoformat(candle["timestamp"].replace("Z", "+00:00"))
                elif isinstance(candle.get("timestamp"), (int, float)):
                    # Assuming milliseconds timestamp
                    timestamp = datetime.fromtimestamp(candle["timestamp"] / 1000)
                else:
                    timestamp = datetime.utcnow()

                point = (
                    Point("ohlcv")
                    .tag("exchange", exchange)
                    .tag("symbol", symbol)
                    .tag("timeframe", timeframe)
                    .field("open", float(candle["open"]))
                    .field("high", float(candle["high"]))
                    .field("low", float(candle["low"]))
                    .field("close", float(candle["close"]))
                    .field("volume", float(candle["volume"]))
                    .time(timestamp)
                )

                points.append(point)

            self.write_api.write(bucket=self.bucket, record=points)
            return True
        except Exception as e:
            logging.error(f"Error saving OHLCV data: {e}")
            return False

    def save_orderbook_data(self, data: dict[str, Any]) -> bool:
        """
        Save orderbook data to InfluxDB

        Args:
            data: Orderbook data dictionary

        Returns:
            True if successful, False otherwise
        """
        try:
            # Extract timestamp
            if "timestamp" in data:
                if isinstance(data["timestamp"], str):
                    timestamp = datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))
                elif isinstance(data["timestamp"], (int, float)):
                    # Assuming milliseconds timestamp
                    timestamp = datetime.fromtimestamp(data["timestamp"] / 1000)
                else:
                    timestamp = datetime.utcnow()
            else:
                timestamp = datetime.utcnow()

            # Calculate orderbook summary metrics
            best_bid = data["bids"][0][0] if data["bids"] else 0
            best_ask = data["asks"][0][0] if data["asks"] else 0
            spread = best_ask - best_bid if best_bid and best_ask else 0
            bid_volume = sum(float(bid[1]) for bid in data["bids"][:5]) if data["bids"] else 0
            ask_volume = sum(float(ask[1]) for ask in data["asks"][:5]) if data["asks"] else 0

            point = (
                Point("orderbook")
                .tag("exchange", data["exchange"])
                .tag("symbol", data["symbol"])
                .field("best_bid", float(best_bid))
                .field("best_ask", float(best_ask))
                .field("spread", float(spread))
                .field("bid_volume", float(bid_volume))
                .field("ask_volume", float(ask_volume))
                .time(timestamp)
            )

            self.write_api.write(bucket=self.bucket, record=point)
            return True
        except Exception as e:
            logging.error(f"Error saving orderbook data: {e}")
            return False

    def get_latest_ticker(self, exchange: str, symbol: str) -> dict[str, Any] | None:
        """
        Get latest ticker data for a symbol

        Args:
            exchange: Exchange name
            symbol: Trading pair symbol

        Returns:
            Latest ticker data or None if not found
        """
        try:
            query = f'''
            from(bucket: "{self.bucket}")
                |> range(start: -1h)
                |> filter(fn: (r) => r._measurement == "ticker")
                |> filter(fn: (r) => r.exchange == "{exchange}")
                |> filter(fn: (r) => r.symbol == "{symbol}")
                |> last()
            '''

            tables = self.query_api.query(query=query, org=self.org)

            # Process results
            if not tables or len(tables) == 0:
                return None

            result = {
                "exchange": exchange,
                "symbol": symbol,
                "timestamp": None,
                "price": None,
                "volume": None,
                "bid": None,
                "ask": None,
            }

            for table in tables:
                for record in table.records:
                    field = record.get_field()
                    value = record.get_value()
                    time = record.get_time()

                    if field in result:
                        result[field] = value

                    if result["timestamp"] is None or time > result["timestamp"]:
                        result["timestamp"] = time

            # Convert timestamp to ISO format
            if result["timestamp"]:
                result["timestamp"] = result["timestamp"].isoformat()

            return result
        except Exception as e:
            logging.error(f"Error getting latest ticker: {e}")
            return None

    def get_ohlcv_data(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        start_time: datetime | str,
        end_time: datetime | str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get OHLCV data for a symbol within a time range

        Args:
            exchange: Exchange name
            symbol: Trading pair symbol
            timeframe: Timeframe (e.g., '1m', '1h', '1d')
            start_time: Start time as datetime or ISO string
            end_time: End time as datetime or ISO string (optional)

        Returns:
            List of OHLCV data points
        """
        try:
            # Convert start_time to datetime if it's a string
            if isinstance(start_time, str):
                start_time = datetime.fromisoformat(start_time.replace("Z", "+00:00"))

            # Convert end_time to datetime if it's a string
            if end_time is None:
                end_time = datetime.utcnow()
            elif isinstance(end_time, str):
                end_time = datetime.fromisoformat(end_time.replace("Z", "+00:00"))

            # Format times for Flux query
            start_str = start_time.isoformat() + "Z"
            end_str = end_time.isoformat() + "Z"

            query = f'''
            from(bucket: "{self.bucket}")
                |> range(start: {start_str}, stop: {end_str})
                |> filter(fn: (r) => r._measurement == "ohlcv")
                |> filter(fn: (r) => r.exchange == "{exchange}")
                |> filter(fn: (r) => r.symbol == "{symbol}")
                |> filter(fn: (r) => r.timeframe == "{timeframe}")
                |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            '''

            tables = self.query_api.query(query=query, org=self.org)

            # Process results
            result = []
            for table in tables:
                for record in table.records:
                    candle = {
                        "timestamp": record.get_time().isoformat(),
                        "open": record.values.get("open", 0),
                        "high": record.values.get("high", 0),
                        "low": record.values.get("low", 0),
                        "close": record.values.get("close", 0),
                        "volume": record.values.get("volume", 0),
                    }
                    result.append(candle)

            return result
        except Exception as e:
            logging.error(f"Error getting OHLCV data: {e}")
            return []

    def get_orderbook_history(
        self,
        exchange: str,
        symbol: str,
        start_time: datetime | str,
        end_time: datetime | str | None = None,
        aggregation: str = "1m",
    ) -> list[dict[str, Any]]:
        """
        Get orderbook history for a symbol within a time range

        Args:
            exchange: Exchange name
            symbol: Trading pair symbol
            start_time: Start time as datetime or ISO string
            end_time: End time as datetime or ISO string (optional)
            aggregation: Time aggregation (e.g., '1m', '5m', '1h')

        Returns:
            List of orderbook history data points
        """
        try:
            # Convert start_time to datetime if it's a string
            if isinstance(start_time, str):
                start_time = datetime.fromisoformat(start_time.replace("Z", "+00:00"))

            # Convert end_time to datetime if it's a string
            if end_time is None:
                end_time = datetime.utcnow()
            elif isinstance(end_time, str):
                end_time = datetime.fromisoformat(end_time.replace("Z", "+00:00"))

            # Format times for Flux query
            start_str = start_time.isoformat() + "Z"
            end_str = end_time.isoformat() + "Z"

            query = f'''
            from(bucket: "{self.bucket}")
                |> range(start: {start_str}, stop: {end_str})
                |> filter(fn: (r) => r._measurement == "orderbook")
                |> filter(fn: (r) => r.exchange == "{exchange}")
                |> filter(fn: (r) => r.symbol == "{symbol}")
                |> aggregateWindow(every: {aggregation}, fn: mean)
                |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            '''

            tables = self.query_api.query(query=query, org=self.org)

            # Process results
            result = []
            for table in tables:
                for record in table.records:
                    snapshot = {
                        "timestamp": record.get_time().isoformat(),
                        "best_bid": record.values.get("best_bid", 0),
                        "best_ask": record.values.get("best_ask", 0),
                        "spread": record.values.get("spread", 0),
                        "bid_volume": record.values.get("bid_volume", 0),
                        "ask_volume": record.values.get("ask_volume", 0),
                    }
                    result.append(snapshot)

            return result
        except Exception as e:
            logging.error(f"Error getting orderbook history: {e}")
            return []

    def create_database_if_not_exists(self) -> bool:
        """
        Create the database/bucket if it doesn't exist

        Returns:
            True if successful, False otherwise
        """
        try:
            buckets_api = self.client.buckets_api()
            bucket = buckets_api.find_bucket_by_name(self.bucket)

            if bucket is None:
                buckets_api.create_bucket(bucket_name=self.bucket, org=self.org)
                logging.info(f"Created InfluxDB bucket: {self.bucket}")

            return True
        except Exception as e:
            logging.error(f"Error creating database: {e}")
            return False

    def close(self):
        """Close database connection"""
        if self.client:
            self.client.close()
