import argparse
import asyncio
import base64
import contextlib
import json
import logging
import os
import sys
import threading
import time
import webbrowser
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from urllib.parse import parse_qs, quote_plus, urlparse

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import requests

try:  # pragma: no cover - optional dependency for streaming
    import websocket  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    websocket = None  # type: ignore


LOGGER = logging.getLogger("schwab_mapper")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


API_BASE = "https://api.schwabapi.com"
AUTH_PATH = "/v1/oauth/token"
AUTH_URL = "https://api.schwabapi.com/v1/oauth/authorize"
MD_SCOPE_HINT = "marketdata"  # Schwab labels market data scope this way in their docs
REDIRECT_URI = "https://127.0.0.1"
BASE_DIR = Path(__file__).resolve().parent
TOKEN_STORE = BASE_DIR / ".schwab_tokens.json"
SCHWAB_PY_TOKEN_STORE = BASE_DIR / ".schwab-py-token.json"
ENV_PATH = BASE_DIR / ".venv"
MAX_STREAM_DEPTH = 10
BOOK_FIELDS_PER_LEVEL = 4


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


def ensure_utc(dt: datetime) -> datetime:
    """Normalize a datetime to UTC with timezone information."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def isoformat_utc(dt: datetime) -> str:
    """Render a datetime as an ISO-8601 UTC string with trailing Z."""
    return ensure_utc(dt).isoformat().replace("+00:00", "Z")


def load_credentials(env_path: Path = ENV_PATH) -> tuple[str, str]:
    app_key = os.getenv("APP_KEY")
    app_secret = os.getenv("APP_SECRET")

    if app_key and app_secret:
        return app_key, app_secret

    if not env_path.exists():
        raise FileNotFoundError(
            "Expected credentials in environment variables or in '.venv' next to schwab.py"
        )

    parsed: Dict[str, str] = {}
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        cleaned = raw.strip()
        if not cleaned or cleaned.startswith("#"):
            continue
        if "=" not in cleaned:
            continue
        key, value = cleaned.split("=", maxsplit=1)
        parsed[key.strip()] = value.strip().strip('"\'')

    app_key = parsed.get("APP_KEY", app_key)
    app_secret = parsed.get("APP_SECRET", app_secret)

    if not app_key or not app_secret:
        raise ValueError(
            "APP_KEY and APP_SECRET must be supplied via environment variables or '.venv'"
        )

    return app_key, app_secret


@dataclass
class TokenBundle:
    access_token: str
    refresh_token: str
    expires_at: datetime
    refresh_expires_at: Optional[datetime]

    @classmethod
    def from_response(
        cls, payload: Dict[str, object], received_at: Optional[datetime] = None
    ) -> "TokenBundle":
        received_at = received_at or utc_now()
        expires_in = int(payload.get("expires_in", 0))
        refresh_expires_in = payload.get("refresh_token_expires_in")
        refresh_dt = (
            received_at + timedelta(seconds=int(refresh_expires_in))
            if refresh_expires_in
            else None
        )
        return cls(
            access_token=str(payload["access_token"]),
            refresh_token=str(payload["refresh_token"]),
            expires_at=received_at + timedelta(seconds=expires_in),
            refresh_expires_at=refresh_dt,
        )

    def to_json(self) -> Dict[str, object]:
        return {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "expires_at": isoformat_utc(self.expires_at),
            "refresh_expires_at": isoformat_utc(self.refresh_expires_at)
            if self.refresh_expires_at
            else None,
        }

    @classmethod
    def from_json(cls, data: Dict[str, object]) -> "TokenBundle":
        expires_at = ensure_utc(datetime.fromisoformat(str(data["expires_at"]).replace("Z", "+00:00")))
        refresh_expires = data.get("refresh_expires_at")
        refresh_dt = (
            ensure_utc(datetime.fromisoformat(str(refresh_expires).replace("Z", "+00:00")))
            if refresh_expires
            else None
        )
        return cls(
            access_token=str(data["access_token"]),
            refresh_token=str(data["refresh_token"]),
            expires_at=expires_at,
            refresh_expires_at=refresh_dt,
        )

    def is_valid(self, skew_seconds: int = 60) -> bool:
        return utc_now() + timedelta(seconds=skew_seconds) < self.expires_at

    def can_refresh(self) -> bool:
        if not self.refresh_expires_at:
            return True
        return utc_now() < self.refresh_expires_at


class SchwabAuthManager:
    def __init__(self, app_key: str, app_secret: str, token_store: Path = TOKEN_STORE):
        self.app_key = app_key
        self.app_secret = app_secret
        self.token_store = token_store
        self._token_bundle: Optional[TokenBundle] = None

    @property
    def authorization_url(self) -> str:
        hint = f"&scope={MD_SCOPE_HINT}" if MD_SCOPE_HINT else ""
        return f"{AUTH_URL}?client_id={self.app_key}&redirect_uri={REDIRECT_URI}{hint}"

    def _load_tokens(self) -> Optional[TokenBundle]:
        if self._token_bundle:
            return self._token_bundle
        if not self.token_store.exists():
            return None
        try:
            data = json.loads(self.token_store.read_text(encoding="utf-8"))
            self._token_bundle = TokenBundle.from_json(data)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("Failed to read cached tokens: %s", exc)
            return None
        return self._token_bundle

    def _save_tokens(self, bundle: TokenBundle) -> None:
        content = json.dumps(bundle.to_json(), indent=2).encode("utf-8")
        fd = os.open(str(self.token_store), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, content)
        finally:
            os.close(fd)
        self._token_bundle = bundle

    def exchange_code_for_token(self, authorization_response: str) -> TokenBundle:
        parsed = urlparse(authorization_response.strip())
        query = parse_qs(parsed.query)
        auth_code_values = query.get("code")
        if not auth_code_values:
            raise ValueError("The response URL must contain an authorization code parameter")
        auth_code = auth_code_values[0]
        payload = {
            "grant_type": "authorization_code",
            "code": auth_code,
            "redirect_uri": REDIRECT_URI,
        }
        return self._request_token(payload)

    def _refresh_token(self, bundle: TokenBundle) -> TokenBundle:
        payload = {
            "grant_type": "refresh_token",
            "refresh_token": bundle.refresh_token,
        }
        return self._request_token(payload)

    def _request_token(self, payload: Dict[str, str]) -> TokenBundle:
        credentials = f"{self.app_key}:{self.app_secret}"
        encoded_credentials = base64.b64encode(credentials.encode("utf-8")).decode("utf-8")
        headers = {
            "Authorization": f"Basic {encoded_credentials}",
            "Content-Type": "application/x-www-form-urlencoded",
        }
        response = requests.post(
            url=f"{API_BASE}{AUTH_PATH}",
            headers=headers,
            data=payload,
            timeout=15,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            LOGGER.error(
                "Token request failed (%s): %s",
                exc,
                response.text.strip() or "<empty response>",
            )
            raise
        bundle = TokenBundle.from_response(response.json())
        self._save_tokens(bundle)
        LOGGER.info("Token bundle updated; access token valid until %s", bundle.expires_at)
        return bundle

    def get_access_token(self) -> str:
        bundle = self._load_tokens()
        if bundle and bundle.is_valid():
            return bundle.access_token
        if bundle and bundle.can_refresh():
            LOGGER.info("Refreshing Schwab access token")
            bundle = self._refresh_token(bundle)
            return bundle.access_token
        raise RuntimeError("No valid Schwab tokens available; run with --authorize to store a token")


class SchwabMarketDataClient:
    def __init__(self, auth_manager: SchwabAuthManager):
        self.auth_manager = auth_manager

    def _auth_header(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.auth_manager.get_access_token()}",
            "X-Schwab-Client-Key": self.auth_manager.app_key,
            "Accept": "application/json",
            "User-Agent": "schwab-orderbook-mapper/1.0",
        }

    def get_order_book(self, symbol: str, depth: int = 20) -> Dict[str, object]:
        upper_symbol = symbol.upper()
        headers = self._auth_header()
        attempts = [
            (
                f"{API_BASE}/marketdata/v1/orderbook",
                {
                    "symbols": upper_symbol,
                    "symbol": upper_symbol,
                    "depth": depth,
                    "apikey": self.auth_manager.app_key,
                },
            ),
            (
                f"{API_BASE}/marketdata/v1/orderbook/{upper_symbol}",
                {"depth": depth, "apikey": self.auth_manager.app_key},
            ),
            (
                f"{API_BASE}/marketdata/v1/{upper_symbol}/orderbook",
                {"depth": depth, "apikey": self.auth_manager.app_key},
            ),
        ]
        responses: List[requests.Response] = []
        for url, params in attempts:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            parsed = self._parse_order_book_response(response, upper_symbol)
            if parsed is not None:
                return parsed
            responses.append(response)

        formatted_status = ", ".join(
            f"{resp.status_code} {resp.text.strip() or '<empty>'} ({resp.request.method} {resp.url})"
            for resp in responses
        )
        LOGGER.error("Unable to retrieve order book for '%s'. Attempts=%s", symbol, formatted_status)
        return {"bids": [], "asks": []}

    def _parse_order_book_response(
        self, response: requests.Response, upper_symbol: str
    ) -> Optional[Dict[str, object]]:
        if response.status_code == 404:
            LOGGER.debug(
                "Order book endpoint %s returned 404 for %s",
                response.url,
                upper_symbol,
            )
            return None
        if response.status_code == 204:
            LOGGER.info("No order book content for %s", upper_symbol)
            return {"bids": [], "asks": []}
        if response.status_code >= 400:
            LOGGER.warning(
                "Order book request for %s failed (%s): %s",
                upper_symbol,
                response.status_code,
                response.text.strip() or "<empty response>",
            )
            return None
        try:
            payload = response.json()
        except ValueError:
            LOGGER.warning("Order book response for %s was not JSON", upper_symbol)
            return None
        if isinstance(payload, dict):
            candidate = payload.get(upper_symbol)
            if isinstance(candidate, dict):
                return candidate
            if "data" in payload and isinstance(payload["data"], list):
                for item in payload["data"]:
                    if not isinstance(item, dict):
                        continue
                    item_symbol = str(item.get("symbol", "")).upper()
                    if item_symbol == upper_symbol:
                        book = item.get("orderBook") or item
                        if isinstance(book, dict):
                            return book
            if "orderBooks" in payload and isinstance(payload["orderBooks"], list):
                for item in payload["orderBooks"]:
                    if not isinstance(item, dict):
                        continue
                    item_symbol = str(item.get("symbol", "")).upper()
                    if item_symbol == upper_symbol:
                        return item
            if "orderBook" in payload and isinstance(payload["orderBook"], dict):
                return payload["orderBook"]
            if "bids" in payload and "asks" in payload:
                return payload
        LOGGER.debug("Order book payload for %s not understood: %s", upper_symbol, payload)
        return None

    def get_times_sales(
        self,
        symbol: str,
        since: Optional[datetime] = None,
        limit: int = 200,
    ) -> List[Dict[str, object]]:
        upper_symbol = symbol.upper()
        params: Dict[str, object] = {
            "symbol": upper_symbol,
            "symbols": upper_symbol,
            "limit": min(limit, 500),
            "apikey": self.auth_manager.app_key,
        }
        if since:
            params["startTime"] = isoformat_utc(since)
        headers = self._auth_header()
        attempts: List[tuple[str, Dict[str, object]]] = [
            (f"{API_BASE}/marketdata/v1/timesales", params),
            (
                f"{API_BASE}/marketdata/v1/timesales/{upper_symbol}",
                {k: v for k, v in params.items() if k != "symbols"},
            ),
            (
                f"{API_BASE}/marketdata/v1/{upper_symbol}/timesales",
                {k: v for k, v in params.items() if k != "symbols"},
            ),
        ]
        responses: List[requests.Response] = []
        for url, attempt_params in attempts:
            response = requests.get(url, headers=headers, params=attempt_params, timeout=10)
            parsed = self._parse_timesales_response(response, upper_symbol)
            if parsed is not None:
                return parsed
            responses.append(response)

        formatted_status = ", ".join(
            f"{resp.status_code} {resp.text.strip() or '<empty>'} ({resp.request.method} {resp.url})"
            for resp in responses
        )
        LOGGER.error("Unable to retrieve time & sales for '%s'. Attempts=%s", symbol, formatted_status)
        return []

    def _parse_timesales_response(
        self, response: requests.Response, upper_symbol: str
    ) -> Optional[List[Dict[str, object]]]:
        if response.status_code == 404:
            LOGGER.debug("Time & sales endpoint %s returned 404 for %s", response.url, upper_symbol)
            return None
        if response.status_code == 204:
            return []
        if response.status_code >= 400:
            LOGGER.warning(
                "Time & sales request for %s failed (%s): %s",
                upper_symbol,
                response.status_code,
                response.text.strip() or "<empty response>",
            )
            return None
        try:
            data = response.json()
        except ValueError:
            LOGGER.warning("Time & sales response for %s was not JSON", upper_symbol)
            return None
        if isinstance(data, dict):
            if "data" in data and isinstance(data["data"], list):
                return data["data"]
            if "timesales" in data and isinstance(data["timesales"], list):
                return data["timesales"]
            if "candles" in data and isinstance(data["candles"], list):
                return data["candles"]
            if "series" in data and isinstance(data["series"], list):
                return data["series"]
        if isinstance(data, list):
            return data
        LOGGER.debug("Unexpected time & sales payload format for %s: %s", upper_symbol, data)
        return None

    def get_streamer_metadata(self) -> Dict[str, object]:
        headers = self._auth_header()
        params = {
            "fields": "streamerConnectionInfo,streamerSubscriptionKeys,surrogateIds",
        }
        response = requests.get(
            f"{API_BASE}/userinfo/v1/userinfo",
            headers=headers,
            params=params,
            timeout=10,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            LOGGER.error(
                "Failed to retrieve streamer metadata (%s): %s",
                exc,
                response.text.strip() or "<empty response>",
            )
            raise
        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError("Streamer metadata response was not valid JSON") from exc

        root: object = payload.get("response") if isinstance(payload, dict) else payload
        if isinstance(root, list) and root:
            root = root[0]
        if not isinstance(root, dict):
            raise RuntimeError("Unexpected structure in streamer metadata response")

        connection: object = (
            root.get("streamerConnectionInfo")
            or root.get("streamerInfo")
            or root.get("streamerConnectionInfos")
        )
        if isinstance(connection, list):
            connection = connection[0] if connection else None
        if not isinstance(connection, dict):
            raise RuntimeError("Streamer connection info missing from Schwab response")

        subscription_keys: List[str] = []
        subscription_section = root.get("streamerSubscriptionKeys")
        if isinstance(subscription_section, dict):
            candidates = subscription_section.get("keys") or subscription_section.get("keyList")
            if isinstance(candidates, list):
                for item in candidates:
                    if isinstance(item, dict) and item.get("key"):
                        subscription_keys.append(str(item["key"]))
                    elif isinstance(item, str):
                        subscription_keys.append(item)

        if not subscription_keys:
            LOGGER.warning("Schwab response did not include streamer subscription keys; subscriptions may fail")

        account_hash = (
            subscription_keys[0]
            if subscription_keys
            else connection.get("accountId")
            or connection.get("accountNumber")
        )
        if not account_hash:
            raise RuntimeError("Unable to determine account hash for streaming subscriptions")

        return {
            "raw": payload,
            "connection_info": connection,
            "subscription_keys": subscription_keys,
            "account_hash": str(account_hash),
        }


class StreamEventHandler:
    def __init__(self, mapper: "OrderBookMapper", depth: int):
        self.mapper = mapper
        self.depth = max(1, min(depth, MAX_STREAM_DEPTH))
        self._pending_trades: List[Dict[str, object]] = []
        self._lock = threading.Lock()

    def add_trade(self, trade: Dict[str, object]) -> None:
        with self._lock:
            self._pending_trades.append(trade)

    def emit_book(self, order_book: Dict[str, object]) -> None:
        with self._lock:
            trades = list(self._pending_trades)
            self._pending_trades.clear()
        self.mapper.update(order_book, trades if trades else None)


def _import_schwab_py() -> tuple:
    """Import schwab-py modules while avoiding name collisions with this script."""
    moved_entry: Optional[str] = None
    if sys.path and Path(sys.path[0]).resolve() == BASE_DIR:
        moved_entry = sys.path.pop(0)
        sys.path.append(moved_entry)
    try:
        from schwab.auth import easy_client as _easy_client  # type: ignore
        from schwab.streaming import StreamClient as _StreamClient  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "Streaming requires the 'schwab-py' package. Install it with 'pip install schwab-py'."
        ) from exc
    finally:
        if moved_entry is not None:
            sys.path.pop()
            sys.path.insert(0, moved_entry)
    return _easy_client, _StreamClient


class SchwabPyBookDispatcher:
    def __init__(self, symbol: str, handler: StreamEventHandler, depth: int):
        self.symbol = symbol.upper()
        self.handler = handler
        self.depth = max(1, min(depth, MAX_STREAM_DEPTH))

    def __call__(self, message: Dict[str, object]) -> None:
        book = self._extract_book(message)
        if book:
            self.handler.emit_book(book)

    def _extract_book(self, message: Dict[str, object]) -> Optional[Dict[str, object]]:
        contents = message.get("content")
        if not isinstance(contents, list):
            return None
        bids: List[Dict[str, float]] = []
        asks: List[Dict[str, float]] = []
        for entry in contents:
            if not isinstance(entry, dict):
                continue
            entry_symbol = str(
                entry.get("SYMBOL") or entry.get("symbol") or entry.get("key") or ""
            ).upper()
            if entry_symbol and entry_symbol != self.symbol:
                continue
            self._collect_levels(
                storage=bids,
                levels=entry.get("BIDS"),
                price_key="BID_PRICE",
                total_key="TOTAL_VOLUME",
                per_exchange_key="BIDS",
                per_exchange_total_key="BID_VOLUME",
            )
            self._collect_levels(
                storage=asks,
                levels=entry.get("ASKS"),
                price_key="ASK_PRICE",
                total_key="TOTAL_VOLUME",
                per_exchange_key="ASKS",
                per_exchange_total_key="ASK_VOLUME",
            )
        if bids or asks:
            return {"bids": bids, "asks": asks}
        return None

    def _collect_levels(
        self,
        *,
        storage: List[Dict[str, float]],
        levels: object,
        price_key: str,
        total_key: str,
        per_exchange_key: str,
        per_exchange_total_key: str,
    ) -> None:
        if not isinstance(levels, list):
            return
        for level in levels:
            if len(storage) >= self.depth:
                break
            if not isinstance(level, dict):
                continue
            price = level.get(price_key)
            total = level.get(total_key)
            if total in (None, 0):
                total = self._sum_per_exchange(level.get(per_exchange_key), per_exchange_total_key)
            if price is None or total is None:
                continue
            try:
                price_value = float(price)
                size_value = float(total)
            except (TypeError, ValueError):
                continue
            if size_value <= 0:
                continue
            storage.append({"price": price_value, "size": size_value})

    @staticmethod
    def _sum_per_exchange(levels: object, volume_key: str) -> Optional[float]:
        if not isinstance(levels, list):
            return None
        total = 0.0
        found = False
        for entry in levels:
            if not isinstance(entry, dict):
                continue
            volume = entry.get(volume_key)
            if volume is None:
                continue
            try:
                total += float(volume)
            except (TypeError, ValueError):
                continue
            found = True
        return total if found else None


class SchwabPyTradeDispatcher:
    def __init__(self, symbol: str, handler: StreamEventHandler):
        self.symbol = symbol.upper()
        self.handler = handler

    def __call__(self, message: Dict[str, object]) -> None:
        trades = self._extract_trades(message)
        for trade in trades:
            self.handler.add_trade(trade)

    def _extract_trades(self, message: Dict[str, object]) -> List[Dict[str, object]]:
        contents = message.get("content")
        if not isinstance(contents, list):
            return []
        trades: List[Dict[str, object]] = []
        for entry in contents:
            if not isinstance(entry, dict):
                continue
            entry_symbol = str(
                entry.get("SYMBOL") or entry.get("symbol") or entry.get("key") or ""
            ).upper()
            if entry_symbol and entry_symbol != self.symbol:
                continue
            price = entry.get("LAST_PRICE")
            size = entry.get("LAST_SIZE")
            if price is None or size is None:
                continue
            try:
                price_value = float(price)
                size_value = float(size)
            except (TypeError, ValueError):
                continue
            if size_value <= 0:
                continue
            timestamp = self._parse_timestamp(entry)
            trade: Dict[str, object] = {"price": price_value, "size": size_value}
            if timestamp is not None:
                trade["time"] = timestamp
            trades.append(trade)
        return trades

    @staticmethod
    def _parse_timestamp(entry: Dict[str, object]) -> Optional[str]:
        millis_source = entry.get("TRADE_TIME_MILLIS") or entry.get("QUOTE_TIME_MILLIS")
        if millis_source is None:
            return None
        try:
            millis = float(millis_source)
        except (TypeError, ValueError):
            return None
        dt = datetime.fromtimestamp(millis / 1000.0, tz=timezone.utc)
        return dt.isoformat()


def _resolve_book_services(selection: str) -> List[str]:
    normalized = (selection or "nasdaq").strip().lower()
    if normalized == "nyse":
        return ["nyse_book"]
    if normalized == "auto":
        return ["nasdaq_book"]
    return ["nasdaq_book"]


class SchwabStreamerClient:
    def __init__(
        self,
        auth_manager: SchwabAuthManager,
        metadata: Dict[str, object],
        qos_level: int = 2,
    ) -> None:
        if websocket is None:  # pragma: no cover - informative failure
            raise RuntimeError(
                "Streaming requires the 'websocket-client' package. Install it with 'pip install websocket-client'."
            )
        self.auth_manager = auth_manager
        self.metadata = metadata
        self.connection_info: Dict[str, object] = dict(metadata.get("connection_info", {}))
        self.account_hash: str = str(metadata.get("account_hash"))
        if not self.connection_info or not self.account_hash:
            raise RuntimeError("Incomplete streamer metadata supplied")
        self.qos_level = max(0, min(int(qos_level), 5))
        self._ws: Optional[object] = None
        self._request_id = 0
        self._logged_in = False
        self._symbols: List[str] = []
        self._depth = 1
        self._include_trades = True
        self._handler: Optional[StreamEventHandler] = None

    def run(
        self,
        symbols: List[str],
        depth: int,
        handler: StreamEventHandler,
        include_trades: bool = True,
    ) -> None:
        self._symbols = [symbol.upper() for symbol in symbols]
        if not self._symbols:
            raise ValueError("At least one symbol must be supplied for streaming")
        self._depth = max(1, min(depth, MAX_STREAM_DEPTH))
        self._include_trades = include_trades
        self._handler = handler
        url = self._socket_url()
        LOGGER.info("Connecting to Schwab streamer at %s", url)
        self._ws = websocket.WebSocketApp(
            url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_close=self._on_close,
            on_error=self._on_error,
        )
        try:
            self._ws.run_forever(ping_interval=30, ping_timeout=15)
        finally:
            LOGGER.info("Streamer session terminated")

    def close(self) -> None:
        if self._ws is not None:
            LOGGER.debug("Closing streamer socket")
            self._ws.close()

    def _socket_url(self) -> str:
        info = self.connection_info
        host = str(info.get("streamerSocketUrl") or info.get("streamerUrl") or "").strip()
        if not host:
            raise RuntimeError("Streamer metadata omitted the socket hostname")
        port_raw = info.get("streamerSocketPort") or info.get("streamerPort") or 443
        try:
            port = int(port_raw)
        except (TypeError, ValueError):
            port = 443
        resource = str(info.get("streamerSocketResource") or info.get("streamerResource") or "/ws")
        protocol = "wss"
        if host.startswith("ws://") or host.startswith("wss://"):
            return f"{host}{resource if resource.startswith('/') else '/' + resource}" if resource else host
        return f"{protocol}://{host}:{port}{resource if resource.startswith('/') else '/' + resource}"

    def _on_open(self, _: object) -> None:
        LOGGER.debug("Streamer socket opened")
        self._send(self._build_login_request())

    def _on_message(self, _: object, message: str) -> None:
        try:
            payload = json.loads(message)
        except ValueError:
            LOGGER.debug("Streamer message was not JSON: %s", message)
            return

        if isinstance(payload, dict):
            if "notify" in payload:
                LOGGER.info("Streamer notify: %s", payload["notify"])
            if "response" in payload:
                self._handle_responses([payload["response"]])
            if "responses" in payload:
                self._handle_responses(payload.get("responses") or [])
            if "data" in payload:
                self._handle_data(payload.get("data") or [])

    def _on_close(self, _: object, status_code: int, message: str) -> None:
        LOGGER.info("Streamer socket closed (code=%s, message=%s)", status_code, message)

    def _on_error(self, _: object, error: object) -> None:
        LOGGER.error("Streamer error: %s", error)

    def _handle_responses(self, responses: List[object]) -> None:
        for response in responses:
            if not isinstance(response, dict):
                continue
            service = response.get("service")
            command = response.get("command")
            content = response.get("content")
            if service == "ADMIN" and command == "LOGIN":
                entries = content if isinstance(content, list) else [content]
                success = False
                for entry in entries:
                    if isinstance(entry, dict) and str(entry.get("code")) == "0":
                        success = True
                        break
                if success:
                    LOGGER.info("Streamer login acknowledged")
                    self._logged_in = True
                    self._after_login()
                else:
                    LOGGER.error("Streamer login failed: %s", content)
                    self.close()

    def _handle_data(self, data_messages: List[object]) -> None:
        if not self._handler:
            return
        for message in data_messages:
            if not isinstance(message, dict):
                continue
            service = message.get("service")
            contents = message.get("content")
            if not isinstance(contents, list):
                continue
            if service == "BOOK":
                for entry in contents:
                    book = self._parse_book(entry)
                    if book:
                        self._handler.emit_book(book)
            elif isinstance(service, str) and service.startswith("TIMESALE"):
                for entry in contents:
                    trade = self._parse_trade(entry)
                    if trade:
                        self._handler.add_trade(trade)

    def _after_login(self) -> None:
        self._send(self._build_qos_request())
        self._send(self._build_book_subscription())
        if self._include_trades:
            try:
                self._send(self._build_timesale_subscription())
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.warning("Unable to subscribe to time & sales: %s", exc)

    def _build_login_request(self) -> Dict[str, object]:
        credential = self._build_credential_string()
        token = (
            self.connection_info.get("token")
            or self.connection_info.get("streamerToken")
            or self.connection_info.get("tokenValue")
        )
        if not token:
            raise RuntimeError("Streamer metadata did not provide a streaming token")
        parameters: Dict[str, object] = {
            "credential": credential,
            "token": token,
            "version": "1.0",
        }
        user_id = self.connection_info.get("userId") or self.connection_info.get("userid")
        if user_id:
            parameters["userid"] = user_id
        return {
            "requests": [
                {
                    "service": "ADMIN",
                    "command": "LOGIN",
                    "requestid": self._next_request_id(),
                    "account": self.account_hash,
                    "source": self._source_id(),
                    "parameters": parameters,
                }
            ]
        }

    def _build_qos_request(self) -> Dict[str, object]:
        return {
            "requests": [
                {
                    "service": "ADMIN",
                    "command": "QOS",
                    "requestid": self._next_request_id(),
                    "account": self.account_hash,
                    "source": self._source_id(),
                    "parameters": {"qoslevel": str(self.qos_level)},
                }
            ]
        }

    def _build_book_subscription(self) -> Dict[str, object]:
        fields = self._book_fields(self._depth)
        return {
            "requests": [
                {
                    "service": "BOOK",
                    "command": "SUBS",
                    "requestid": self._next_request_id(),
                    "account": self.account_hash,
                    "source": self._source_id(),
                    "parameters": {
                        "keys": ",".join(self._symbols),
                        "fields": fields,
                    },
                }
            ]
        }

    def _build_timesale_subscription(self) -> Dict[str, object]:
        service = "TIMESALE_EQUITY"
        return {
            "requests": [
                {
                    "service": service,
                    "command": "SUBS",
                    "requestid": self._next_request_id(),
                    "account": self.account_hash,
                    "source": self._source_id(),
                    "parameters": {
                        "keys": ",".join(self._symbols),
                        "fields": "0,1,2,3,4",
                    },
                }
            ]
        }

    def _next_request_id(self) -> str:
        self._request_id += 1
        return str(self._request_id)

    def _send(self, payload: Dict[str, object]) -> None:
        if not self._ws:
            raise RuntimeError("WebSocket connection is not initialized")
        message = json.dumps(payload)
        LOGGER.debug("Streamer outbound: %s", message)
        self._ws.send(message)

    def _source_id(self) -> str:
        return str(
            self.connection_info.get("appId")
            or self.connection_info.get("appID")
            or self.connection_info.get("streamerAppId")
            or self.auth_manager.app_key
        )

    def _build_credential_string(self) -> str:
        info = self.connection_info
        fields = {
            "userid": info.get("userId") or info.get("userid"),
            "token": info.get("token") or info.get("streamerToken"),
            "company": info.get("company"),
            "segment": info.get("segment"),
            "cddomain": info.get("cdDomain") or info.get("cddomain"),
            "usergroup": info.get("userGroup") or info.get("usergroup"),
            "accesslevel": info.get("accessLevel") or info.get("accesslevel"),
            "authorized": info.get("authorized") or "Y",
            "acl": info.get("acl"),
            "timestamp": info.get("tokenTimestamp") or info.get("tokenTimeStamp"),
            "appid": info.get("appId") or info.get("appID") or info.get("streamerAppId"),
        }
        components = [f"{key}={quote_plus(str(value))}" for key, value in fields.items() if value is not None]
        return "&".join(components)

    def _book_fields(self, depth: int) -> str:
        # Field 0 = symbol, 1 = timestamp; bids occupy fields starting at 2, asks follow.
        fields: List[int] = [0, 1]
        depth = max(1, min(depth, MAX_STREAM_DEPTH))
        for level in range(depth):
            base = 2 + level * BOOK_FIELDS_PER_LEVEL
            fields.extend([base, base + 1])
        ask_base = 2 + MAX_STREAM_DEPTH * BOOK_FIELDS_PER_LEVEL
        for level in range(depth):
            base = ask_base + level * BOOK_FIELDS_PER_LEVEL
            fields.extend([base, base + 1])
        unique_fields = sorted(set(fields))
        return ",".join(str(field) for field in unique_fields)

    def _parse_book(self, entry: object) -> Optional[Dict[str, object]]:
        if not isinstance(entry, dict):
            return None
        bids: List[Dict[str, float]] = []
        asks: List[Dict[str, float]] = []
        depth = self._depth
        for level in range(depth):
            bid_price = self._extract_numeric(entry, 2 + level * BOOK_FIELDS_PER_LEVEL, f"BID_{level + 1}_PRICE")
            bid_size = self._extract_numeric(entry, 3 + level * BOOK_FIELDS_PER_LEVEL, f"BID_{level + 1}_SIZE")
            if bid_price is not None and bid_size is not None and bid_size > 0:
                bids.append({"price": bid_price, "size": bid_size})

        ask_base = 2 + MAX_STREAM_DEPTH * BOOK_FIELDS_PER_LEVEL
        for level in range(depth):
            ask_price = self._extract_numeric(entry, ask_base + level * BOOK_FIELDS_PER_LEVEL, f"ASK_{level + 1}_PRICE")
            ask_size = self._extract_numeric(entry, ask_base + level * BOOK_FIELDS_PER_LEVEL + 1, f"ASK_{level + 1}_SIZE")
            if ask_price is not None and ask_size is not None and ask_size > 0:
                asks.append({"price": ask_price, "size": ask_size})

        if not bids and not asks:
            return None
        return {"bids": bids, "asks": asks}

    def _parse_trade(self, entry: object) -> Optional[Dict[str, object]]:
        if not isinstance(entry, dict):
            return None
        price = self._extract_numeric(entry, 2, "LAST_PRICE")
        size = self._extract_numeric(entry, 3, "LAST_SIZE")
        if price is None or size is None:
            return None
        raw_time = entry.get("1") or entry.get("TRADE_TIME") or entry.get("time")
        timestamp: Optional[datetime] = None
        if raw_time is not None:
            try:
                if isinstance(raw_time, (int, float)):
                    value = float(raw_time)
                else:
                    value = float(str(raw_time))
                if value > 1e12:
                    value /= 1000.0
                timestamp = datetime.fromtimestamp(value, tz=timezone.utc)
            except Exception:
                try:
                    timestamp = ensure_utc(datetime.fromisoformat(str(raw_time).replace("Z", "+00:00")))
                except Exception:
                    timestamp = None
        trade: Dict[str, object] = {
            "price": price,
            "size": size,
        }
        if timestamp is not None:
            trade["time"] = timestamp.isoformat()
        return trade

    def _extract_numeric(self, entry: Dict[str, object], index: int, fallback_key: str) -> Optional[float]:
        candidates = [entry.get(str(index)), entry.get(fallback_key), entry.get(fallback_key.lower())]
        for value in candidates:
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return None


def _prepare_level(level: Dict[str, object]) -> tuple[float, float]:
    price_keys = ("price", "P", "p")
    size_keys = ("size", "quantity", "qty", "Q", "q")
    price = next((float(level[key]) for key in price_keys if key in level), None)
    size = next((float(level[key]) for key in size_keys if key in level), None)
    if price is None or size is None:
        raise ValueError(f"Level missing price or size fields: {level}")
    return price, size


class OrderBookMapper:
    def __init__(self, window: int = 120):
        self.window = window
        self.snapshots: List[np.ndarray] = []
        self.timestamps: List[datetime] = []
        self.price_grid: List[float] = []
        self.price_index: Dict[float, int] = {}
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.volume_ax = self.ax.twinx()
        self.cmap = plt.get_cmap("viridis")
        self.max_intensity = 1.0
        self.mid_prices: List[float] = []
        self.last_trade_prices: List[float] = []
        self.trade_volumes: List[float] = []

    def _ensure_grid(self, prices: Iterable[float]) -> None:
        incoming = set(round(p, 4) for p in prices)
        if not incoming:
            return
        merged = sorted(set(self.price_grid) | incoming)
        if merged == self.price_grid:
            return
        if not self.price_grid:
            self.price_grid = merged
            self.price_index = {price: idx for idx, price in enumerate(self.price_grid)}
            return
        expanded_history: List[np.ndarray] = []
        new_index = {price: idx for idx, price in enumerate(merged)}
        for snapshot in self.snapshots:
            expanded = np.zeros(len(merged))
            for price, old_idx in self.price_index.items():
                expanded[new_index[price]] = snapshot[old_idx]
            expanded_history.append(expanded)
        self.snapshots = expanded_history
        self.price_grid = merged
        self.price_index = new_index

    def _build_snapshot(self, bids: List[tuple[float, float]], asks: List[tuple[float, float]]) -> np.ndarray:
        snapshot = np.zeros(len(self.price_grid))
        for price, size in bids:
            idx = self.price_index.get(round(price, 4))
            if idx is not None:
                snapshot[idx] -= size
        for price, size in asks:
            idx = self.price_index.get(round(price, 4))
            if idx is not None:
                snapshot[idx] += size
        return snapshot

    def _update_plot(self, trades: Optional[List[Dict[str, object]]] = None) -> None:
        if not self.snapshots:
            return
        data = np.stack(self.snapshots).T
        now = utc_now()
        if not self.timestamps:
            self.timestamps.append(now)
        times_num = [mdates.date2num(ts) for ts in self.timestamps]
        if len(times_num) == 1:
            imshow_times = [times_num[0] - 1 / 1440, times_num[0]]
        else:
            imshow_times = times_num
        data_to_plot = data
        extent = [imshow_times[0], imshow_times[-1], self.price_grid[0], self.price_grid[-1]]
        self.ax.clear()
        self.volume_ax.cla()
        vmax = max(self.max_intensity, np.max(np.abs(data_to_plot)))
        self.max_intensity = max(self.max_intensity, vmax)
        self.ax.imshow(
            data_to_plot,
            aspect="auto",
            origin="lower",
            cmap=self.cmap,
            extent=extent,
            vmin=-vmax,
            vmax=vmax,
        )
        self.ax.set_ylabel("Price")
        self.ax.xaxis_date()
        self.fig.autofmt_xdate()
        self.volume_ax.set_ylabel("Volume")
        if trades:
            trade_times: List[float] = []
            trade_prices: List[float] = []
            trade_sizes: List[float] = []
            for trade in trades:
                raw_time = trade.get("time") or trade.get("timestamp") or trade.get("datetime")
                if not raw_time:
                    continue
                if isinstance(raw_time, (int, float)):
                    ts = datetime.fromtimestamp(float(raw_time), tz=timezone.utc)
                else:
                    try:
                        ts = ensure_utc(
                            datetime.fromisoformat(str(raw_time).replace("Z", "+00:00"))
                        )
                    except ValueError:
                        continue
                price_value = next((trade[k] for k in ("price", "P", "p") if trade.get(k) is not None), None)
                size_value = next((trade[k] for k in ("size", "quantity", "Q", "q") if trade.get(k) is not None), None)
                if price_value is None or size_value is None:
                    continue
                trade_times.append(mdates.date2num(ts))
                trade_prices.append(float(price_value))
                trade_sizes.append(float(size_value))
            if trade_times:
                marker_sizes = np.clip(np.array(trade_sizes) * 0.1, 15, 150)
                self.ax.scatter(
                    trade_times,
                    trade_prices,
                    s=marker_sizes,
                    c="#ffffff",
                    edgecolors="none",
                    alpha=0.7,
                )
        if len(self.mid_prices) >= 1:
            self.ax.plot(
                times_num,
                self.mid_prices,
                color="#ffdd57",
                linewidth=1.2,
                label="Mid Price",
            )
        if len(self.last_trade_prices) >= 1:
            self.ax.plot(
                times_num,
                self.last_trade_prices,
                color="#ff4136",
                linewidth=1.0,
                linestyle="--",
                label="Last Trade",
            )
        volume_poly = None
        if self.trade_volumes:
            volume_poly = self.volume_ax.fill_between(
                times_num,
                self.trade_volumes,
                step="mid",
                alpha=0.25,
                color="#7FDBFF",
                label="Volume",
            )
            self.volume_ax.set_ylim(bottom=0)
            self.volume_ax.tick_params(axis="y", labelcolor="#0074D9")
        handles, labels = [], []
        for line in self.ax.lines:
            if line.get_label() and not line.get_label().startswith("_line"):
                handles.append(line)
                labels.append(line.get_label())
        if volume_poly is not None:
            handles.append(volume_poly)
            labels.append("Volume")
        if handles:
            self.ax.legend(handles, labels, loc="upper left")
        self.ax.set_title("Schwab Order Book Mapper")
        self.ax.set_xlabel("Time")
        self.fig.canvas.draw()
        plt.pause(0.001)

    def update(
        self,
        order_book: Dict[str, object],
        trades: Optional[List[Dict[str, object]]] = None,
    ) -> None:
        bids_raw = order_book.get("bids")
        if bids_raw is None:
            bids_raw = order_book.get("bid")
        if bids_raw is None:
            bids_raw = []
        asks_raw = order_book.get("asks")
        if asks_raw is None:
            asks_raw = order_book.get("ask")
        if asks_raw is None:
            asks_raw = []
        bids: List[tuple[float, float]] = []
        asks: List[tuple[float, float]] = []
        for raw in bids_raw:
            try:
                bids.append(_prepare_level(raw))
            except ValueError:
                continue
        for raw in asks_raw:
            try:
                asks.append(_prepare_level(raw))
            except ValueError:
                continue
        self._ensure_grid([price for price, _ in bids + asks])
        if not self.price_grid:
            LOGGER.warning("No price levels to plot yet")
            return
        best_bid = max((price for price, _ in bids), default=None)
        best_ask = min((price for price, _ in asks), default=None)
        mid_price: Optional[float] = None
        if best_bid is not None and best_ask is not None:
            mid_price = (best_bid + best_ask) / 2
        elif best_bid is not None:
            mid_price = best_bid
        elif best_ask is not None:
            mid_price = best_ask

        last_trade_price: Optional[float] = None
        volume_total = 0.0
        if trades:
            latest_trade = None
            latest_time = None
            for trade in trades:
                price_value = next((trade[k] for k in ("price", "P", "p") if trade.get(k) is not None), None)
                if price_value is None:
                    continue
                raw_time = next((trade[k] for k in ("time", "timestamp", "datetime") if trade.get(k) is not None), None)
                if raw_time is None:
                    candidate_time = utc_now()
                elif isinstance(raw_time, (int, float)):
                    candidate_time = datetime.fromtimestamp(float(raw_time), tz=timezone.utc)
                else:
                    try:
                        candidate_time = ensure_utc(
                            datetime.fromisoformat(str(raw_time).replace("Z", "+00:00"))
                        )
                    except ValueError:
                        candidate_time = utc_now()
                if not latest_trade or candidate_time >= latest_time:
                    latest_trade = float(price_value)
                    latest_time = candidate_time
                size_value = next((trade[k] for k in ("size", "quantity", "Q", "q") if trade.get(k) is not None), None)
                if size_value is not None:
                    volume_total += float(size_value)
            if latest_trade is not None:
                last_trade_price = latest_trade
        if last_trade_price is None and self.last_trade_prices:
            last_trade_price = self.last_trade_prices[-1]
        if last_trade_price is None:
            last_trade_price = mid_price

        snapshot = self._build_snapshot(bids, asks)
        self.snapshots.append(snapshot)
        self.timestamps.append(utc_now())
        if mid_price is None and self.mid_prices:
            mid_price = self.mid_prices[-1]
        self.mid_prices.append(mid_price if mid_price is not None else np.nan)
        self.last_trade_prices.append(last_trade_price if last_trade_price is not None else np.nan)
        self.trade_volumes.append(volume_total)
        if len(self.snapshots) > self.window:
            self.snapshots = self.snapshots[-self.window :]
            self.timestamps = self.timestamps[-self.window :]
            self.mid_prices = self.mid_prices[-self.window :]
            self.last_trade_prices = self.last_trade_prices[-self.window :]
            self.trade_volumes = self.trade_volumes[-self.window :]
        self._update_plot(trades=trades)


def authorize_flow(auth_manager: SchwabAuthManager) -> None:
    url = auth_manager.authorization_url
    LOGGER.info("Launching default browser for Schwab authorization")
    LOGGER.info(url)
    opened = webbrowser.open(url)
    if not opened:
        LOGGER.warning("Could not open browser automatically; copy the URL above into your browser")
    LOGGER.info("After approving, paste the full redirect URL including the code parameter:")
    response_url = input("Redirect URL: ").strip()
    bundle = auth_manager.exchange_code_for_token(response_url)
    LOGGER.info("Access token stored; valid until %s", bundle.expires_at)


def poll_market_data(
    client: SchwabMarketDataClient,
    mapper: OrderBookMapper,
    symbol: str,
    depth: int,
    interval: float,
    trade_lookback: int,
) -> None:
    last_trade_fetch = utc_now() - timedelta(seconds=trade_lookback)
    try:
        while True:
            try:
                order_book = client.get_order_book(symbol=symbol, depth=depth)
                trades = client.get_times_sales(symbol=symbol, since=last_trade_fetch)
                last_trade_fetch = utc_now()
                mapper.update(order_book, trades)
            except requests.HTTPError as exc:
                LOGGER.error("HTTP error while polling Schwab API: %s", exc)
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.exception("Unexpected error while polling Schwab API: %s", exc)
            time.sleep(max(interval, 0.5))
    except KeyboardInterrupt:
        LOGGER.info("Stopping mapper loop")


def stream_market_data(
    client: SchwabMarketDataClient,
    mapper: OrderBookMapper,
    symbol: str,
    depth: int,
    include_trades: bool,
    qos_level: int,
    stream_venue: str,
    stream_account: Optional[str],
) -> None:
    easy_client, StreamClient = _import_schwab_py()
    if qos_level != 2:
        LOGGER.info(
            "schwab-py does not expose QoS controls; ignoring requested stream QoS level %s",
            qos_level,
        )
    callback_url = os.getenv("SCHWAB_PY_CALLBACK_URL") or REDIRECT_URI
    handler = StreamEventHandler(mapper=mapper, depth=depth)
    symbol_upper = symbol.upper()
    book_services = _resolve_book_services(stream_venue)
    book_dispatcher = SchwabPyBookDispatcher(symbol=symbol_upper, handler=handler, depth=depth)
    trade_dispatcher = SchwabPyTradeDispatcher(symbol=symbol_upper, handler=handler) if include_trades else None

    try:
        http_client = easy_client(
            api_key=client.auth_manager.app_key,
            app_secret=client.auth_manager.app_secret,
            callback_url=callback_url,
            token_path=str(SCHWAB_PY_TOKEN_STORE),
            asyncio=True,
        )
    except Exception as exc:  # pragma: no cover - dependency guard
        LOGGER.error(
            "schwab-py easy_client failed (%s). If this is a callback error, set SCHWAB_PY_CALLBACK_URL to a URL with an explicit port (e.g. https://127.0.0.1:8182) registered with your app.",
            exc,
        )
        raise

    async def _run_stream() -> None:
        stream_client = StreamClient(http_client, account_id=stream_account)
        await stream_client.login()
        for service in book_services:
            add_handler = getattr(stream_client, f"add_{service}_handler")
            add_handler(book_dispatcher)
        if include_trades and trade_dispatcher is not None:
            stream_client.add_level_one_equity_handler(trade_dispatcher)
        for service in book_services:
            subscribe = getattr(stream_client, f"{service}_subs")
            await subscribe([symbol_upper])
        if include_trades and trade_dispatcher is not None:
            quote_fields = [
                StreamClient.LevelOneEquityFields.LAST_PRICE,
                StreamClient.LevelOneEquityFields.LAST_SIZE,
                StreamClient.LevelOneEquityFields.TRADE_TIME_MILLIS,
            ]
            await stream_client.level_one_equity_subs([symbol_upper], fields=list(quote_fields))
        LOGGER.info(
            "Streaming %s order book via schwab-py (%s)",
            symbol_upper,
            ",".join(service.upper() for service in book_services),
        )
        try:
            while True:
                await stream_client.handle_message()
        finally:
            with contextlib.suppress(Exception):
                await stream_client.logout()

    try:
        asyncio.run(_run_stream())
    except KeyboardInterrupt:
        LOGGER.info("Stopping streamer loop")


def simulate_market(
    mapper: OrderBookMapper,
    symbol: str,
    depth: int,
    interval: float,
    duration: Optional[int],
    seed: Optional[int] = None,
) -> None:
    rng = np.random.default_rng(seed)
    price = 100.0
    drift = 0.0
    tick = 0
    start = utc_now()
    end_time = start + timedelta(seconds=duration) if duration else None
    LOGGER.info(
        "Starting mock stream for %s (depth=%s, interval=%ss, duration=%s)",
        symbol,
        depth,
        interval,
        duration or "infinite",
    )
    try:
        while True:
            now = utc_now()
            if end_time and now >= end_time:
                break
            drift += rng.normal(0, 0.02)
            price = max(0.01, price + drift)
            levels = np.arange(1, depth + 1)
            spread = np.maximum(rng.normal(0.01, 0.003), 0.002)
            bid_prices = np.round(price - spread * levels, 4)
            ask_prices = np.round(price + spread * levels, 4)
            base_size = np.abs(rng.normal(100, 40, size=depth)) + 5
            heat_scale = 1.0 + np.abs(np.sin(tick / 20.0))
            bids = [
                {"price": float(p), "size": float(s * heat_scale)}
                for p, s in zip(bid_prices, base_size * rng.uniform(0.5, 1.5, depth))
            ]
            asks = [
                {"price": float(p), "size": float(s * heat_scale)}
                for p, s in zip(ask_prices, base_size * rng.uniform(0.5, 1.5, depth))
            ]
            trades: List[Dict[str, object]] = []
            if rng.random() < 0.7:
                trade_count = rng.integers(1, 4)
                for _ in range(trade_count):
                    ts = now - timedelta(seconds=float(rng.random()) * interval)
                    trades.append(
                        {
                            "price": float(rng.normal(price, spread)),
                            "size": float(np.maximum(rng.normal(80, 30), 5)),
                            "time": ts.isoformat(),
                        }
                    )
            mapper.update({"bids": bids, "asks": asks}, trades=trades)
            tick += 1
            time.sleep(max(interval, 0.2))
    except KeyboardInterrupt:
        LOGGER.info("Mock stream interrupted by user")
    LOGGER.info("Mock stream complete")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Schwab order book heat-map mapper")
    parser.add_argument("symbol", help="Ticker symbol to visualize")
    parser.add_argument("--depth", type=int, default=20, help="Number of levels per side")
    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="Polling interval in seconds",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Use the Schwab Streamer API instead of REST polling",
    )
    parser.add_argument(
        "--no-trades",
        action="store_true",
        help="Disable the time & sales overlay when streaming",
    )
    parser.add_argument(
        "--stream-qos",
        type=int,
        default=2,
        help="Requested Schwab streamer quality-of-service level (legacy Schwab API, ignored by schwab-py)",
    )
    parser.add_argument(
        "--stream-venue",
        choices=["nasdaq", "nyse", "auto"],
        default="auto",
        help="Order book venue to subscribe to when streaming (auto tries NASDAQ first)",
    )
    parser.add_argument(
        "--stream-account",
        help="Optional Schwab account number to bind to the streaming session",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=180,
        help="Maximum number of snapshots to retain in the heat-map",
    )
    parser.add_argument(
        "--trade-lookback",
        type=int,
        default=120,
        help="Seconds of time & sales to request on each poll",
    )
    parser.add_argument(
        "--authorize",
        action="store_true",
        help="Run the OAuth flow before starting the mapper",
    )
    parser.add_argument(
        "--authorize-only",
        action="store_true",
        help="Run the OAuth flow and exit without launching the mapper",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use a synthetic market data simulator instead of live Schwab APIs",
    )
    parser.add_argument(
        "--mock-duration",
        type=int,
        default=0,
        help="Seconds to run mock simulation (0 keeps running until interrupted)",
    )
    parser.add_argument(
        "--mock-seed",
        type=int,
        help="Optional random seed for repeatable mock runs",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    auth_manager: Optional[SchwabAuthManager] = None
    if args.authorize or args.authorize_only or not args.mock:
        app_key, app_secret = load_credentials()
        auth_manager = SchwabAuthManager(app_key=app_key, app_secret=app_secret)

    if args.authorize or args.authorize_only:
        if auth_manager is None:  # pragma: no cover - defensive guard
            raise RuntimeError("Authorization requested but credentials are unavailable")
        authorize_flow(auth_manager)
        if args.authorize_only:
            return 0

    mapper = OrderBookMapper(window=args.window)

    if args.mock:
        if args.stream:
            LOGGER.warning("Streaming mode ignored while --mock is active")
        simulate_market(
            mapper=mapper,
            symbol=args.symbol,
            depth=args.depth,
            interval=args.interval,
            duration=args.mock_duration if args.mock_duration > 0 else None,
            seed=args.mock_seed,
        )
        return 0

    if auth_manager is None:  # pragma: no cover - defensive guard
        raise RuntimeError("Live polling requires credentials; none were loaded")
    client = SchwabMarketDataClient(auth_manager=auth_manager)

    if args.stream:
        stream_market_data(
            client=client,
            mapper=mapper,
            symbol=args.symbol,
            depth=args.depth,
            include_trades=not args.no_trades,
            qos_level=args.stream_qos,
            stream_venue=args.stream_venue,
            stream_account=args.stream_account,
        )
        return 0

    poll_market_data(
        client=client,
        mapper=mapper,
        symbol=args.symbol,
        depth=args.depth,
        interval=args.interval,
        trade_lookback=args.trade_lookback,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())