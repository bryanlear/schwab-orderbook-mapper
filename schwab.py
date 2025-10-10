import argparse
import base64
import json
import logging
import os
import sys
import time
import webbrowser
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from urllib.parse import parse_qs, urlparse

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import requests


LOGGER = logging.getLogger("schwab_mapper")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


API_BASE = "https://api.schwabapi.com"
AUTH_PATH = "/v1/oauth/token"
AUTH_URL = "https://api.schwabapi.com/v1/oauth/authorize"
REDIRECT_URI = "https://127.0.0.1"
TOKEN_STORE = Path(".schwab_tokens.json")
ENV_PATH = Path(".venv")


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
        received_at = received_at or datetime.utcnow()
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
            "expires_at": self.expires_at.isoformat(),
            "refresh_expires_at": self.refresh_expires_at.isoformat()
            if self.refresh_expires_at
            else None,
        }

    @classmethod
    def from_json(cls, data: Dict[str, object]) -> "TokenBundle":
        expires_at = datetime.fromisoformat(str(data["expires_at"]))
        refresh_expires = data.get("refresh_expires_at")
        refresh_dt = (
            datetime.fromisoformat(str(refresh_expires)) if refresh_expires else None
        )
        return cls(
            access_token=str(data["access_token"]),
            refresh_token=str(data["refresh_token"]),
            expires_at=expires_at,
            refresh_expires_at=refresh_dt,
        )

    def is_valid(self, skew_seconds: int = 60) -> bool:
        return datetime.utcnow() + timedelta(seconds=skew_seconds) < self.expires_at

    def can_refresh(self) -> bool:
        if not self.refresh_expires_at:
            return True
        return datetime.utcnow() < self.refresh_expires_at


class SchwabAuthManager:
    def __init__(self, app_key: str, app_secret: str, token_store: Path = TOKEN_STORE):
        self.app_key = app_key
        self.app_secret = app_secret
        self.token_store = token_store
        self._token_bundle: Optional[TokenBundle] = None

    @property
    def authorization_url(self) -> str:
        return f"{AUTH_URL}?client_id={self.app_key}&redirect_uri={REDIRECT_URI}"

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
        self.token_store.write_text(json.dumps(bundle.to_json(), indent=2), encoding="utf-8")
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
        return {"Authorization": f"Bearer {self.auth_manager.get_access_token()}"}

    def get_order_book(self, symbol: str, depth: int = 20) -> Dict[str, object]:
        params = {"depth": depth}
        url = f"{API_BASE}/marketdata/v1/{symbol}/orderbook"
        response = requests.get(url, headers=self._auth_header(), params=params, timeout=10)
        if response.status_code == 404:
            LOGGER.error("Symbol '%s' not found when requesting order book", symbol)
            return {"bids": [], "asks": []}
        response.raise_for_status()
        return response.json()

    def get_times_sales(
        self,
        symbol: str,
        since: Optional[datetime] = None,
        limit: int = 200,
    ) -> List[Dict[str, object]]:
        params: Dict[str, object] = {"limit": min(limit, 500)}
        if since:
            params["startTime"] = since.isoformat()
        url = f"{API_BASE}/marketdata/v1/{symbol}/timesales"
        response = requests.get(url, headers=self._auth_header(), params=params, timeout=10)
        if response.status_code == 404:
            LOGGER.error("Symbol '%s' not found when requesting time & sales", symbol)
            return []
        if response.status_code == 204:
            return []
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict) and "data" in data:
            return data["data"]
        if isinstance(data, dict) and "candles" in data:
            return data["candles"]
        if isinstance(data, list):
            return data
        LOGGER.debug("Unexpected time & sales payload format: %s", data)
        return []


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
        now = datetime.utcnow()
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
                    ts = datetime.fromtimestamp(float(raw_time))
                else:
                    try:
                        ts = datetime.fromisoformat(str(raw_time).replace("Z", "+00:00"))
                    except ValueError:
                        continue
                trade_times.append(mdates.date2num(ts))
                price_value = trade.get("price") or trade.get("P") or trade.get("p")
                size_value = trade.get("size") or trade.get("quantity") or trade.get("Q") or trade.get("q")
                if price_value is None or size_value is None:
                    continue
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
        bids_raw = order_book.get("bids") or order_book.get("bid") or []
        asks_raw = order_book.get("asks") or order_book.get("ask") or []
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
                price_value = trade.get("price") or trade.get("P") or trade.get("p")
                if price_value is None:
                    continue
                raw_time = trade.get("time") or trade.get("timestamp") or trade.get("datetime")
                if raw_time is None:
                    candidate_time = datetime.utcnow()
                elif isinstance(raw_time, (int, float)):
                    candidate_time = datetime.fromtimestamp(float(raw_time))
                else:
                    try:
                        candidate_time = datetime.fromisoformat(str(raw_time).replace("Z", "+00:00"))
                    except ValueError:
                        candidate_time = datetime.utcnow()
                if not latest_trade or candidate_time >= latest_time:
                    latest_trade = float(price_value)
                    latest_time = candidate_time
                size_value = trade.get("size") or trade.get("quantity") or trade.get("Q") or trade.get("q")
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
        self.timestamps.append(datetime.utcnow())
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
    last_trade_fetch = datetime.utcnow() - timedelta(seconds=trade_lookback)
    while True:
        try:
            order_book = client.get_order_book(symbol=symbol, depth=depth)
            trades = client.get_times_sales(symbol=symbol, since=last_trade_fetch)
            last_trade_fetch = datetime.utcnow()
            mapper.update(order_book, trades)
        except KeyboardInterrupt:
            LOGGER.info("Stopping mapper loop")
            break
        except requests.HTTPError as exc:
            LOGGER.error("HTTP error while polling Schwab API: %s", exc)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.exception("Unexpected error while polling Schwab API: %s", exc)
        time.sleep(max(interval, 0.5))


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
    start = datetime.utcnow()
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
            now = datetime.utcnow()
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
        help="Run one-time OAuth flow and store tokens",
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

    if args.authorize:
        app_key, app_secret = load_credentials()
        auth_manager = SchwabAuthManager(app_key=app_key, app_secret=app_secret)
        authorize_flow(auth_manager)
        return 0

    mapper = OrderBookMapper(window=args.window)

    if args.mock:
        simulate_market(
            mapper=mapper,
            symbol=args.symbol,
            depth=args.depth,
            interval=args.interval,
            duration=args.mock_duration if args.mock_duration > 0 else None,
            seed=args.mock_seed,
        )
        return 0

    app_key, app_secret = load_credentials()
    auth_manager = SchwabAuthManager(app_key=app_key, app_secret=app_secret)

    client = SchwabMarketDataClient(auth_manager=auth_manager)
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