import os
import json
import time
import logging
from typing import List, Dict, Any, Optional, Tuple

import requests
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials

# ------------------------
# Config & Globals
# ------------------------

OANDA_API_URLS = {
    "practice": "https://api-fxpractice.oanda.com/v3",
    "live": "https://api-fxtrade.oanda.com/v3",
}

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

logger = logging.getLogger(__name__)


# ------------------------
# Google Sheets helpers
# ------------------------

def get_gspread_client() -> gspread.Client:
    """Authenticate to Google Sheets using a service account JSON in env GOOGLE_CREDS_JSON."""
    creds_json = os.environ["GOOGLE_CREDS_JSON"]
    info = json.loads(creds_json)
    credentials = Credentials.from_service_account_info(info, scopes=SCOPES)
    client = gspread.authorize(credentials)
    return client


def write_dataframe_to_sheet(df: pd.DataFrame, sheet_name: str, tab_name: str) -> None:
    """
    Create/replace the screener tab data in columns A:Q ONLY.
    Columns R onward are left untouched so user formulas & notes persist.
    """
    gc = get_gspread_client()
    sh = gc.open(sheet_name)

    try:
        ws = sh.worksheet(tab_name)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(
            title=tab_name,
            rows=str(len(df) + 100),
            cols="30",  # give some room for user columns (R+)
        )

    header = df.columns.tolist()
    data_rows = df.fillna("").astype(str).values.tolist()
    values = [header] + data_rows

    num_data_rows = len(values)
    num_data_cols = len(header)

    max_rows = max(ws.row_count, num_data_rows)
    blank_grid = [[""] * num_data_cols for _ in range(max_rows)]

    for r, row in enumerate(values):
        for c, val in enumerate(row):
            blank_grid[r][c] = val

    ws.update(range_name="A1", values=blank_grid)
    logger.info(
        "Updated sheet '%s' tab '%s' A:Q with %d data rows",
        sheet_name,
        tab_name,
        len(df),
    )


# ------------------------
# Oanda helpers
# ------------------------

def get_oanda_session():
    # If you prefer OANDA_API_KEY, just swap the env var name here.
    token = os.environ["OANDA_API_TOKEN"]
    env = os.getenv("OANDA_ENV", "practice").lower()
    base_url = OANDA_API_URLS.get(env, OANDA_API_URLS["practice"])

    session = requests.Session()
    session.headers.update({
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    })
    return session, base_url


def fetch_instruments(session: requests.Session, base_url: str, account_id: str) -> List[Dict[str, Any]]:
    """Fetch all tradable currency instruments for the account."""
    url = f"{base_url}/accounts/{account_id}/instruments"
    resp = session.get(url)
    resp.raise_for_status()
    data = resp.json()
    instruments = data.get("instruments", [])

    fx_instruments = [
        ins for ins in instruments
        if ins.get("type") == "CURRENCY" and ins.get("tradeable", True)
    ]
    logger.info("Fetched %d FX instruments from Oanda", len(fx_instruments))
    return fx_instruments


def fetch_candles(
    session: requests.Session,
    base_url: str,
    instrument: str,
    granularity: str,
    count: int,
) -> Tuple[List[float], List[float], List[float], List[int]]:
    """Fetch up to `count` candles for an instrument at a given granularity."""
    url = f"{base_url}/instruments/{instrument}/candles"
    params = {
        "granularity": granularity,
        "count": count,
        "price": "M",
    }
    resp = session.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()
    candles = [c for c in data.get("candles", []) if c.get("complete", False)]

    closes = [float(c["mid"]["c"]) for c in candles]
    highs = [float(c["mid"]["h"]) for c in candles]
    lows = [float(c["mid"]["l"]) for c in candles]
    volumes = [int(c.get("volume", 0)) for c in candles]

    return closes, highs, lows, volumes


def fetch_account_summary(session: requests.Session, base_url: str, account_id: str) -> Dict[str, Any]:
    """Fetch account summary (balance, NAV, marginAvailable, etc.)."""
    url = f"{base_url}/accounts/{account_id}/summary"
    resp = session.get(url)
    resp.raise_for_status()
    data = resp.json()
    return data.get("account", {})


def fetch_pricing(
    session: requests.Session,
    base_url: str,
    account_id: str,
    instruments: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Fetch current bid/ask/mid prices for a list of instruments.
    Returns dict: { "EUR_USD": {"bid": ..., "ask": ..., "mid": ...}, ... }
    """
    if not instruments:
        return {}

    url = f"{base_url}/accounts/{account_id}/pricing"
    params = {"instruments": ",".join(instruments)}
    resp = session.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()

    result: Dict[str, Dict[str, float]] = {}

    for p in data.get("prices", []):
        inst = p.get("instrument")
        bids = p.get("bids", [])
        asks = p.get("asks", [])
        if not inst:
            continue

        bid = float(bids[0]["price"]) if bids else None
        ask = float(asks[0]["price"]) if asks else None

        if bid is None and ask is None:
            continue

        if bid is not None and ask is not None:
            mid = (bid + ask) / 2.0
        elif bid is not None:
            mid = bid
        else:
            mid = ask  # type: ignore

        result[inst] = {
            "bid": bid if bid is not None else mid,
            "ask": ask if ask is not None else mid,
            "mid": mid,
        }

    logger.info("Fetched pricing for %d instruments", len(result))
    return result


def place_market_order(
    session: requests.Session,
    base_url: str,
    account_id: str,
    instrument: str,
    units: int,
) -> Dict[str, Any]:
    """
    Place a MARKET order (FOK) for the given instrument and units.
    Positive units = buy; negative = sell.
    """
    url = f"{base_url}/accounts/{account_id}/orders"
    order = {
        "order": {
            "units": str(units),
            "instrument": instrument,
            "timeInForce": "FOK",
            "type": "MARKET",
            "positionFill": "DEFAULT",
        }
    }
    resp = session.post(url, data=json.dumps(order))
    resp.raise_for_status()
    data = resp.json()
    return data


# ------------------------
# Pip helpers
# ------------------------

def build_pip_location_map(
    instruments: List[Dict[str, Any]],
) -> Dict[str, int]:
    """
    Build a dict: { "EUR_USD": -4, "USD_JPY": -2, ... }
    using Oanda's pipLocation from the instruments endpoint.
    """
    pip_map: Dict[str, int] = {}

    for ins in instruments:
        name = ins.get("name")
        pip_loc = ins.get("pipLocation")
        if name is None or pip_loc is None:
            continue
        try:
            pip_map[name] = int(pip_loc)
        except Exception:
            pip_map[name] = -4

    return pip_map


def round_price_to_pip(price: float, pip_location: int) -> float:
    """
    Round a price to the nearest pip using pipLocation.
    Example: pipLocation=-4 => 4 decimal places => 0.0001 pip.
    """
    decimals = max(0, -pip_location)
    return round(price, decimals)


# ------------------------
# Indicator helpers (UT-style logic)
# ------------------------

def compute_atr_series(
    highs: List[float],
    lows: List[float],
    closes: List[float],
    period: int = 10,
) -> Optional[pd.Series]:
    """
    Compute ATR series using Wilder's smoothing (similar to Pine's atr()).
    Returns a pandas Series aligned with input lists, or None if not enough data.
    """
    if len(closes) < period + 1:
        return None

    h = pd.Series(highs, dtype=float)
    l = pd.Series(lows, dtype=float)
    c = pd.Series(closes, dtype=float)

    prev_close = c.shift(1)

    tr = pd.concat(
        [
            (h - l),
            (h - prev_close).abs(),
            (l - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    if len(tr) < period:
        return None

    atr = pd.Series(index=tr.index, dtype=float)

    # First ATR value is simple mean of first 'period' TR values
    atr.iloc[period - 1] = tr.iloc[:period].mean()

    # Wilder's smoothing for remaining values
    for i in range(period, len(tr)):
        atr.iloc[i] = (atr.iloc[i - 1] * (period - 1) + tr.iloc[i]) / period

    return atr


def compute_ut_trailing_stop(
    closes: List[float],
    atr_series: pd.Series,
    atr_mult: float = 1.2,
) -> Optional[pd.Series]:
    """
    SuperTrend-style trailing stop logic, approximating the UT Bot's xATRTrailingStop.
    """
    price = pd.Series(closes, dtype=float)
    atr = atr_series

    first_valid = atr.first_valid_index()
    if first_valid is None:
        return None

    stop = pd.Series(index=price.index, dtype=float)

    # Initialize stop similar to Pine: price - entryLoss on first valid ATR
    entry_loss0 = atr_mult * atr.iloc[first_valid]
    prev_stop = price.iloc[first_valid] - entry_loss0
    stop.iloc[first_valid] = prev_stop

    for i in range(first_valid + 1, len(price)):
        p = price.iloc[i]
        p1 = price.iloc[i - 1]
        prev_stop = stop.iloc[i - 1]
        entry_loss = atr_mult * atr.iloc[i]

        if p > prev_stop and p1 > prev_stop:
            # Long regime: trail up, never down
            stop_val = max(prev_stop, p - entry_loss)
        elif p < prev_stop and p1 < prev_stop:
            # Short regime: trail down, never up
            stop_val = min(prev_stop, p + entry_loss)
        elif p > prev_stop:
            # Flip to long
            stop_val = p - entry_loss
        else:
            # Flip to short
            stop_val = p + entry_loss

        stop.iloc[i] = stop_val

    return stop


def compute_h1_sma(
    closes: List[float],
    length: int = 240,
) -> Optional[float]:
    """Compute simple moving average of H1 closes with given length."""
    if len(closes) < length:
        return None
    s = pd.Series(closes, dtype=float)
    ma = s.rolling(window=length, min_periods=length).mean().iloc[-1]
    return float(ma) if pd.notna(ma) else None


def compute_short_signal_and_metrics(
    h1_closes: List[float],
    h1_highs: List[float],
    h1_lows: List[float],
) -> Optional[Dict[str, Any]]:
    """
    Apply UT-style H1 logic and 240-period H1 SMA filter for SHORT side:

    Condition:
      - Last bar has a UT-style *sell* signal (price crossing below trailing stop)
      - AND last price is above the 240-period H1 SMA
    """
    if len(h1_closes) < 260:  # enough for ATR & SMA
        return None

    # 1) ATR & trailing stop on H1
    atr_series = compute_atr_series(h1_highs, h1_lows, h1_closes, period=10)
    if atr_series is None:
        return None

    stop_series = compute_ut_trailing_stop(h1_closes, atr_series, atr_mult=1.2)
    if stop_series is None:
        return None

    last_idx = len(h1_closes) - 1
    if pd.isna(stop_series.iloc[last_idx]) or pd.isna(stop_series.iloc[last_idx - 1]):
        return None

    price_prev = h1_closes[last_idx - 1]
    price_last = h1_closes[last_idx]
    stop_prev = stop_series.iloc[last_idx - 1]
    stop_last = stop_series.iloc[last_idx]

    # UT-style sell: price crosses *below* the trailing stop on the last bar
    crossed_down = price_prev >= stop_prev and price_last < stop_last
    sell_signal = crossed_down and (price_last < stop_last)

    if not sell_signal:
        return None

    # 2) H1 SMA 240: require price ABOVE MA for short setup
    sma240 = compute_h1_sma(h1_closes, length=240)
    if sma240 is None or sma240 <= 0:
        return None

    if price_last <= sma240:
        # Not above MA, so no short setup
        return None

    pct_above_ma = (price_last / sma240 - 1.0) * 100.0

    result = {
        "last_price": price_last,
        "h1_trailing_stop": float(stop_last),
        "h1_atr10": float(atr_series.iloc[last_idx]),
        "h1_sma240": float(sma240),
        "%_above_h1_sma240": float(pct_above_ma),
        "sell_signal_H1": True,
    }
    return result


# ------------------------
# Combined screener + opener logic
# ------------------------

def run_bot_once():
    """
    - Scan all tradable FX instruments on Oanda using H1 data.
    - For each instrument:
        * If UT-style SHORT condition is met (sell label + price above H1 SMA240),
          attempt to open a short position sized at 0.1% of marginAvailable.
    - Log results to a Google Sheets tab.
    """
    session, base_url = get_oanda_session()
    account_id = os.environ["OANDA_ACCOUNT_ID"]

    # Fetch all FX instruments and build pip map
    instruments_meta = fetch_instruments(session, base_url, account_id)
    instrument_names = [ins.get("name") for ins in instruments_meta if ins.get("name")]
    pip_map = build_pip_location_map(instruments_meta)

    # Fetch account summary (available funds)
    account = fetch_account_summary(session, base_url, account_id)
    margin_available_str = account.get("marginAvailable") or account.get("NAV") or account.get("balance")
    margin_available = float(margin_available_str)

    allocation_percent = float(os.getenv("UT_SHORT_ALLOCATION_PERCENT", "0.1"))  # default 0.1%
    alloc_fraction = allocation_percent / 100.0

    if margin_available <= 0 or alloc_fraction <= 0:
        logger.warning(
            "Non-positive margin_available=%.4f or allocation fraction=%.4f; skipping trading.",
            margin_available,
            alloc_fraction,
        )
        notional_per_trade = 0.0
    else:
        notional_per_trade = margin_available * alloc_fraction

    logger.info(
        "marginAvailable=%.4f, allocation=%.4f%% => notional_per_trade=%.4f",
        margin_available,
        allocation_percent,
        notional_per_trade,
    )

    # First pass: find all candidates with short signals
    candidates: List[Tuple[str, Dict[str, Any]]] = []

    for name in instrument_names:
        try:
            h1_closes, h1_highs, h1_lows, _ = fetch_candles(
                session, base_url, name, granularity="H1", count=400
            )
        except Exception as exc:
            logger.exception("Failed to fetch H1 candles for %s: %s", name, exc)
            continue

        if not h1_closes:
            logger.warning("No H1 data for %s", name)
            continue

        metrics = compute_short_signal_and_metrics(
            h1_closes=h1_closes,
            h1_highs=h1_highs,
            h1_lows=h1_lows,
        )

        if metrics is not None:
            candidates.append((name, metrics))

    if not candidates:
        logger.info("No instruments met UT-sell + above-H1-SMA240 conditions this run.")
        df = pd.DataFrame(columns=[
            "pair",
            "last_price",
            "H1_SMA240",
            "%_above_H1_SMA240",
            "H1_ATR10",
            "H1_trailing_stop",
            "sell_signal_H1",
            "short_units",
            "entry_price_used",
            "order_status",
            "updated_at",
        ])
    else:
        logger.info("Found %d short candidates with UT sell signals.", len(candidates))

        # Fetch pricing for all candidates in one call
        price_map = fetch_pricing(session, base_url, account_id, [pair for pair, _ in candidates])

        rows_for_sheet: List[Dict[str, Any]] = []
        now_str = pd.Timestamp.utcnow().isoformat()

        for pair, metrics in candidates:
            order_status = "NOT_ATTEMPTED"
            short_units: Optional[int] = None
            entry_price_used: Optional[float] = None

            try:
                price_info = price_map.get(pair)
                if price_info is None:
                    logger.warning("No pricing info for %s; skipping order.", pair)
                    order_status = "SKIPPED_NO_PRICE"
                elif notional_per_trade <= 0:
                    order_status = "SKIPPED_NO_FUNDS"
                else:
                    # For shorts, we assume entry near the *bid*
                    bid = price_info["bid"]
                    pip_loc = pip_map.get(pair, -4)
                    rounded_price = round_price_to_pip(bid, pip_loc)

                    if rounded_price <= 0:
                        logger.warning(
                            "Rounded bid price for %s is non-positive (%.8f); skipping.",
                            pair,
                            rounded_price,
                        )
                        order_status = "SKIPPED_BAD_PRICE"
                    else:
                        units_mag = int(round(notional_per_trade / rounded_price))
                        if units_mag <= 0:
                            logger.warning(
                                "Computed units <= 0 for %s (notional=%.4f, price=%.8f); skipping",
                                pair,
                                notional_per_trade,
                                rounded_price,
                            )
                            order_status = "SKIPPED_ZERO_UNITS"
                        else:
                            units = -units_mag  # negative for short
                            logger.info(
                                "Placing SHORT market order: pair=%s, units=%d, approx_price=%.8f (pipLocation=%d)",
                                pair,
                                units,
                                rounded_price,
                                pip_loc,
                            )
                            resp = place_market_order(session, base_url, account_id, pair, units)
                            logger.info("Order response for %s: %s", pair, json.dumps(resp))

                            short_units = units
                            entry_price_used = rounded_price
                            order_status = "FILLED"

            except Exception as exc:
                logger.exception("Failed to place short order for %s: %s", pair, exc)
                order_status = "ERROR"

            row = {
                "pair": pair,
                "last_price": metrics["last_price"],
                "H1_SMA240": metrics["h1_sma240"],
                "%_above_H1_SMA240": metrics["%_above_h1_sma240"],
                "H1_ATR10": metrics["h1_atr10"],
                "H1_trailing_stop": metrics["h1_trailing_stop"],
                "sell_signal_H1": metrics["sell_signal_H1"],
                "short_units": short_units if short_units is not None else "",
                "entry_price_used": entry_price_used if entry_price_used is not None else "",
                "order_status": order_status,
                "updated_at": now_str,
            }
            rows_for_sheet.append(row)

        df = pd.DataFrame(rows_for_sheet)

        columns = [
            "pair",
            "last_price",
            "H1_SMA240",
            "%_above_H1_SMA240",
            "H1_ATR10",
            "H1_trailing_stop",
            "sell_signal_H1",
            "short_units",
            "entry_price_used",
            "order_status",
            "updated_at",
        ]
        df = df[columns]

    sheet_name = os.getenv("GOOGLE_SHEET_NAME", "Active-Investing")
    screener_tab = os.getenv("OANDA_UT_SHORT_TAB", "Oanda-UT-Short-Combined")

    write_dataframe_to_sheet(df, sheet_name, screener_tab)


# ------------------------
# Main loop
# ------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Default: rerun every hour (3600s) to align with H1 bar closes
    interval_seconds = int(os.getenv("UT_SHORT_INTERVAL_SECONDS", "3600"))

    logger.info(
        "Starting UT-style SHORT screener+opener (%.3f%% allocation, interval=%ss)",
        float(os.getenv("UT_SHORT_ALLOCATION_PERCENT", "0.1")),
        interval_seconds,
    )

    while True:
        try:
            run_bot_once()
        except Exception as exc:
            logger.exception("Error in UT SHORT combined bot loop: %s", exc)
        logger.info("Sleeping for %s seconds...", interval_seconds)
        time.sleep(interval_seconds)
