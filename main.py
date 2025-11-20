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
            (h -
