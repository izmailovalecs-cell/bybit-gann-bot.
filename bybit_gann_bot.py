from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")
import os, time, math, traceback
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from pybit.unified_trading import HTTP

# ===================== ENV =====================
API_KEY = os.getenv("BYBIT_API_KEY", "")
API_SECRET = os.getenv("BYBIT_API_SECRET", "")

SYMBOLS = [s.strip().upper() for s in os.getenv(
    "SYMBOLS",
    "BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,ADAUSDT,TONUSDT"
).split(",") if s.strip()]

LEVERAGE = int(os.getenv("LEVERAGE", "5"))
RISK_PCT = float(os.getenv("RISK_PER_TRADE", "0.05"))  # 5% от свободного USDT

TF_ENTRY   = int(os.getenv("TF_ENTRY", "15"))    # 15m
TF_FILTER  = int(os.getenv("TF_FILTER", "60"))   # 1h
TF_FILTER2 = int(os.getenv("TF_FILTER2","240"))  # 4h

STOP_ATR_MULT   = float(os.getenv("STOP_ATR_MULT", "2.0"))   # SL = ATR*mult
MIN_ATR_PCT     = float(os.getenv("MIN_ATR_PCT", "0.2"))     # мин. волатильность (%)
VOL_LOOKBACK    = int(os.getenv("VOL_LOOKBACK", "20"))

COOLDOWN_SEC        = int(os.getenv("COOLDOWN_SEC", "120"))   # пауза между входами
TRADING_HOURS_UTC   = os.getenv("TRADING_HOURS_UTC", "7-22")  # "start-end" UTC
DAILY_STOP_PCT      = float(os.getenv("DAILY_STOP_PCT", "3"))/100.0   # -3% стоп-день
BE_TRIGGER_PCT      = float(os.getenv("BE_TRIGGER_PCT", "0.005"))     # 0.5% безубыток
TRAIL_ACTIVATE_PCT  = float(os.getenv("TRAIL_ACTIVATE_PCT","0.004"))  # 0.4% активация трейла
TRAIL_DISTANCE_PCT  = float(os.getenv("TRAIL_DISTANCE_PCT","0.003"))  # трейл 0.3%

# ===================== API =====================
session = HTTP(api_key=API_KEY, api_secret=API_SECRET)

# ================== INDICATORS =================
def ema(arr, n):
    s = pd.Series(arr, dtype="float64")
    return s.ewm(span=n, adjust=False).mean().values

def rsi(arr, n=14):
    s = pd.Series(arr, dtype="float64")
    delta = s.diff()
    up = delta.clip(lower=0).rolling(n).mean()
    dn = (-delta.clip(upper=0)).rolling(n).mean()
    rs = up / (dn.replace(0, 1e-12))
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def macd(arr, fast=12, slow=26, signal=9):
    fast_ema = ema(arr, fast)
    slow_ema = ema(arr, slow)
    line = fast_ema - slow_ema
    sig  = pd.Series(line).ewm(span=signal, adjust=False).mean().values
    hist = line - sig
    return line, sig, hist

def atr(high, low, close, n=14):
    h = pd.Series(high, dtype="float64")
    l = pd.Series(low, dtype="float64")
    c = pd.Series(close, dtype="float64")
    prev_close = c.shift(1)
    tr = pd.concat([(h-l).abs(),
                    (h-prev_close).abs(),
                    (l-prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean().values

def heikin_ashi_ohlc(o,h,l,c):
    o,h,l,c = map(pd.Series, (o,h,l,c))
    ha_close = (o + h + l + c) / 4
    ha_open = pd.Series(index=o.index, dtype="float64")
    ha_open.iloc[0] = o.iloc[0]
    for i in range(1,len(o)):
        ha_open.iloc[i] = (ha_open.iloc[i-1] + ha_close.iloc[i-1]) / 2
    ha_high = pd.concat([h, ha_open, ha_close], axis=1).max(axis=1)
    ha_low  = pd.concat([l, ha_open, ha_close], axis=1).min(axis=1)
    return ha_open.values, ha_high.values, ha_low.values, ha_close.values

# ================== DATA FETCH =================
def get_klines(symbol, interval, limit=400):
    resp = session.get_kline(category="linear", symbol=symbol, interval=str(interval), limit=limit)
    data = resp["result"]["list"][::-1]
    o = np.array([float(x[1]) for x in data], dtype="float64")
    h = np.array([float(x[2]) for x in data], dtype="float64")
    l = np.array([float(x[3]) for x in data], dtype="float64")
    c = np.array([float(x[4]) for x in data], dtype="float64")
    v = np.array([float(x[5]) for x in data], dtype="float64")
    t = np.array([int(x[0])   for x in data], dtype="int64")
    return t,o,h,l,c,v

def get_last_price(symbol):
    r = session.get_tickers(category="linear", symbol=symbol)
    return float(r["result"]["list"][0]["lastPrice"])

def get_free_usdt():
    # Если у тебя не UTA — поменяй accountType на "CONTRACT"
    r = session.get_wallet_balance(accountType="UNIFIED", coin="USDT")
    lst = r["result"]["list"]
    if not lst: return 0.0
    return float(lst[0]["totalAvailableBalance"])

def set_leverage(symbol, lev):
    try:
        session.set_leverage(category="linear", symbol=symbol,
                             buyLeverage=str(lev), sellLeverage=str(lev))
    except Exception:
        pass

# ================== POSITIONS ==================
def fetch_positions_both(symbol):
    r = session.get_positions(category="linear", symbol=symbol)
    lst = r["result"]["list"]
    buy = next((p for p in lst if p["side"] == "Buy"  and float(p["size"])>0), None)
    sell= next((p for p in lst if p["side"] == "Sell" and float(p["size"])>0), None)
    return buy, sell

def close_side_market(symbol, side):  # side: "Buy"/"Sell"
    buy, sell = fetch_positions_both(symbol)
    p = buy if side=="Buy" else sell
    if not p: return
    qty = p["size"]
    close_side = "Sell" if side=="Buy" else "Buy"
    session.place_order(category="linear", symbol=symbol,
                        side=close_side, orderType="Market", qty=qty, reduceOnly=True)

def close_all_market(symbol):
    buy, sell = fetch_positions_both(symbol)
    if buy:  close_side_market(symbol, "Buy")
    if sell: close_side_market(symbol, "Sell")

def open_market_with_sl(symbol, side, qty, sl_price):
    session.place_order(category="linear", symbol=symbol,
                        side=side, orderType="Market", qty=qty, reduceOnly=False)
    try:
        session.set_trading_stop(category="linear", symbol=symbol,
                                 stopLoss=str(round(sl_price, 4)), tpSlMode="Full")
    except Exception:
        pass

# =================== GANN DEEP =================
def pivots_hl(h, l, left=3, right=3):
    N = len(h)
    piv_hi, piv_lo = [], []
    for i in range(left, N-right):
        if h[i] == max(h[i-left:i+right+1]): piv_hi.append(i)
        if l[i] == min(l[i-left:i+right+1]): piv_lo.append(i)
    return piv_hi, piv_lo

def gann_direction_from_last_pivot(t, o, h, l, c, base_tf_minutes):
    atr_arr = atr(h, l, c, n=14)
    scale = np.nanmean(atr_arr[-20:])
    if not np.isfinite(scale) or scale <= 0:
        return "neutral"

    piv_hi, piv_lo = pivots_hl(h, l, left=3, right=3)
    if not piv_hi and not piv_lo:
        return "neutral"

    last_hi = piv_hi[-1] if piv_hi else -10**9
    last_lo = piv_lo[-1] if piv_lo else -10**9
    use_low = last_lo > last_hi
    pivot_idx = last_lo if use_low else last_hi
    pivot_px  = l[pivot_idx] if use_low else h[pivot_idx]
    bars_from = (len(c) - 1) - pivot_idx
    if bars_from < 1:
        return "neutral"

    slope_1x1 = scale * 1.0

    if use_low:
        ray_1x1 = pivot_px + slope_1x1 * bars_from
        return "bull" if c[-1] > ray_1x1 else "neutral"
    else:
        ray_1x1 = pivot_px - slope_1x1 * bars_from
        return "bear" if c[-1] < ray_1x1 else "neutral"

# ============== EXTRA FILTERS / GUARDS =========
def volume_ok(vol, lookback=20):
    if len(vol) < lookback+1: return False
    ma = pd.Series(vol).rolling(lookback).mean().values
    return vol[-1] > ma[-1]

def volume_cum_ok(v, n=3, lookback=20):
    v = np.asarray(v)
    if len(v) < max(n,lookback)+2: return False
    recent = v[-n:].sum()
    avg = pd.Series(v).rolling(lookback).mean().iloc[-1] * n
    return recent > avg

def atr_pct_ok(h,l,c, min_pct=0.2):
    a = atr(h,l,c,14)
    pct = (a[-1] / c[-1]) * 100.0
    return pct >= min_pct

def ema_trend_ok(c, fast=50, slow=200, side="Buy"):
    e1, e2 = ema(c,fast), ema(c,slow)
    if side=="Buy":
        return c[-1] > e1[-1] > e2[-1]
    else:
        return c[-1] < e1[-1] < e2[-1]

def macd_ok(c, side="Buy"):
    line, sig, hist = macd(c)
    if side=="Buy":  return line[-1] > sig[-1] and hist[-1] > 0
    else:            return line[-1] < sig[-1] and hist[-1] < 0

def rsi_ok(c, side="Buy"):
    r = rsi(c,14)
    if side=="Buy":  return r[-1] < 70 and r[-1] > 40
    else:            return r[-1] > 30 and r[-1] < 60

def obv_slope_ok(c, v, side="Buy", lookback=20, span=5):
    c = np.asarray(c); v = np.asarray(v)
    delta = np.sign(np.diff(c, prepend=c[0]))
    obv = np.cumsum(delta * v)
    slope = obv[-1] - obv[-span]
    return slope > 0 if side=="Buy" else slope < 0

def rsi_divergence_block(o,h,l,c, side="Buy", lookback=60):
    r = rsi(c,14)
    piv_hi, piv_lo = pivots_hl(h, l, left=3, right=3)
    if side=="Buy":
        H = [i for i in piv_hi if i >= len(c)-lookback]
        if len(H) < 2: return False
        h1, h2 = H[-2], H[-1]
        price_higher_high = c[h2] > c[h1]
        rsi_lower_high   = r[h2] < r[h1]
        return price_higher_high and rsi_lower_high
    else:
        L = [i for i in piv_lo if i >= len(c)-lookback]
        if len(L) < 2: return False
        l1, l2 = L[-2], L[-1]
        price_lower_low  = c[l2] < c[l1]
        rsi_higher_low   = r[l2] > r[l1]
        return price_lower_low and rsi_higher_low

def trading_hours_ok():
    try:
        a,b = TRADING_HOURS_UTC.split("-")
        a,b = int(a), int(b)
    except:
        a,b = 0,24
    hour = datetime.now(timezone.utc).hour
    if a <= b: return a <= hour < b
    return hour >= a or hour < b

# ================== GLOBAL STATE =================
STATE = {sym: {"last_trade_ts": 0, "day_date": None, "day_start": None, "disabled": False}
         for sym in SYMBOLS}

def daily_guard(symbol):
    st = STATE[symbol]
    today = datetime.now(timezone.utc).date()
    free = get_free_usdt()
    if st["day_date"] != today:
        st["day_date"] = today
        st["day_start"] = free
        st["disabled"] = False
    if st["day_start"] and free < st["day_start"]*(1-DAILY_STOP_PCT):
        st["disabled"] = True
    return not st["disabled"]

# ================== TREND / ENTRY =================
def higher_tf_trend(symbol):
    _,o1,h1,l1,c1,v1 = get_klines(symbol, TF_FILTER, 400)    # 1h
    _,o4,h4,l4,c4,v4 = get_klines(symbol, TF_FILTER2, 300)   # 4h

    ema200 = ema(c1, 200)
    ema_filter_bull = c1[-1] > ema200[-1]
    ema_filter_bear = c1[-1] < ema200[-1]

    g1 = gann_direction_from_last_pivot(None, o1, h1, l1, c1, TF_FILTER)

    hao, hah, hal, hac = heikin_ashi_ohlc(o4,h4,l4,c4)
    ha_bull = hac[-1] > hao[-1]
    ha_bear = hac[-1] < hao[-1]

    bull = ema_filter_bull and (g1=="bull") and ha_bull
    bear = ema_filter_bear and (g1=="bear") and ha_bear
    if bull and not bear: return "bull"
    if bear and not bull: return "bear"
    return "neutral"

def allow_entry_on_entry_tf(symbol, side):
    _, o,h,l,c,v = get_klines(symbol, TF_ENTRY, 400)

    if not atr_pct_ok(h,l,c, MIN_ATR_PCT): return False
    if not volume_ok(v, VOL_LOOKBACK): return False
    if not volume_cum_ok(v, n=3, lookback=VOL_LOOKBACK): return False

    g = gann_direction_from_last_pivot(None, o,h,l,c, TF_ENTRY)
    if side == "Buy" and g != "bull": return False
    if side == "Sell" and g != "bear": return False

    if rsi_divergence_block(o,h,l,c, side=side): return False

    if not rsi_ok(c, side=side):       return False
    if not ema_trend_ok(c, side=side): return False
    if not macd_ok(c, side=side):      return False
    if not obv_slope_ok(c, v, side=side, lookback=20, span=5): return False
    return True

# ================== SIZING / SL / TRAIL =============
def compute_qty(symbol, price):
    free = get_free_usdt()
    notional = max(5.0, free * RISK_PCT)
    qty = round(notional / price, 6)
    return qty

def compute_sl(symbol, side):
    _,o,h,l,c,v = get_klines(symbol, TF_ENTRY, 100)
    a = atr(h,l,c,14)[-1]
    if side=="Buy":
        return c[-1] - a * STOP_ATR_MULT
    else:
        return c[-1] + a * STOP_ATR_MULT

def adjust_sl_trailing(symbol):
    buy, sell = fetch_positions_both(symbol)
    pos = buy or sell
    if not pos: return
    side = pos["side"]
    entry = float(pos["avgPrice"])
    last  = get_last_price(symbol)
    pnl_pct = (last/entry - 1.0) if side=="Buy" else (entry/last - 1.0)

    new_sl = None
    # безубыток
    if pnl_pct >= BE_TRIGGER_PCT:
        new_sl = entry

    # трейлинг
    if pnl_pct >= TRAIL_ACTIVATE_PCT:
        trail = last*(1 - TRAIL_DISTANCE_PCT) if side=="Buy" else last*(1 + TRAIL_DISTANCE_PCT)
        new_sl = max(new_sl, trail) if side=="Buy" else min(new_sl, trail)

    if new_sl:
        try:
            session.set_trading_stop(category="linear", symbol=symbol,
                                     stopLoss=str(round(new_sl, 4)), tpSlMode="Full")
        except Exception:
            pass

# ===================== CORE ======================
def maybe_flip(symbol):
    st = STATE[symbol]
    if not daily_guard(symbol): return
    if not trading_hours_ok():  return
    if time.time() - st["last_trade_ts"] < COOLDOWN_SEC: return

    ht = higher_tf_trend(symbol)
    if ht == "neutral": return
    side = "Buy" if ht=="bull" else "Sell"

    if not allow_entry_on_entry_tf(symbol, side): return

    set_leverage(symbol, LEVERAGE)

    buy, sell = fetch_positions_both(symbol)
    if side=="Buy" and sell:
        close_side_market(symbol, "Sell")
        time.sleep(0.8)
    if side=="Sell" and buy:
        close_side_market(symbol, "Buy")
        time.sleep(0.8)

    buy, sell = fetch_positions_both(symbol)
    if (side=="Buy" and buy) or (side=="Sell" and sell):
        return

    price = get_last_price(symbol)
    qty = compute_qty(symbol, price)
    sl = compute_sl(symbol, side)
    open_market_with_sl(symbol, side, qty, sl)
    st["last_trade_ts"] = time.time()
    print(f"[{datetime.now(timezone.utc)}] {symbol}: OPEN {side} qty={qty} @ ~{price:.2f} SL={sl:.2f}")

def main_loop():
    print("Bot started. Symbols:", SYMBOLS)
    while True:
        try:
            for sym in SYMBOLS:
                try:
                    maybe_flip(sym)
                    adjust_sl_trailing(sym)
                except Exception as e:
                    print(sym, "error:", e)
            time.sleep(15)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print("loop error:", e)
            traceback.print_exc()
            time.sleep(5)

if __name__ == "__main__":
    main_loop()
