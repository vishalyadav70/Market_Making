"""
╔══════════════════════════════════════════════════════════════╗
║   HFT LIVE DASHBOARD v2 — 12 Coins + 14 Charts             ║
║   FIXED: Trades now happen, OB depth, all graphs live       ║
╚══════════════════════════════════════════════════════════════╝
pip install ccxt pandas numpy matplotlib scipy
"""

import threading, time, warnings
warnings.filterwarnings('ignore')
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List
from collections import deque
import numpy as np
import pandas as pd
import ccxt

import matplotlib
for _b in ['Qt5Agg','TkAgg','WxAgg','Agg']:
    try:
        matplotlib.use(_b)
        import matplotlib.pyplot as plt
        plt.figure(); plt.close(); break
    except: continue
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
print(f"✅ Backend: {matplotlib.get_backend()}")

# ══════════════════════════════════════════════
# 12 COINS
# ══════════════════════════════════════════════
COINS = [
    'BTC/USDT','ETH/USDT','BNB/USDT','SOL/USDT',
    'XRP/USDT','DOGE/USDT','ADA/USDT','AVAX/USDT',
    'MATIC/USDT','DOT/USDT','LINK/USDT','UNI/USDT',
]
SHORT = [c.split('/')[0] for c in COINS]

STARTING_CAPITAL = 10000.0
STOP_LOSS_PCT    = 0.015
TAKE_PROFIT_PCT  = 0.035
POSITION_SIZE    = 0.15
MAX_POSITIONS    = 6
TICK_INTERVAL    = 3
TOTAL_TICKS      = 600

# ── RELAXED THRESHOLDS (trades honge) ─────────
RSI_BUY   = 48    # was 40 → 48
RSI_SELL  = 52    # was 60 → 52
Z_THRESH  = 0.25  # was 0.6 → 0.25
VOL_THRESH= 1.1   # was 1.3 → 1.1
MIN_SCORE = 0.8   # was 1.2 → 0.8

# ── Colors ──────────────────────────────────
BG='#0d1117'; BG2='#161b22'
GRN='#3fb950'; RED='#f85149'
BLU='#58a6ff'; YLW='#ffa657'
PRP='#d2a8ff'; GRY='#8b949e'
CYN='#39d353'; ORG='#e3b341'
GRD=dict(color='#21262d', linewidth=0.4)

COIN_COLORS = [
    '#58a6ff','#3fb950','#ffa657','#d2a8ff',
    '#f85149','#39d353','#79c0ff','#e3b341',
    '#a5f3fc','#c084fc','#fb923c','#4ade80',
]

# ══════════════════════════════════════════════
# SIGNALS (continuous scoring)
# ══════════════════════════════════════════════
def calc_rsi(prices, period=9):
    d=prices.diff().dropna()
    g=d.clip(lower=0).ewm(span=period).mean().iloc[-1]
    l=(-d.clip(upper=0)).ewm(span=period).mean().iloc[-1]
    return 100-(100/(1+g/l)) if l!=0 else 50

def calc_zscore(prices, window=20):
    m=prices.rolling(window).mean().iloc[-1]
    s=prices.rolling(window).std().iloc[-1]
    return (prices.iloc[-1]-m)/s if s!=0 else 0

def calc_vol_ratio(volumes, window=20):
    avg=volumes.rolling(window).mean().iloc[-1]
    return volumes.iloc[-1]/avg if avg!=0 else 1

def calc_rv(prices, window=20):
    r=prices.pct_change().dropna()
    v=r.rolling(window).std().iloc[-1]
    return float(v*np.sqrt(1440*365)*100) if v and not np.isnan(v) else 0.0

def calc_bb_pct(prices, period=20, n=2):
    ma=prices.rolling(period).mean().iloc[-1]
    sd=prices.rolling(period).std().iloc[-1]
    up=ma+n*sd; lo=ma-n*sd
    return (prices.iloc[-1]-lo)/(up-lo) if (up-lo)!=0 else 0.5

def compute_signals(df):
    p=df['close']; v=df['volume']

    rsi   = calc_rsi(p)
    z     = calc_zscore(p)
    vr    = calc_vol_ratio(v)
    bb    = calc_bb_pct(p)
    rv    = calc_rv(p)

    # Continuous scores — map to [-1.2, 1.2]
    rsi_s = float(np.clip((50-rsi)/50*1.2, -1.2, 1.2))
    z_s   = float(np.clip(-z/max(Z_THRESH,0.01)*1.0, -1.2, 1.2))
    bb_s  = float(np.clip((0.5-bb)*2.0, -1.0, 1.0))
    vr_s  = 0.4 if vr > VOL_THRESH else 0.0

    score  = rsi_s + z_s + bb_s + vr_s
    signal = 1 if score >= MIN_SCORE else (-1 if score <= -MIN_SCORE else 0)

    return dict(
        signal=signal, score=round(float(score),3),
        price=float(p.iloc[-1]),
        rsi=round(float(rsi),1), zscore=round(float(z),3),
        vol_ratio=round(float(vr),2), bb=round(float(bb),3),
        rv=round(float(rv),2),
        rsi_s=rsi_s, z_s=z_s,
    )

# ══════════════════════════════════════════════
# PORTFOLIO
# ══════════════════════════════════════════════
@dataclass
class Position:
    coin:str; side:str; entry:float
    size_usd:float; qty:float; sl:float; tp:float; peak:float=0.0

@dataclass
class Trade:
    coin:str; pnl:float; pnl_pct:float; reason:str

class Portfolio:
    def __init__(self, cap):
        self.cash       = cap
        self.positions: Dict[str,Position] = {}
        self.trades:    List[Trade]        = []
        self.eq_h    = deque([cap],  maxlen=600)
        self.roi_h   = deque([0.0],  maxlen=600)
        self.pnl_h   = deque([0.0],  maxlen=600)
        self.spr_h   = deque([0.0],  maxlen=600)
        self.coin_pnl= {s:0.0 for s in SHORT}
        self.spread_income = 0.0

    def open(self, coin, signal, price):
        if coin in self.positions: return None
        if len(self.positions) >= MAX_POSITIONS: return None
        usd  = self.cash * POSITION_SIZE
        if usd < 5: return None
        side = 'long' if signal == 1 else 'short'
        qty  = usd / price
        sl   = price*(1-STOP_LOSS_PCT)   if side=='long' else price*(1+STOP_LOSS_PCT)
        tp   = price*(1+TAKE_PROFIT_PCT) if side=='long' else price*(1-TAKE_PROFIT_PCT)
        self.cash -= usd
        pos  = Position(coin=coin, side=side, entry=price,
                        size_usd=usd, qty=qty, sl=sl, tp=tp, peak=price)
        self.positions[coin] = pos
        return pos

    def check_exits(self, prices):
        closed = []
        for coin, pos in list(self.positions.items()):
            p = prices.get(coin)
            if p is None: continue

            # Update trailing stop
            if pos.side=='long' and p > pos.peak:
                pos.peak = p
                pos.sl   = max(pos.sl, p*(1-0.010))
            elif pos.side=='short' and (p < pos.peak or pos.peak==pos.entry):
                pos.peak = p
                pos.sl   = min(pos.sl, p*(1+0.010))

            reason = None
            if pos.side=='long':
                if p <= pos.sl:   reason='SL'
                elif p >= pos.tp: reason='TP ✨'
            else:
                if p >= pos.sl:   reason='SL'
                elif p <= pos.tp: reason='TP ✨'

            if reason:
                pnl = (p-pos.entry)*pos.qty if pos.side=='long' else (pos.entry-p)*pos.qty
                self.cash += pos.size_usd + pnl
                t = Trade(coin=coin, pnl=pnl,
                          pnl_pct=pnl/pos.size_usd*100, reason=reason)
                self.trades.append(t); closed.append(t)
                self.coin_pnl[coin] = self.coin_pnl.get(coin,0) + pnl
                del self.positions[coin]
        return closed

    def equity(self, prices):
        eq = self.cash
        for coin, pos in self.positions.items():
            p = prices.get(coin, pos.entry)
            eq += pos.size_usd + ((p-pos.entry)*pos.qty if pos.side=='long'
                                  else (pos.entry-p)*pos.qty)
        return eq

    def snap(self, prices):
        eq  = self.equity(prices)
        roi = (eq-STARTING_CAPITAL)/STARTING_CAPITAL*100
        pnl = eq-STARTING_CAPITAL
        self.eq_h.append(eq); self.roi_h.append(roi); self.pnl_h.append(pnl)
        self.spr_h.append(self.spread_income)
        return eq, roi, pnl

    def metrics(self):
        if not self.trades: return {}
        pnls  = [t.pnl for t in self.trades]
        wins  = [p for p in pnls if p>0]; losses=[p for p in pnls if p<=0]
        eq    = pd.Series(list(self.eq_h)); ret=eq.pct_change().dropna()
        sharpe= (ret.mean()/ret.std()*np.sqrt(86400)) if ret.std()>0 else 0
        mdd   = ((eq-eq.cummax())/eq.cummax()).min()*100
        return dict(
            roi=(list(self.eq_h)[-1]-STARTING_CAPITAL)/STARTING_CAPITAL*100,
            pnl=list(self.pnl_h)[-1], sharpe=sharpe, mdd=mdd,
            n=len(self.trades),
            wr=len(wins)/len(self.trades)*100,
            pf=abs(sum(wins)/sum(losses)) if losses and sum(losses)!=0 else 999,
            avg_w=np.mean(wins) if wins else 0,
            avg_l=np.mean(losses) if losses else 0)

# ══════════════════════════════════════════════
# STATE
# ══════════════════════════════════════════════
class State:
    def __init__(self):
        self.lock       = threading.Lock()
        self.portfolio  = Portfolio(STARTING_CAPITAL)
        self.prices     = {}
        self.signals    = {}
        self.ob_bids    = []
        self.ob_asks    = []
        self.ob_coin    = 'BTC'
        self.tick       = 0
        self.log        = deque(maxlen=20)
        self.running    = True
        self.fetch_ms   = 0
        # Per-coin history
        self.rsi_h  = {s: deque([50.0], maxlen=120) for s in SHORT}
        self.z_h    = {s: deque([0.0],  maxlen=120) for s in SHORT}
        self.vr_h   = {s: deque([1.0],  maxlen=120) for s in SHORT}
        self.rv_h   = {s: deque([0.0],  maxlen=120) for s in SHORT}
        self.sc_h   = {s: deque([0.0],  maxlen=120) for s in SHORT}
        self.px_h   = {s: deque([1.0],  maxlen=120) for s in SHORT}

    def add_log(self, msg):
        self.log.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

# ══════════════════════════════════════════════
# TRADING THREAD
# ══════════════════════════════════════════════
def trading_thread(state: State):
    ex = ccxt.binance({'enableRateLimit': True})
    print("✅ Binance connected — 12 coins\n")

    # Startup diagnosis
    print("─"*60)
    print("  SIGNAL DIAGNOSIS (startup)")
    print("─"*60)

    for tick in range(TOTAL_TICKS):
        if not state.running: break
        t0 = time.time()
        prices={}; signals={}

        for coin in COINS:
            short = coin.split('/')[0]
            try:
                data = ex.fetch_ohlcv(coin, '1m', limit=50)
                df   = pd.DataFrame(data, columns=['ts','open','high','low','close','volume'])
                prices[short] = float(df['close'].iloc[-1])
                sig  = compute_signals(df)
                signals[short] = sig

                # Startup: show signal diagnosis
                if tick == 0:
                    print(f"  {short:<8} | score:{sig['score']:+.3f} "
                          f"| RSI:{sig['rsi']:.0f} "
                          f"| Z:{sig['zscore']:+.2f} "
                          f"| {'→ TRADE!' if abs(sig['score'])>=MIN_SCORE else 'waiting...'}")
            except Exception as e:
                pass

        if tick == 0:
            print("─"*60 + "\n")

        if not prices:
            time.sleep(TICK_INTERVAL); continue

        # Fetch OB for BTC only (avoids rate limits)
        try:
            ob = ex.fetch_order_book('BTC/USDT', limit=10)
            with state.lock:
                state.ob_bids = ob.get('bids', [])[:10]
                state.ob_asks = ob.get('asks', [])[:10]
                state.ob_coin = 'BTC'
        except: pass

        fms = int((time.time()-t0)*1000)

        with state.lock:
            state.prices  = prices
            state.signals = signals
            state.tick    = tick+1
            state.fetch_ms= fms

            # Update histories
            for short, sig in signals.items():
                state.rsi_h[short].append(sig.get('rsi',50))
                state.z_h[short].append(sig.get('zscore',0))
                state.vr_h[short].append(sig.get('vol_ratio',1))
                state.rv_h[short].append(sig.get('rv',0))
                state.sc_h[short].append(sig.get('score',0))
                state.px_h[short].append(sig.get('price',1))

            # Exits
            for t in state.portfolio.check_exits(prices):
                icon = '✅' if t.pnl>0 else '❌'
                msg  = f"{icon} {t.coin} {t.reason} ${t.pnl:+.4f} ({t.pnl_pct:+.1f}%)"
                state.add_log(msg); print(f"  {msg}")

            # Entries
            for short, sig in signals.items():
                sv = sig.get('signal', 0)
                if sv != 0:
                    pos = state.portfolio.open(short, sv, sig['price'])
                    if pos:
                        spread = sig['price'] * 0.0005
                        state.portfolio.spread_income += spread
                        s  = '🟢 LONG' if pos.side=='long' else '🔴 SHORT'
                        msg= (f"▶ {s} {short} @ ${sig['price']:,.2f} "
                              f"score:{sig['score']:+.2f} "
                              f"RSI:{sig['rsi']:.0f} Z:{sig['zscore']:+.2f}")
                        state.add_log(msg); print(f"  {msg}")

            eq, roi, pnl = state.portfolio.snap(prices)

        # Active positions status
        with state.lock:
            pos_str = " | ".join([
                f"{c}:{p.side[0].upper()}@${p.entry:,.0f}"
                for c,p in state.portfolio.positions.items()
            ]) or "no positions"

        print(f"[{datetime.now().strftime('%H:%M:%S')}] "
              f"T#{tick+1:04d} ${eq:,.2f} ROI:{roi:+.4f}% | "
              f"Pos:{len(state.portfolio.positions)} "
              f"Trades:{len(state.portfolio.trades)} | "
              f"{pos_str} | {fms}ms")

        elapsed = time.time()-t0
        time.sleep(max(0, TICK_INTERVAL-elapsed))

    state.running = False
    print("\n✅ Session complete!")

# ══════════════════════════════════════════════
# CHART HELPER
# ══════════════════════════════════════════════
def sax(ax, title='', fs=8):
    ax.set_facecolor(BG2)
    ax.tick_params(colors=GRY, labelsize=6)
    ax.grid(**GRD)
    for sp in ax.spines.values(): sp.set_color('#21262d')
    if title: ax.set_title(title, color='#c9d1d9', fontsize=fs, pad=4)

def annotate_last(ax, data, color, fmt_str='{:.2f}', xoff=5, yoff=4):
    if len(data) < 2: return
    try:
        ax.annotate(fmt_str.format(data[-1]),
                    xy=(len(data)-1, data[-1]),
                    color=color, fontsize=6,
                    xytext=(xoff, yoff), textcoords='offset points')
    except: pass

# ══════════════════════════════════════════════
# DRAW DASHBOARD
# ══════════════════════════════════════════════
def draw_dashboard(state, axes):
    with state.lock:
        eq_h    = list(state.portfolio.eq_h)
        roi_h   = list(state.portfolio.roi_h)
        pnl_h   = list(state.portfolio.pnl_h)
        spr_h   = list(state.portfolio.spr_h)
        trades  = list(state.portfolio.trades)
        prices  = dict(state.prices)
        sigs    = dict(state.signals)
        ob_bids = list(state.ob_bids)
        ob_asks = list(state.ob_asks)
        ob_coin = state.ob_coin
        log     = list(state.log)
        tick    = state.tick
        run     = state.running
        fms     = state.fetch_ms
        m       = state.portfolio.metrics()
        rsi_h   = {s: list(v) for s,v in state.rsi_h.items()}
        z_h     = {s: list(v) for s,v in state.z_h.items()}
        vr_h    = {s: list(v) for s,v in state.vr_h.items()}
        rv_h    = {s: list(v) for s,v in state.rv_h.items()}
        sc_h    = {s: list(v) for s,v in state.sc_h.items()}
        cpnl    = dict(state.portfolio.coin_pnl)
        spread  = state.portfolio.spread_income
        n_pos   = len(state.portfolio.positions)
        open_pos= dict(state.portfolio.positions)

    (ax_eq, ax_roi, ax_stats, ax_log,
     ax_pnl, ax_trade, ax_ob, ax_spr,
     ax_heat, ax_rsi, ax_z, ax_vr,
     ax_rv, ax_cpnl, ax_sc, ax_pos) = axes

    xs = range(len(eq_h))

    # ── 1. EQUITY CURVE ───────────────────────
    ax_eq.cla(); sax(ax_eq, f'Portfolio Equity  |  T#{tick}  |  {datetime.now().strftime("%H:%M:%S")}  |  {fms}ms  |  {n_pos}/{MAX_POSITIONS} positions', fs=8)
    if len(eq_h)>1:
        ax_eq.plot(xs, eq_h, color=BLU, lw=1.8, zorder=3)
        ax_eq.axhline(STARTING_CAPITAL, color=GRY, lw=0.7, ls='--')
        ax_eq.fill_between(xs, eq_h, STARTING_CAPITAL,
            where=[e>=STARTING_CAPITAL for e in eq_h], color=GRN, alpha=0.13)
        ax_eq.fill_between(xs, eq_h, STARTING_CAPITAL,
            where=[e<STARTING_CAPITAL  for e in eq_h], color=RED, alpha=0.13)
        c = GRN if eq_h[-1]>=STARTING_CAPITAL else RED
        annotate_last(ax_eq, eq_h, c, '${:,.2f}')

    # ── 2. ROI % ──────────────────────────────
    ax_roi.cla(); sax(ax_roi, 'ROI %')
    if len(roi_h)>1:
        ax_roi.plot(xs, roi_h, color=YLW, lw=1.5)
        ax_roi.fill_between(xs, roi_h, 0,
            where=[r>=0 for r in roi_h], color=GRN, alpha=0.15)
        ax_roi.fill_between(xs, roi_h, 0,
            where=[r<0  for r in roi_h], color=RED, alpha=0.15)
        ax_roi.axhline(0, color=GRY, lw=0.7, ls='--')
        c = GRN if roi_h[-1]>=0 else RED
        annotate_last(ax_roi, roi_h, c, '{:+.4f}%')

    # ── 3. STATS PANEL ────────────────────────
    ax_stats.cla(); sax(ax_stats, 'Live Stats')
    ax_stats.axis('off')
    roi_n = roi_h[-1] if roi_h else 0
    pnl_n = pnl_h[-1] if pnl_h else 0
    status_col = GRN if run else YLW
    rows = [
        ('STATUS',   '🔴 LIVE' if run else '✅ DONE',  status_col, True),
        ('ROI',      f'{roi_n:+.4f}%',     GRN if roi_n>=0 else RED, True),
        ('PnL',      f'${pnl_n:+,.4f}',    GRN if pnl_n>=0 else RED, True),
        ('Spread',   f'${spread:.5f}',      CYN,  False),
        ('Sharpe',   f"{m.get('sharpe',0):.3f}",  GRN if m.get('sharpe',0)>1 else YLW, False),
        ('MaxDD',    f"{m.get('mdd',0):.2f}%",    RED,  False),
        ('Win Rate', f"{m.get('wr',0):.0f}%",     GRN if m.get('wr',0)>50 else YLW, False),
        ('PF',       f"{m.get('pf',0):.2f}x",     GRN if m.get('pf',0)>1.5 else YLW, False),
        ('Trades',   f"{m.get('n',0)}",            PRP,  False),
        ('Avg Win',  f"${m.get('avg_w',0):.4f}",  GRN,  False),
        ('Avg Loss', f"${m.get('avg_l',0):.4f}",  RED,  False),
        ('Coins',    f"{len(prices)}/12",           BLU,  False),
    ]
    for i,(k,v,c,bold) in enumerate(rows):
        ax_stats.text(0.02, 0.97-i*0.078, k+':',
                      color=GRY, fontsize=6.5, transform=ax_stats.transAxes,
                      fontfamily='monospace')
        ax_stats.text(0.45, 0.97-i*0.078, v,
                      color=c, fontsize=6.5,
                      fontweight='bold' if bold else 'normal',
                      transform=ax_stats.transAxes, fontfamily='monospace')

    # ── 4. ACTIVITY LOG ───────────────────────
    ax_log.cla(); sax(ax_log, 'Activity Log')
    ax_log.axis('off')
    for i, msg in enumerate(list(log)[-13:]):
        col = (GRN if '✅' in msg or '🟢' in msg
               else RED if '❌' in msg or '🔴' in msg
               else GRY)
        ax_log.text(0.02, 0.97-i*0.073, msg[:68],
                    color=col, fontsize=5.8,
                    transform=ax_log.transAxes, fontfamily='monospace')

    # ── 5. CUM PnL ────────────────────────────
    ax_pnl.cla(); sax(ax_pnl, 'Cumulative PnL ($)')
    if len(pnl_h)>1:
        ax_pnl.plot(xs, pnl_h, color=PRP, lw=1.5)
        ax_pnl.fill_between(xs, pnl_h, 0,
            where=[p>=0 for p in pnl_h], color=GRN, alpha=0.13)
        ax_pnl.fill_between(xs, pnl_h, 0,
            where=[p<0  for p in pnl_h], color=RED, alpha=0.13)
        ax_pnl.axhline(0, color=GRY, lw=0.7, ls='--')

    # ── 6. PER-TRADE PnL ──────────────────────
    ax_trade.cla(); sax(ax_trade, 'Per Trade PnL ($)')
    if trades:
        tp    = [t.pnl for t in trades]
        tc    = [GRN if p>0 else RED for p in tp]
        ax_trade.bar(range(len(tp)), tp, color=tc, width=0.7, zorder=3)
        ax_trade.axhline(0, color=GRY, lw=0.7)
        # Cumulative overlay
        ax2 = ax_trade.twinx()
        ax2.plot(range(len(tp)), np.cumsum(tp), color=YLW, lw=1.3, alpha=0.7)
        ax2.tick_params(colors=GRY, labelsize=5)
        ax2.set_facecolor(BG2)
        for sp in ax2.spines.values(): sp.set_color('#21262d')
        ax2.set_ylabel('Cum PnL', color=YLW, fontsize=5)
    else:
        ax_trade.text(0.5, 0.5, 'Trades incoming...',
                      color=GRY, ha='center', va='center',
                      transform=ax_trade.transAxes, fontsize=9)

    # ── 7. ORDER BOOK ─────────────────────────
    ax_ob.cla(); sax(ax_ob, f'Order Book — {ob_coin}/USDT (Live Depth)')
    ax_ob.axis('off')
    if ob_bids and ob_asks:
        max_vol = max([b[1] for b in ob_bids]+[a[1] for a in ob_asks]+[0.001])
        # Header
        ax_ob.text(0.02, 0.97, 'QTY',      color=GRY, fontsize=6, transform=ax_ob.transAxes)
        ax_ob.text(0.32, 0.97, 'BID',      color=GRN, fontsize=6, fontweight='bold', transform=ax_ob.transAxes)
        ax_ob.text(0.56, 0.97, 'ASK',      color=RED, fontsize=6, fontweight='bold', transform=ax_ob.transAxes)
        ax_ob.text(0.82, 0.97, 'QTY',      color=GRY, fontsize=6, transform=ax_ob.transAxes)

        # 8 levels each side
        for i in range(min(8, len(ob_bids), len(ob_asks))):
            y = 0.87 - i*0.108
            bid = ob_bids[i]; ask = ob_asks[i]
            bw  = bid[1]/max_vol*0.28; aw = ask[1]/max_vol*0.28
            # Bid bar
            ax_ob.barh(y, bw, height=0.09, left=0.30-bw, color=GRN, alpha=0.25)
            ax_ob.text(0.02, y, f"{bid[1]:.3f}", color=GRY,   fontsize=5.5, va='center', fontfamily='monospace')
            ax_ob.text(0.30, y, f"${bid[0]:,.1f}", color=GRN, fontsize=5.5, va='center', ha='right', fontfamily='monospace')
            # Ask bar
            ax_ob.text(0.56, y, f"${ask[0]:,.1f}", color=RED, fontsize=5.5, va='center', fontfamily='monospace')
            ax_ob.barh(y, aw, height=0.09, left=0.85, color=RED, alpha=0.25)
            ax_ob.text(0.88, y, f"{ask[1]:.3f}", color=GRY,   fontsize=5.5, va='center', fontfamily='monospace')

        # Spread
        if ob_bids and ob_asks:
            spread_val = ob_asks[0][0] - ob_bids[0][0]
            mid        = (ob_asks[0][0] + ob_bids[0][0]) / 2
            ax_ob.text(0.5, 0.03,
                       f"Mid ${mid:,.1f}  |  Spread ${spread_val:.2f}  ({spread_val/mid*100:.4f}%)",
                       color=YLW, ha='center', fontsize=6, transform=ax_ob.transAxes,
                       fontfamily='monospace')

    # ── 8. SPREAD INCOME ──────────────────────
    ax_spr.cla(); sax(ax_spr, 'Spread Income ($) Captured')
    if len(spr_h)>1:
        ax_spr.plot(xs, spr_h, color=CYN, lw=1.5)
        ax_spr.fill_between(xs, spr_h, 0, color=CYN, alpha=0.13)
        ax_spr.axhline(0, color=GRY, lw=0.6)
        annotate_last(ax_spr, spr_h, CYN, '${:.5f}')

    # ── 9. SIGNAL HEATMAP ─────────────────────
    ax_heat.cla(); sax(ax_heat, 'Signal Score Heatmap — All 12 Coins  (GREEN=BUY  RED=SELL  threshold=±0.8)')
    if sigs:
        cmap = LinearSegmentedColormap.from_list('rg',
               ['#f85149','#161b22','#3fb950'], N=256)
        scores = [sigs.get(s,{}).get('score',0) for s in SHORT]
        for i,(s,sc) in enumerate(zip(SHORT, scores)):
            c   = cmap((sc+3)/6)
            val = abs(sc)
            ax_heat.barh(i, val if sc>=0 else 0,   height=0.7, color=GRN, alpha=0.7)
            ax_heat.barh(i, -val if sc<0 else 0,  height=0.7, color=RED, alpha=0.7)
            sig = sigs.get(s,{})
            sv  = sig.get('signal',0)
            act = '▲ BUY' if sv==1 else ('▼ SELL' if sv==-1 else '  —  ')
            ac  = GRN if sv==1 else (RED if sv==-1 else GRY)
            p   = sig.get('price',0)
            # Coin label
            ax_heat.text(-3.8, i, f"{s:<6}", color='#c9d1d9',
                         fontsize=6.5, va='center', fontfamily='monospace')
            # Score
            ax_heat.text(0, i, f"{sc:+.2f}",
                         color='white', fontsize=6, ha='center', va='center', fontweight='bold')
            # Action
            ax_heat.text(2.0, i, act, color=ac,
                         fontsize=6.5, va='center', fontweight='bold')
            # Price
            ax_heat.text(3.2, i, f"${p:,.2f}" if p>1 else f"${p:.4f}",
                         color=GRY, fontsize=5.5, va='center')
            # RSI + Z
            rsi_v = sig.get('rsi',50); z_v = sig.get('zscore',0)
            ax_heat.text(-3.8, i-0.35,
                         f"RSI:{rsi_v:.0f}  Z:{z_v:+.2f}  VR:{sig.get('vol_ratio',1):.1f}x",
                         color=GRY, fontsize=4.8, va='center', fontfamily='monospace')

        ax_heat.axvline( MIN_SCORE, color=GRN, lw=0.8, ls='--', alpha=0.6)
        ax_heat.axvline(-MIN_SCORE, color=RED, lw=0.8, ls='--', alpha=0.6)
        ax_heat.axvline(0, color=GRY, lw=0.5)
        ax_heat.set_xlim(-4.2, 4.5)
        ax_heat.set_yticks([])
        ax_heat.set_xlabel('Signal Score', color=GRY, fontsize=7)

    # ── 10. RSI ALL COINS ─────────────────────
    ax_rsi.cla(); sax(ax_rsi, 'RSI (9) — All 12 Coins')
    for i, s in enumerate(SHORT):
        rh = rsi_h.get(s, [])
        if len(rh)>1:
            ax_rsi.plot(range(len(rh)), rh, color=COIN_COLORS[i],
                        lw=0.8, alpha=0.85, label=s)
    ax_rsi.axhline(RSI_BUY,  color=GRN, lw=0.8, ls='--', alpha=0.6)
    ax_rsi.axhline(RSI_SELL, color=RED, lw=0.8, ls='--', alpha=0.6)
    ax_rsi.axhline(50, color=GRY, lw=0.4)
    ax_rsi.set_ylim(0, 100)
    ax_rsi.fill_between(range(max(1,len(list(rsi_h.values())[0]))),
                         0, RSI_BUY, alpha=0.04, color=GRN)
    ax_rsi.fill_between(range(max(1,len(list(rsi_h.values())[0]))),
                         RSI_SELL, 100, alpha=0.04, color=RED)
    ax_rsi.legend(facecolor=BG2, labelcolor='white', fontsize=4.5,
                  loc='upper left', ncol=4, borderpad=0.2, handlelength=1)

    # ── 11. Z-SCORE ALL COINS ─────────────────
    ax_z.cla(); sax(ax_z, 'Z-Score (20) — All 12 Coins')
    for i, s in enumerate(SHORT):
        zh = z_h.get(s, [])
        if len(zh)>1:
            ax_z.plot(range(len(zh)), zh, color=COIN_COLORS[i],
                      lw=0.8, alpha=0.85, label=s)
    ax_z.axhline( Z_THRESH, color=RED, lw=0.8, ls='--', alpha=0.6)
    ax_z.axhline(-Z_THRESH, color=GRN, lw=0.8, ls='--', alpha=0.6)
    ax_z.axhline(0, color=GRY, lw=0.4)
    ax_z.fill_between(range(max(1,len(list(z_h.values())[0]))),
                       -Z_THRESH, Z_THRESH, alpha=0.04, color=YLW)
    ax_z.legend(facecolor=BG2, labelcolor='white', fontsize=4.5,
                loc='upper left', ncol=4, borderpad=0.2, handlelength=1)

    # ── 12. VOLUME RATIO ──────────────────────
    ax_vr.cla(); sax(ax_vr, 'Volume Ratio (vs 20-bar avg)')
    for i, s in enumerate(SHORT):
        vh = vr_h.get(s, [])
        if len(vh)>1:
            ax_vr.plot(range(len(vh)), vh, color=COIN_COLORS[i],
                       lw=0.8, alpha=0.75, label=s)
    ax_vr.axhline(VOL_THRESH, color=YLW, lw=0.8, ls='--', alpha=0.7)
    ax_vr.axhline(1.0, color=GRY, lw=0.4)
    ax_vr.legend(facecolor=BG2, labelcolor='white', fontsize=4.5,
                 loc='upper left', ncol=4, borderpad=0.2, handlelength=1)

    # ── 13. REALIZED VOL ──────────────────────
    ax_rv.cla(); sax(ax_rv, 'Realized Volatility % (annualized)')
    cur_rvs = [rv_h.get(s,[0])[-1] for s in SHORT]
    ax_rv.bar(range(len(SHORT)), cur_rvs, color=COIN_COLORS, width=0.7, zorder=3)
    ax_rv.set_xticks(range(len(SHORT)))
    ax_rv.set_xticklabels(SHORT, rotation=45, ha='right', fontsize=5.5)
    ax_rv.set_ylabel('RV %', color=GRY, fontsize=6)
    for i, v in enumerate(cur_rvs):
        if v>0:
            ax_rv.text(i, v+0.5, f'{v:.0f}', ha='center', fontsize=5, color='white')

    # ── 14. COIN PnL BREAKDOWN ────────────────
    ax_cpnl.cla(); sax(ax_cpnl, 'PnL by Coin ($)')
    vals   = [cpnl.get(s,0) for s in SHORT]
    colors_c = [GRN if v>0 else (RED if v<0 else GRY) for v in vals]
    ax_cpnl.bar(range(len(SHORT)), vals, color=colors_c, width=0.7, zorder=3)
    ax_cpnl.axhline(0, color=GRY, lw=0.7)
    ax_cpnl.set_xticks(range(len(SHORT)))
    ax_cpnl.set_xticklabels(SHORT, rotation=45, ha='right', fontsize=5.5)
    mx = max([abs(v) for v in vals]+[0.001])
    for i,v in enumerate(vals):
        if abs(v)>0.0001:
            ax_cpnl.text(i, v+(mx*0.12 if v>=0 else -mx*0.20),
                         f'${v:.4f}', ha='center', fontsize=4.8, color='white')

    # ── 15. SCORE HISTORY ─────────────────────
    ax_sc.cla(); sax(ax_sc, 'Signal Score History — All Coins')
    for i, s in enumerate(SHORT):
        sh = sc_h.get(s, [])
        if len(sh)>1:
            ax_sc.plot(range(len(sh)), sh, color=COIN_COLORS[i],
                       lw=0.8, alpha=0.85, label=s)
    ax_sc.axhline( MIN_SCORE, color=GRN, lw=0.8, ls='--', alpha=0.6)
    ax_sc.axhline(-MIN_SCORE, color=RED, lw=0.8, ls='--', alpha=0.6)
    ax_sc.axhline(0, color=GRY, lw=0.4)
    ax_sc.fill_between(range(max(1,len(list(sc_h.values())[0]))),
                        MIN_SCORE, 4, alpha=0.04, color=GRN)
    ax_sc.fill_between(range(max(1,len(list(sc_h.values())[0]))),
                        -MIN_SCORE, -4, alpha=0.04, color=RED)
    ax_sc.set_ylim(-4,4)
    ax_sc.legend(facecolor=BG2, labelcolor='white', fontsize=4.5,
                 loc='upper left', ncol=4, borderpad=0.2, handlelength=1)

    # ── 16. OPEN POSITIONS ────────────────────
    ax_pos.cla(); sax(ax_pos, f'Open Positions ({n_pos}/{MAX_POSITIONS})')
    ax_pos.axis('off')
    if open_pos:
        hdrs = ['Coin','Side','Entry','Current','PnL $','PnL %','SL','TP']
        cx   = [0.02,0.14,0.24,0.38,0.52,0.64,0.75,0.87]
        for j,h in enumerate(hdrs):
            ax_pos.text(cx[j], 0.93, h, color=GRY, fontsize=6,
                        fontweight='bold', transform=ax_pos.transAxes)
        for i,(coin,pos) in enumerate(open_pos.items()):
            y   = 0.82 - i*0.12
            cur = prices.get(coin, pos.entry)
            pnl = (cur-pos.entry)*pos.qty if pos.side=='long' else (pos.entry-cur)*pos.qty
            ppc = pnl/pos.size_usd*100
            pc  = GRN if pnl>=0 else RED
            sc  = GRN if pos.side=='long' else RED
            vals= [coin, pos.side.upper(), f"${pos.entry:,.2f}",
                   f"${cur:,.2f}", f"${pnl:+.4f}", f"{ppc:+.1f}%",
                   f"${pos.sl:,.2f}", f"${pos.tp:,.2f}"]
            cols= ['#f0f6fc', sc,'#c9d1d9','#c9d1d9', pc, pc,'#c9d1d9','#c9d1d9']
            for j,(v,c) in enumerate(zip(vals,cols)):
                ax_pos.text(cx[j], y, v, color=c, fontsize=6,
                            transform=ax_pos.transAxes, fontfamily='monospace')
    else:
        ax_pos.text(0.5, 0.5,
                    'No open positions\nScanning all 12 coins for signals...',
                    color=GRY, ha='center', va='center',
                    fontsize=9, transform=ax_pos.transAxes)

# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════
if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════╗")
    print("║   HFT LIVE DASHBOARD v2.0                           ║")
    print("║   12 Coins + 16 Charts + Order Book + Signals       ║")
    print(f"║   Thresholds: RSI {RSI_BUY}/{RSI_SELL} | Z±{Z_THRESH} | Score±{MIN_SCORE}          ║")
    print("║   PAPER MODE — Real Binance Prices                  ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    state = State()
    t = threading.Thread(target=trading_thread, args=(state,), daemon=True)
    t.start()

    print("⏳ Fetching data + diagnosis (5s)...")
    time.sleep(5)

    # ── Build Figure ──────────────────────────
    fig = plt.figure(figsize=(24, 15), facecolor=BG)
    fig.suptitle(
        'HFT Live Dashboard v2.0  |  12 Coins  |  16 Charts  |  Paper Trading  |  Real Binance Prices',
        fontsize=12, color='white', fontweight='bold', y=0.99)

    gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.58, wspace=0.30)

    # Row 0
    ax_eq    = fig.add_subplot(gs[0, :2])
    ax_roi   = fig.add_subplot(gs[0, 2])
    ax_stats = fig.add_subplot(gs[0, 3])

    # Row 1
    ax_pnl   = fig.add_subplot(gs[1, 0])
    ax_trade = fig.add_subplot(gs[1, 1])
    ax_ob    = fig.add_subplot(gs[1, 2])
    ax_spr   = fig.add_subplot(gs[1, 3])

    # Row 2
    ax_heat  = fig.add_subplot(gs[2, :2])
    ax_rsi   = fig.add_subplot(gs[2, 2])
    ax_z     = fig.add_subplot(gs[2, 3])

    # Row 3
    ax_vr    = fig.add_subplot(gs[3, 0])
    ax_rv    = fig.add_subplot(gs[3, 1])
    ax_cpnl  = fig.add_subplot(gs[3, 2])
    ax_sc    = fig.add_subplot(gs[3, 3])

    # Extra panels: log + positions (overlay row 0)
    ax_log   = fig.add_axes([0.755, 0.742, 0.236, 0.115], facecolor=BG2)
    ax_pos   = fig.add_axes([0.005, 0.005, 0.99, 0.095], facecolor=BG2)

    all_axes = (ax_eq, ax_roi, ax_stats, ax_log,
                ax_pnl, ax_trade, ax_ob, ax_spr,
                ax_heat, ax_rsi, ax_z, ax_vr,
                ax_rv, ax_cpnl, ax_sc, ax_pos)

    def do_animate(frame):
        draw_dashboard(state, all_axes)

    ani = animation.FuncAnimation(
        fig, do_animate,
        interval=3000, cache_frame_data=False)

    try:
        plt.show(block=True)
    except KeyboardInterrupt:
        state.running = False
    finally:
        state.running = False

    time.sleep(0.5)
    print("\n" + "═"*54)
    m = state.portfolio.metrics()
    if m:
        print(f"  ROI:{m['roi']:+.3f}%  PnL:${m['pnl']:+,.4f}")
        print(f"  Sharpe:{m['sharpe']:.3f}  WR:{m['wr']:.0f}%  Trades:{m['n']}")
    print("═"*54)
