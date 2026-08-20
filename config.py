import os
from dotenv import load_dotenv
load_dotenv()

# Broker — MetaTrader via MQL5 Expert Advisor bridge.
# The official MetaTrader5 Python package is Windows-only, so the Python
# agents talk to an EA running inside the terminal over a socket instead.
MT_BRIDGE_HOST     = os.getenv('MT_BRIDGE_HOST', '127.0.0.1')
MT_BRIDGE_PORT     = int(os.getenv('MT_BRIDGE_PORT', '9000'))
MT_BRIDGE_TIMEOUT  = 30          # seconds before an unfilled order is cancelled
MT_ACCOUNT_LOGIN   = os.getenv('MT_ACCOUNT_LOGIN')
MT_ACCOUNT_SERVER  = os.getenv('MT_ACCOUNT_SERVER')

# Capital
TOTAL_CAPITAL = 1_350_000  # Rs 13.5 lakhs

# Instrument universe (MVP) — 7 forex majors + gold.
# XAUUSD is a direct MetaTrader symbol, not a synthetic construction: the
# v4.1 two-leg build (MCX Gold + short USD/INR) existed only to cancel the
# rupee term, and there is no rupee in an XAUUSD quote.
FX_PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF',
            'AUDUSD', 'USDCAD', 'NZDUSD', 'XAUUSD']

# Legs per symbol, used for exposure-concentration checks.
# XAU counts as a leg so gold competes for the USD exposure budget.
PAIR_LEGS = {
    'EURUSD': ('EUR', 'USD'),
    'GBPUSD': ('GBP', 'USD'),
    'USDJPY': ('USD', 'JPY'),
    'USDCHF': ('USD', 'CHF'),
    'AUDUSD': ('AUD', 'USD'),
    'USDCAD': ('USD', 'CAD'),
    'NZDUSD': ('NZD', 'USD'),
    'XAUUSD': ('XAU', 'USD'),
}

# JPY pairs quote to 3 decimals, gold to 2, the rest to 5. Drives pip math.
PIP_SIZE = {
    'EURUSD': 0.0001, 'GBPUSD': 0.0001, 'USDCHF': 0.0001,
    'AUDUSD': 0.0001, 'USDCAD': 0.0001, 'NZDUSD': 0.0001,
    'USDJPY': 0.01,
    'XAUUSD': 0.01,
}

# Units per standard lot. Gold is 100 troy oz, FX is 100k currency units —
# sizing math must branch on this, not assume 100k.
CONTRACT_SIZE = {
    'EURUSD': 100_000, 'GBPUSD': 100_000, 'USDJPY': 100_000,
    'USDCHF': 100_000, 'AUDUSD': 100_000, 'USDCAD': 100_000,
    'NZDUSD': 100_000,
    'XAUUSD': 100,
}
MIN_LOT = 0.01     # micro lot

# --- Hedging -------------------------------------------------------------
# HEDGING mode is architecturally required, not a preference. Per-edge P&L
# attribution drives decay detection, Bayesian win rates, the shadow ledger
# and RL rewards. Under NETTING the broker collapses all positions on a
# symbol into one, so a mean-reversion short and a momentum long on EURUSD
# become a single blended position and neither edge can be measured.
# Verify the funded account is opened in hedging mode before going live.
MT_HEDGING_MODE = True

# Different edges may hold opposing positions on the same symbol; the RL
# layer learns which combinations pay. The same edge still cannot double up
# (enforced by the duplicate-tuple check in risk_governor).
ALLOW_OPPOSING_POSITIONS = True

# Operational schedule (IST) — FX session checkpoints.
# PLACEHOLDER: pending calibration against broker server timezone and
# per-pair liquidity profile (see Appendix A).
SCHEDULE_TIMES        = ['07:00', '13:00', '18:30']
FRIDAY_PRECLOSE_CHECK = '21:00'

# Live scanner (continuous mode). Forex trades ~24/5: Sunday 22:00 UTC
# through Friday 22:00 UTC.
LIVE_SCAN_INTERVAL_SECS = 300
FX_WEEK_OPEN_UTC        = ('SUN', '22:00')
FX_WEEK_CLOSE_UTC       = ('FRI', '22:00')

# Risk parameters
MAX_POSITION_PCT      = 0.07
MAX_POSITIONS_TOTAL   = 8
MAX_MARGIN_PCT        = 0.60
MIN_MARGIN_BUFFER_PCT = 0.25
MAX_PER_CURRENCY_EXPOSURE = 2   # positions sharing a base or quote currency

# Stop loss — adverse move on the pair, by holding period
FX_STOP = {2: 0.010, 5: 0.015, 10: 0.020, 15: 0.025}

# Gold runs roughly 2-3x the daily range of a major, so the shared FX_STOP
# bands would stop it out on noise. PLACEHOLDER multiplier — settle this by
# revalidation against real XAUUSD history, not by assumption.
SYMBOL_STOP_MULTIPLIER = {'XAUUSD': 2.0}

# Trailing stop multipliers on vol_5d
TRAIL_MULTIPLIER = {5: 1.5, 10: 2.0, 15: 2.5}

# Execution filters
MAX_SPREAD_PCT      = 0.0003   # ~2-3 pips on a EUR/USD-class pair
MAX_SLIPPAGE_PIPS   = {2: 15, 5: 20, 10: 25, 15: 30}
NEWS_BLACKOUT_MINS  = 15       # no entries within N mins of high-impact news

# Holding periods (days)
HOLDING_PERIODS     = [2, 5, 10, 15]
MAX_SLOTS_BY_PERIOD = {2: 3, 5: 2, 10: 2, 15: 1}
MAX_LOTS_BY_PERIOD  = {2: 1, 5: 2, 10: 2, 15: 3}

# Regime engine — unchanged. Nifty/India VIX HMM serves as a macro
# risk-on/risk-off filter gating FX signals.
CONFIDENCE_THRESHOLD_H  = 0.70
CONFIDENCE_THRESHOLD_L  = 0.50
HMM_PROB_DROP_THRESHOLD = 0.20
VIX_SPIKE_THRESHOLD     = 0.25

# Carry — policy rate per currency, used for carry_differential.
# PLACEHOLDER: refresh from a rates feed; static values decay fast.
#
# XAU has no policy rate. Holding gold forgoes USD interest and pays a small
# lease/storage cost, so its effective carry is roughly -(USD rate) less the
# lease rate. Modelled here as a near-zero own-rate; the differential then
# falls out as approximately -USD_rate, which is the right sign and rough
# magnitude. PLACEHOLDER — calibrate against observed XAUUSD swap charges.
POLICY_RATES = {
    'USD': 0.0450, 'EUR': 0.0240, 'GBP': 0.0400, 'JPY': 0.0050,
    'CHF': 0.0025, 'AUD': 0.0360, 'CAD': 0.0275, 'NZD': 0.0325,
    'XAU': 0.0000,
}
CARRY_FLIP_RETEST_THRESHOLD = 0.0025  # differential swing that flags a retest

# --- Cross-pair relative value (SEED-007) --------------------------------
# Long one pair against another is mean reversion on the spread, not a risk
# hedge — the same statistical-arbitrage mechanism as SEED-001 applied to a
# ratio instead of a single series (Avellaneda & Lee 2010).
#
# Note many of these have a directly tradeable cross on MetaTrader (EURGBP,
# AUDNZD). The two-leg form is kept because it permits a beta-adjusted hedge
# ratio, which a direct cross fixes at 1:1 — at the cost of two spreads and
# two swaps. Compare both during revalidation.
CROSS_PAIRS = [
    ('EURUSD', 'GBPUSD'),   # ~ EURGBP
    ('AUDUSD', 'NZDUSD'),   # ~ AUDNZD
    ('USDCHF', 'USDJPY'),   # safe-haven relative value
]
CROSS_SPREAD_Z_ENTRY = 2.0    # |z| on the spread that opens a position
CROSS_SPREAD_Z_EXIT  = 0.5    # |z| that closes it
CROSS_BETA_WINDOW    = 60     # days of history for the hedge-ratio regression

# --- Funded / prop-firm account limits -----------------------------------
# Breaching a firm limit terminates the account, so internal halts sit well
# inside them rather than at them. Defaults follow the common industry
# template; replace with the real numbers once a firm is chosen.
PROP_MAX_DAILY_LOSS_PCT = 0.05
PROP_MAX_DRAWDOWN_PCT   = 0.10
PROP_SAFETY_MARGIN      = 0.65   # halt at 65% of the firm's limit
PROP_TRAILING_DRAWDOWN  = False  # True if drawdown trails the high-water mark

# Derived internal halts — these are what the Risk Governor enforces.
DAILY_LOSS_HALT_PCT = PROP_MAX_DAILY_LOSS_PCT * PROP_SAFETY_MARGIN   # 3.25%
DRAWDOWN_HALT_PCT   = PROP_MAX_DRAWDOWN_PCT   * PROP_SAFETY_MARGIN   # 6.5%

# RL parameters
BANDIT_MIN_TRADES       = 200
META_LABELER_MIN_TRADES = 150
META_LABELER_THRESHOLD  = 0.58

# Database
DB_PATH        = 'database/sherm_quanty.db'
KAIROS_DB_PATH = 'logs/kairos_log.db'
REGIME_PARQUET = 'data/processed/regime_history.parquet'

# VortexMomentum correlation monitoring (soft alert only)
VM_CORRELATION_ALERT_THRESHOLD = 0.40
VM_CORRELATION_WINDOW          = 30

# Calibration flags (False = using placeholder, True = calibrated)
CALIBRATED = {
    'hmm_confidence_threshold': False,
    'hmm_prob_drop_threshold':  False,
    'vix_spike_threshold':      False,
    'session_checkpoint_times': False,
    'trail_multiplier':         False,
    'spread_slippage_limits':   False,
    'carry_flip_threshold':     False,
    'xauusd_stop_multiplier':   False,
    'xauusd_carry_convention':  False,
    'cross_spread_z_bands':     False,
    'prop_firm_limits':         False,
}
