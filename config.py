import os
from dotenv import load_dotenv
load_dotenv()

# Broker
ZERODHA_API_KEY      = os.getenv('ZERODHA_API_KEY')
ZERODHA_API_SECRET   = os.getenv('ZERODHA_API_SECRET')
ZERODHA_ACCESS_TOKEN = os.getenv('ZERODHA_ACCESS_TOKEN')

# Capital
TOTAL_CAPITAL = 1_350_000  # Rs 13.5 lakhs

# Instrument universe (MVP)
INDEX_UNDERLYINGS     = ['NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY']
COMMODITY_UNDERLYINGS = ['GOLD', 'SILVER', 'CRUDEOIL', 'COPPER']
CURRENCY_UNDERLYINGS  = ['USDINR', 'EURINR', 'GBPINR', 'JPYINR',
                          'EURUSD', 'GBPUSD', 'USDJPY']

# Options expiry preference
INDEX_OPTIONS_EXPIRY     = 'WEEKLY'
COMMODITY_OPTIONS_EXPIRY = 'MONTHLY'
CURRENCY_EXPIRY          = 'MONTHLY'
MIN_DTE_INDEX            = 2
MIN_DTE_COMMODITY        = 5

# Operational schedule (IST)
SCHEDULE_TIMES          = ['09:30', '13:00', '15:20']
COMMODITY_EVENING_CHECK = '23:00'

# Risk parameters
MAX_POSITION_PCT       = 0.07
MAX_POSITIONS_TOTAL    = 8
MAX_MARGIN_PCT         = 0.60
MIN_MARGIN_BUFFER_PCT  = 0.25

# Stop loss parameters
OPTION_STOP_PCT  = 0.35
THETA_STOP_PCT   = 0.30
FUTURES_STOP     = {2: 0.010, 5: 0.015, 10: 0.020, 15: 0.025}

# Holding periods (days)
HOLDING_PERIODS      = [2, 5, 10, 15]
MAX_SLOTS_BY_PERIOD  = {2: 3, 5: 2, 10: 2, 15: 1}

# Regime engine
CONFIDENCE_THRESHOLD_H  = 0.70
CONFIDENCE_THRESHOLD_L  = 0.50
HMM_PROB_DROP_THRESHOLD = 0.20
VIX_SPIKE_THRESHOLD     = 0.25

# RL parameters
BANDIT_MIN_TRADES         = 200
META_LABELER_MIN_TRADES   = 150
META_LABELER_THRESHOLD    = 0.58

# Database
DB_PATH          = 'database/sherm_quanty.db'
KAIROS_DB_PATH   = 'logs/kairos_log.db'
REGIME_PARQUET   = 'data/processed/regime_history.parquet'

# Synthetic options pricing
IV_PROXY_MULTIPLIER = 1.10
RISK_FREE_RATE      = 0.065

# VortexMomentum correlation monitoring (soft alert only)
VM_CORRELATION_ALERT_THRESHOLD = 0.40
VM_CORRELATION_WINDOW          = 30

# Calibration flags (False = using placeholder, True = calibrated)
CALIBRATED = {
    'hmm_confidence_threshold': False,
    'hmm_prob_drop_threshold':  False,
    'vix_spike_threshold':      False,
    'iv_proxy_multiplier':      False,
    'theta_stop_pct':           False,
    'trail_multiplier_option':  False,
}
