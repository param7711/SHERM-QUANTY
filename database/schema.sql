-- EDGE LIBRARY (v2-forex)
CREATE TABLE IF NOT EXISTS edge_library (
    edge_id                      TEXT PRIMARY KEY,
    pair                         TEXT NOT NULL,
    -- MAJOR / CROSS. Distinguishes single-symbol edges from the two-leg
    -- relative-value edge (SEED-007), which trades a spread rather than a
    -- pair. Replaces v4.1's instrument_class, now that there is one asset
    -- class rather than several.
    pair_type                    TEXT NOT NULL DEFAULT 'MAJOR',
    trigger_feature              TEXT NOT NULL,
    trigger_condition            TEXT NOT NULL,
    direction                    TEXT NOT NULL,
    holding_period               INTEGER NOT NULL,
    win_rate                     REAL,
    avg_return_net               REAL,
    avg_return_gross             REAL,
    avg_return_pips              REAL,
    sample_size                  INTEGER,
    p_value                      REAL,
    p_value_corrected            REAL,
    oos_win_rate                 REAL,
    regime                       TEXT NOT NULL,
    vix_bucket                   TEXT,
    live_hit_rate                REAL,
    decay_flag                   INTEGER DEFAULT 0,
    decay_cause                  TEXT,
    last_active                  TEXT,
    valid_until                  TEXT,
    status                       TEXT DEFAULT 'ACTIVE',
    created_at                   TEXT NOT NULL,
    -- Feature formulas moved (28 -> 22, options/futures fields dropped,
    -- carry added) so this bumps; the regime engine itself is untouched by
    -- the pivot, so its version does not.
    feature_set_version          TEXT DEFAULT 'v2-forex',
    regime_model_version         TEXT DEFAULT 'v1',
    capacity_ceiling_lots        REAL,
    edge_provenance              TEXT NOT NULL,
    seed_source_citation         TEXT,
    regime_at_discovery          TEXT,
    -- Sign of carry_differential when the edge was seeded/discovered. A
    -- sign flip since is what CARRY_REGIME_FLIP (decay signal 4) detects.
    carry_direction_at_discovery INTEGER,
    trigger_frequency_baseline   REAL,
    trigger_frequency_current    REAL,
    frequency_decay_flag         INTEGER DEFAULT 0,
    win_rate_watch_flag          INTEGER DEFAULT 0,
    posterior_win_rate           REAL,
    live_wins_in_regime          INTEGER DEFAULT 0,
    live_losses_in_regime        INTEGER DEFAULT 0,
    resurrection_priority        TEXT,
    resurrection_attempts        INTEGER DEFAULT 0,
    mechanism_story              TEXT,
    signal_strength               TEXT,
    -- Written by populate_edges.py at seed time; revalidate_edges.py must
    -- clear it once real-data priors replace the research estimates. See
    -- Section 6.5 of the build sequence.
    needs_revalidation            INTEGER DEFAULT 1
);

-- HYPOTHESIS GRAVEYARD
CREATE TABLE IF NOT EXISTS hypothesis_graveyard (
    hypothesis_id          TEXT PRIMARY KEY,
    pair                    TEXT,
    trigger_feature        TEXT,
    trigger_condition      TEXT,
    regime                 TEXT,
    vix_bucket             TEXT,
    holding_period         INTEGER,
    failure_reason         TEXT,
    p_value_achieved       REAL,
    win_rate_achieved      REAL,
    sample_size            INTEGER,
    feature_set_version    TEXT,
    regime_model_version   TEXT,
    needs_retest           INTEGER DEFAULT 0,
    tested_at              TEXT
);

-- SHADOW LEDGER
CREATE TABLE IF NOT EXISTS shadow_ledger (
    signal_id                  TEXT PRIMARY KEY,
    edge_id                    TEXT,
    pair                       TEXT,
    -- Direction is its own column. It was previously inferred from
    -- ai2_decision, which holds APPROVED/REJECTED and never 'SHORT', so
    -- every shadow outcome was scored as if the signal were long.
    direction                  TEXT,
    signal_date                TEXT,
    trigger_condition          TEXT,
    regime_at_signal           TEXT,
    hmm_confidence_at_signal   REAL,
    ai2_decision               TEXT,
    ai2_rejection_reason       TEXT,
    shadow_outcome             TEXT,
    shadow_return_pct          REAL,
    actual_return_pct          REAL,
    holding_period             INTEGER,
    exit_date                  TEXT,
    exit_reason                TEXT,
    feature_values_json        TEXT,
    created_at                 TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_shadow_pending
    ON shadow_ledger (signal_date, shadow_outcome);

-- EXECUTION QUALITY LOG
CREATE TABLE IF NOT EXISTS execution_quality_log (
    trade_id                        TEXT PRIMARY KEY,
    edge_id                         TEXT,
    pair                            TEXT,
    lot_size                        REAL,
    direction                       TEXT,
    signal_price                    REAL,
    fill_price                      REAL,
    slippage_pips                   REAL,
    spread_at_entry_pips            REAL,
    -- Overnight financing charged while the position was held. Populated
    -- from the MT bridge; a real cost that the shadow ledger's spread-only
    -- adjustment does not capture.
    swap_charged                    REAL,
    session_at_entry                TEXT,
    fill_status                     TEXT,
    fill_pct                        REAL,
    cost_budgeted_bps               REAL,
    cost_realised_bps               REAL,
    signal_date                     TEXT,
    entry_date                      TEXT,
    exit_date                       TEXT,
    holding_period                  INTEGER,
    net_return_pct                  REAL,
    gross_return_pct                REAL,
    regime_at_entry                 TEXT,
    regime_at_exit                  TEXT,
    created_at                      TEXT NOT NULL
);

-- RL REPLAY BUFFER
CREATE TABLE IF NOT EXISTS rl_replay_buffer (
    transition_id              TEXT PRIMARY KEY,
    edge_id                    TEXT,
    pair                       TEXT,
    holding_period             INTEGER,
    regime_at_trade            TEXT,
    hmm_confidence_at_trade    REAL,
    regime_model_version       TEXT,
    state_features_json        TEXT,
    action                     TEXT,
    reward                     REAL,
    trade_date                 TEXT,
    created_at                 TEXT NOT NULL
);

-- SYSTEM VERSION REGISTRY
CREATE TABLE IF NOT EXISTS version_registry (
    version_id            TEXT PRIMARY KEY,
    feature_set_version   TEXT,
    regime_model_version  TEXT,
    activated_at          TEXT,
    notes                 TEXT
);

INSERT OR IGNORE INTO version_registry VALUES (
    'v2_forex_init', 'v2-forex', 'v1', datetime('now'),
    'Forex-only pivot: 8-symbol universe (7 majors + XAUUSD) via MetaTrader. '
    || 'Options/futures fields dropped; regime engine (v1) unchanged by the pivot.'
);
