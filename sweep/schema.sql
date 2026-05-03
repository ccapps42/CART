-- sweep/schema.sql — source of truth for all tables.
-- Applied by generate_configs.py on DB creation.

CREATE TABLE IF NOT EXISTS sweep_meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
-- Populated by generate_configs.py with all fixed hyperparameters.
-- Keys include: vocab_size, d_head, mla_compression_ratio,
--               n_hyper, rope_base, ffn_mult,
--               seq_len_stage1, seq_len_stage2, steps_stage1, steps_stage2,
--               tokenizer_name, data_files (json), peak_lr, warmup_steps,
--               weight_decay, grad_clip

CREATE TABLE IF NOT EXISTS configs (
    config_id    TEXT PRIMARY KEY,
    d_model      INTEGER NOT NULL,
    n_loops      INTEGER NOT NULL,
    n_prelude    INTEGER NOT NULL,
    seed         INTEGER NOT NULL DEFAULT 42,
    stage        INTEGER NOT NULL DEFAULT 1,
    status       TEXT NOT NULL DEFAULT 'pending',
    hardware     TEXT NOT NULL,    -- '3050' or '3090'
    model_type   TEXT NOT NULL DEFAULT 'cart',  -- 'cart' or 'dense'
    retry_count  INTEGER NOT NULL DEFAULT 0,
    error_msg    TEXT,
    created_at   TEXT NOT NULL,
    started_at   TEXT,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS results (
    result_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    config_id      TEXT NOT NULL REFERENCES configs(config_id),
    step           INTEGER NOT NULL,
    train_loss     REAL,
    eval_ppl_tiny  REAL,
    eval_ppl_wiki  REAL,
    eval_ppl_edu   REAL,
    peak_vram_gb   REAL,
    tokens_per_sec REAL,
    lti_spectral_radius REAL,    -- model.lti.spectral_radius() at this step
    wall_sec       REAL,
    recorded_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS train_log (
    log_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    config_id      TEXT NOT NULL REFERENCES configs(config_id),
    step           INTEGER NOT NULL,
    train_loss     REAL NOT NULL,  -- raw loss at this step (not smoothed)
    grad_norm      REAL,           -- gradient norm after clipping
    lr             REAL,           -- current learning rate
    n_tokens_seen  INTEGER,        -- cumulative tokens processed so far
    lti_spectral_radius REAL,      -- logged every 50 steps for early instability detection
    wall_sec       REAL,           -- elapsed wall clock since run start
    recorded_at    TEXT NOT NULL
);
-- train_log is the source of loss curve figures.
-- Written every 50 steps. Lightweight — no eval, no VRAM measurement.
-- results table is written only at eval checkpoints (steps 500, 1000, 1500, 2000, 2500, 3000)

CREATE INDEX IF NOT EXISTS idx_results_config   ON results(config_id, step);
CREATE INDEX IF NOT EXISTS idx_train_log_config ON train_log(config_id, step);
CREATE INDEX IF NOT EXISTS idx_configs_status   ON configs(status, stage, hardware);
