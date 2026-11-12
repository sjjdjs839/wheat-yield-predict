"""数据库表结构定义"""

FEATURE_COLS = [
    "spi_12", "precip_anom", "hot_days", "cold_days", "gdd",
    "season_tmax", "season_tmin", "season_precip", "season_wind", "season_sunshine",
    "season_precip_norm", "season_tmax_34", "season_precip_34", "gdd_34", "cold_days_34",
    "season_tmax_56", "season_precip_56", "gdd_56", "hot_days_56", "hot_dry_days",
]

DDL = [
    """
    CREATE TABLE IF NOT EXISTS counties (
        id INT AUTO_INCREMENT PRIMARY KEY,
        name VARCHAR(64) NOT NULL UNIQUE,
        city VARCHAR(64) NOT NULL,
        region_type VARCHAR(8) NOT NULL,
        KEY idx_city (city)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS weather_daily (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        county_id INT NOT NULL,
        obs_date DATE NOT NULL,
        t2m_max DOUBLE, t2m_min DOUBLE, t2m DOUBLE, precip DOUBLE,
        wind DOUBLE, radiation DOUBLE, dew DOUBLE, rh DOUBLE,
        UNIQUE KEY uk_county_date (county_id, obs_date),
        CONSTRAINT fk_weather_county FOREIGN KEY (county_id) REFERENCES counties(id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    f"""
    CREATE TABLE IF NOT EXISTS feature_table (
        id INT AUTO_INCREMENT PRIMARY KEY,
        county_id INT NOT NULL,
        year SMALLINT NOT NULL,
        {', '.join(f'{c} DOUBLE' for c in FEATURE_COLS)},
        UNIQUE KEY uk_county_year (county_id, year),
        CONSTRAINT fk_feat_county FOREIGN KEY (county_id) REFERENCES counties(id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS yield_labels (
        id INT AUTO_INCREMENT PRIMARY KEY,
        county_id INT NOT NULL,
        year SMALLINT NOT NULL,
        wheat_yield DOUBLE NOT NULL,
        source VARCHAR(16) NOT NULL DEFAULT 'demo',
        UNIQUE KEY uk_yield (county_id, year),
        CONSTRAINT fk_yield_county FOREIGN KEY (county_id) REFERENCES counties(id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
]
