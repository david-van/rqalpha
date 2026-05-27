-- ============================================================
-- ETF T+0 日内高低开策略 — MySQL 数据库设计
-- ============================================================
-- 使用方式：
--   mysql -u root -p < db_schema.sql
-- ============================================================

CREATE DATABASE IF NOT EXISTS rqalpha_etf
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE rqalpha_etf;

-- -----------------------------------------------------------
-- 1. ETF 日线行情表（原始数据）
-- -----------------------------------------------------------
DROP TABLE IF EXISTS etf_daily_bars;
CREATE TABLE etf_daily_bars (
    id              BIGINT          AUTO_INCREMENT PRIMARY KEY,
    order_book_id   VARCHAR(20)     NOT NULL COMMENT '标的代码，如 513030.XSHG',
    trade_date      DATE            NOT NULL COMMENT '交易日期',
    open            DECIMAL(12,4)   NOT NULL DEFAULT 0 COMMENT '开盘价',
    high            DECIMAL(12,4)   NOT NULL DEFAULT 0 COMMENT '最高价',
    low             DECIMAL(12,4)   NOT NULL DEFAULT 0 COMMENT '最低价',
    close           DECIMAL(12,4)   NOT NULL DEFAULT 0 COMMENT '收盘价',
    prev_close      DECIMAL(12,4)   NOT NULL DEFAULT 0 COMMENT '前收盘价',
    volume          DECIMAL(18,2)   NOT NULL DEFAULT 0 COMMENT '成交量（股）',
    total_turnover  DECIMAL(18,2)   NOT NULL DEFAULT 0 COMMENT '成交额',

    UNIQUE INDEX uk_symbol_date (order_book_id, trade_date),
    INDEX idx_trade_date (trade_date),
    INDEX idx_order_book_id (order_book_id)
) ENGINE=InnoDB COMMENT 'ETF 日线行情原始数据（来自 RQAlpha bundle）';


-- -----------------------------------------------------------
-- 2. T+0 ETF 标的列表（由 etf_data_to_mysql.py 自动从 instruments.pk 填充）
-- -----------------------------------------------------------
DROP TABLE IF EXISTS etf_universe;
CREATE TABLE etf_universe (
    order_book_id   VARCHAR(20)     PRIMARY KEY COMMENT '标的代码',
    name            VARCHAR(200)    DEFAULT '' COMMENT '中文简称',
    market          VARCHAR(10)     DEFAULT '' COMMENT '市场 SH/SZ',
    t0_type         VARCHAR(20)     NOT NULL DEFAULT 'cross_border' COMMENT '类别: cross_border | gold | bond | money_market | commodity | other',
    underlying      VARCHAR(200)    DEFAULT '' COMMENT '底层标的代码',
    listed_date     DATE            DEFAULT NULL COMMENT '上市日期',
    round_lot       INT             DEFAULT 100 COMMENT '最小交易单位',
    active          TINYINT(1)      DEFAULT 1 COMMENT '是否活跃'
) ENGINE=InnoDB COMMENT 'T+0 ETF 标的池（自动发现）';


-- -----------------------------------------------------------
-- 3. 视图：高低开策略分析字段
--    在日线数据基础上预计算 gap / return 等派生字段
--    net_return 为扣费后收益率（默认单边万0.5，ETF 免印花税）
-- -----------------------------------------------------------
DROP VIEW IF EXISTS v_etf_gap_analysis;
CREATE VIEW v_etf_gap_analysis AS
SELECT
    b.order_book_id,
    b.trade_date,
    YEAR(b.trade_date)              AS trade_year,
    b.open,
    b.close,
    b.prev_close,
    b.volume,
    b.total_turnover,
    -- 高低开价差
    b.open - b.prev_close           AS gap,
    (b.open - b.prev_close) / NULLIF(b.prev_close, 0) AS gap_pct,
    -- 开盘买入-收盘卖出 毛收益率
    b.close / NULLIF(b.open, 0) - 1 AS gross_return,
    -- 开盘买入-收盘卖出 净收益率（扣除双边手续费）
    (b.close * (1 - 0.00005)
      / NULLIF(b.open * (1 + 0.00005), 0)
      - 1)                          AS net_return,
    -- 价差收益（单位价格）
    b.close - b.open                AS pnl_price
FROM etf_daily_bars b
WHERE b.open > 0 AND b.close > 0 AND b.prev_close > 0;


-- -----------------------------------------------------------
-- 4. 存储过程：全市场阈值扫描
--    输入起始/结束日期和步长，输出各阈值下各 ETF 的表现汇总
-- -----------------------------------------------------------
DROP PROCEDURE IF EXISTS sp_gap_threshold_scan;
DELIMITER //
CREATE PROCEDURE sp_gap_threshold_scan(
    IN p_start_date     DATE,
    IN p_end_date       DATE,
    IN p_min_gap        DECIMAL(10,4),
    IN p_max_gap        DECIMAL(10,4),
    IN p_step           DECIMAL(10,4)
)
BEGIN
    DECLARE v_total_steps INT DEFAULT 0;
    DECLARE v_idx         INT DEFAULT 0;
    DECLARE v_threshold   DECIMAL(10,4);

    -- 用整数计数避免 decimal 循环精度问题
    SET v_total_steps = FLOOR((p_max_gap - p_min_gap) / p_step) + 1;

    DROP TEMPORARY TABLE IF EXISTS tmp_threshold_results;
    CREATE TEMPORARY TABLE tmp_threshold_results (
        threshold       DECIMAL(10,4)   NOT NULL COMMENT '高低开阈值',
        order_book_id   VARCHAR(20)     NOT NULL COMMENT '标的代码',
        trade_count     INT             NOT NULL DEFAULT 0 COMMENT '触发次数',
        win_rate        DECIMAL(8,6)    DEFAULT 0 COMMENT '胜率',
        avg_return      DECIMAL(12,8)   DEFAULT 0 COMMENT '平均单笔收益率',
        total_return    DECIMAL(12,8)   DEFAULT 0 COMMENT '累计收益率',
        max_drawdown    DECIMAL(8,6)    DEFAULT 0 COMMENT '最大回撤',
        avg_pnl_price   DECIMAL(12,6)   DEFAULT 0 COMMENT '平均价差',
        total_pnl_price DECIMAL(12,4)   DEFAULT 0 COMMENT '累计价差',
        profit_loss     DECIMAL(12,4)   DEFAULT NULL COMMENT '盈亏比'
    );

    WHILE v_idx < v_total_steps DO
        SET v_threshold = p_min_gap + v_idx * p_step;

        INSERT INTO tmp_threshold_results
        SELECT
            v_threshold,
            v.order_book_id,
            COUNT(*)                                AS trade_count,
            AVG(CASE WHEN net_return > 0 THEN 1 ELSE 0 END) AS win_rate,
            AVG(net_return)                         AS avg_return,
            EXP(SUM(LN(GREATEST(1 + COALESCE(net_return, 0), 0.0001)))) - 1 AS total_return,
            0,   -- max_drawdown 需要逐日净值计算，此处略
            AVG(pnl_price)                          AS avg_pnl_price,
            SUM(pnl_price)                          AS total_pnl_price,
            CASE WHEN SUM(CASE WHEN net_return < 0 THEN -net_return ELSE 0 END) > 0
                 THEN SUM(CASE WHEN net_return > 0 THEN net_return ELSE 0 END)
                    / SUM(CASE WHEN net_return < 0 THEN -net_return ELSE 0 END)
            END                                     AS profit_loss
        FROM v_etf_gap_analysis v
        WHERE v.trade_date BETWEEN p_start_date AND p_end_date
          AND v.gap >= v_threshold
        GROUP BY v.order_book_id
        HAVING trade_count > 0;

        SET v_idx = v_idx + 1;
    END WHILE;

    SELECT * FROM tmp_threshold_results
    ORDER BY total_return DESC, threshold;
END //
DELIMITER ;


-- -----------------------------------------------------------
-- 5. 存储过程：按高低开分档 + 年份汇总
-- -----------------------------------------------------------
DROP PROCEDURE IF EXISTS sp_gap_bucket_summary;
DELIMITER //
CREATE PROCEDURE sp_gap_bucket_summary(
    IN p_start_date     DATE,
    IN p_end_date       DATE,
    IN p_step           DECIMAL(10,4)  -- 分档步长，如 0.001
)
BEGIN
    SELECT
        v.order_book_id,
        v.trade_year,
        ROUND(ROUND(v.gap / p_step, 0) * p_step, 4) AS gap_bucket,
        AVG(v.gap)                          AS avg_gap,
        AVG(v.gap_pct)                      AS avg_gap_pct,
        COUNT(*)                            AS trade_count,
        AVG(CASE WHEN v.net_return > 0 THEN 1 ELSE 0 END) AS win_rate,
        AVG(v.net_return)                   AS avg_return,
        EXP(SUM(LN(1 + COALESCE(v.net_return, 0)))) - 1    AS total_return,
        AVG(v.pnl_price)                    AS avg_pnl_price,
        SUM(v.pnl_price)                    AS total_pnl_price
    FROM v_etf_gap_analysis v
    WHERE v.trade_date BETWEEN p_start_date AND p_end_date
    GROUP BY v.order_book_id, v.trade_year, gap_bucket
    HAVING trade_count > 0
    ORDER BY v.order_book_id, v.trade_year, gap_bucket;
END //
DELIMITER ;


-- ============================================================
-- 标的列表由 etf_data_to_mysql.py 在首次运行时自动填充
-- 来源: RQAlpha 的 instruments.pk (market_tplus=0 的 ETF)
-- ============================================================
