"""Batch prediction script using akshare data for multiple stocks.

Runs Kronos predictions for a list of stock symbols and saves results
to an output directory.
"""

import os
import sys
import argparse
import logging
from datetime import datetime

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kronos import Kronos

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


DEFAULT_STOCKS = [
    "000001",  # Ping An Bank
    "600036",  # China Merchants Bank
    "000858",  # Wuliangye
    "600519",  # Kweichow Moutai
    "300750",  # CATL
]


def fetch_stock_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Fetch OHLCV data via akshare for a given symbol and date range."""
    try:
        import akshare as ak
    except ImportError:
        raise ImportError("akshare is required: pip install akshare")

    df = ak.stock_zh_a_hist(
        symbol=symbol,
        period="daily",
        start_date=start.replace("-", ""),
        end_date=end.replace("-", ""),
        adjust="qfq",
    )
    df = df.rename(
        columns={
            "日期": "date",
            "开盘": "open",
            "最高": "high",
            "最低": "low",
            "收盘": "close",
            "成交量": "volume",
        }
    )
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df[["date", "open", "high", "low", "close", "volume"]]


def run_prediction(symbol: str, df: pd.DataFrame, predict_days: int, output_dir: str):
    """Run Kronos prediction for a single stock and save the result plot."""
    if len(df) < 60:
        logger.warning("Skipping %s: insufficient data (%d rows)", symbol, len(df))
        return

    close_prices = df["close"].values.astype(float)

    model = Kronos()
    try:
        prediction = model.predict(close_prices, steps=predict_days)
    except Exception as exc:
        logger.error("Prediction failed for %s: %s", symbol, exc)
        return

    # Persist numeric results
    result_df = pd.DataFrame(
        {
            "step": range(1, predict_days + 1),
            "predicted_close": prediction,
        }
    )
    csv_path = os.path.join(output_dir, f"{symbol}_prediction.csv")
    result_df.to_csv(csv_path, index=False)
    logger.info("Saved prediction for %s -> %s", symbol, csv_path)


def main():
    parser = argparse.ArgumentParser(description="Batch Kronos predictions via akshare")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_STOCKS)
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default=datetime.today().strftime("%Y-%m-%d"))
    parser.add_argument("--predict-days", type=int, default=30)
    parser.add_argument("--output-dir", default="output/batch_predictions")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(
        "Running batch predictions for %d stocks (%s to %s), %d days ahead",
        len(args.symbols),
        args.start,
        args.end,
        args.predict_days,
    )

    for symbol in args.symbols:
        logger.info("Processing %s ...", symbol)
        try:
            df = fetch_stock_data(symbol, args.start, args.end)
            run_prediction(symbol, df, args.predict_days, args.output_dir)
        except Exception as exc:
            logger.error("Error processing %s: %s", symbol, exc)

    logger.info("Batch prediction complete. Results in: %s", args.output_dir)


if __name__ == "__main__":
    main()
