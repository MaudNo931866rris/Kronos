"""Kronos - Time series prediction library for stock market analysis.

Fork of shiyu-coder/Kronos with extended support for Chinese markets,
akshare data integration, and batch prediction utilities.
"""

__version__ = "0.2.0"
__author__ = "Kronos Contributors"

from kronos.model import Kronos
from kronos.predictor import StockPredictor

__all__ = ["Kronos", "StockPredictor"]
