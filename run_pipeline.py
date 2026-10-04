"""Ejecutar desde la raíz: python run_pipeline.py"""
from pathlib import Path
import argparse
import json
from src.pipeline import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TFM Favorita: preparación, backtesting y forecast")
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    config = json.loads((root / args.config).read_text())
    run(root, config)
