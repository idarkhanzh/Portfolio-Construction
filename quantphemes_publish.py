"""
Push the latest portfolio_engine.py output to a Quantphemes mimic-trading strategy.

Deliberately kept separate from portfolio_engine.py: a failed API call must never
take down the weekly allocation run, and the CSV in outputs/ stays the source of
truth either way.

Usage
-----
    set QP_API_KEY=qc_live_...
    set QP_PORTFOLIO_ID=d4ddf27b-9871-40a7-8063-5ff69ebd6fbb

    python quantphemes_publish.py --list             # discover strategy ids
    python quantphemes_publish.py --dry-run          # print payload, send nothing
    python quantphemes_publish.py                    # PATCH latest targets
    python quantphemes_publish.py --verify           # PATCH, then read back fills

Set QP_STRATEGY_ID once you know it to skip the discovery call.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://api.quantphemes.com"
OUTPUTS_DIR = Path(__file__).parent / "outputs"
TIMEOUT = 30


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------

def _headers() -> dict:
    key = os.environ.get("QP_API_KEY")
    if not key:
        sys.exit("QP_API_KEY is not set. Export your qc_live_... key first.")
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _request(method: str, path: str, payload: dict | None = None) -> dict:
    response = requests.request(
        method, f"{BASE_URL}{path}", headers=_headers(), json=payload, timeout=TIMEOUT
    )
    if not response.ok:
        sys.exit(f"{method} {path} -> HTTP {response.status_code}\n{response.text}")
    body = response.json()
    if not body.get("success", False):
        sys.exit(f"{method} {path} -> API reported failure\n{json.dumps(body, indent=2)}")
    return body["data"]


# --------------------------------------------------------------------------
# Strategy resolution
# --------------------------------------------------------------------------

def list_strategies() -> None:
    """Print every portfolio/strategy on the account so ids can be copied out."""
    portfolios = _request("GET", "/api/v1/portfolio")["portfolios"]
    for portfolio in portfolios:
        print(f"\nportfolio  {portfolio['portfolioId']}  {portfolio['name']}"
              f"  (mode={portfolio.get('mode')}, type={portfolio.get('type')})")
        strategies = portfolio.get("strategies") or []
        if not strategies:
            print("    (no strategies — create one in the web UI first)")
        for strategy in strategies:
            print(f"    strategy {strategy['strategyId']}  {strategy['name']}"
                  f"  (status={strategy.get('status')}, type={strategy.get('type')})")


def resolve_strategy_id() -> str:
    """QP_STRATEGY_ID if set, else the single strategy under QP_PORTFOLIO_ID."""
    explicit = os.environ.get("QP_STRATEGY_ID")
    if explicit:
        return explicit

    portfolio_id = os.environ.get("QP_PORTFOLIO_ID")
    if not portfolio_id:
        sys.exit("Set QP_STRATEGY_ID, or QP_PORTFOLIO_ID to auto-discover it.")

    strategies = _request(
        "GET", f"/api/v1/portfolio/{portfolio_id}/strategy"
    )["portfolio"].get("strategies") or []

    if len(strategies) != 1:
        names = ", ".join(f"{s['strategyId']} ({s['name']})" for s in strategies) or "none"
        sys.exit(f"Expected exactly 1 strategy under {portfolio_id}, found {len(strategies)}: "
                 f"{names}. Set QP_STRATEGY_ID explicitly.")

    return strategies[0]["strategyId"]


# --------------------------------------------------------------------------
# Payload
# --------------------------------------------------------------------------

def latest_csv() -> Path:
    """Most recent outputs/portfolio_<date>.csv, by filename date."""
    runs = sorted(OUTPUTS_DIR.glob("portfolio_[0-9]*.csv"))
    if not runs:
        sys.exit(f"No portfolio_<date>.csv in {OUTPUTS_DIR}. Run portfolio_engine.py first.")
    return runs[-1]


def build_payload(csv_path: Path) -> dict:
    frame = pd.read_csv(csv_path)

    missing = {"ticker", "shares", "rebalance_date"} - set(frame.columns)
    if missing:
        sys.exit(f"{csv_path.name} is missing column(s): {', '.join(sorted(missing))}")

    dates = frame["rebalance_date"].unique()
    if len(dates) != 1:
        sys.exit(f"{csv_path.name} mixes rebalance dates: {list(dates)}")

    positions = frame.loc[frame["shares"] > 0]
    if positions.empty:
        sys.exit(f"{csv_path.name} has no positions with shares > 0 — refusing to publish "
                 "an empty allocation.")

    return {
        "holdings": [{
            "effective_datetime": f"{dates[0]}T00:00:00Z",
            "stocks": [
                {"symbol": str(row.ticker), "quantity": int(row.shares)}
                for row in positions.itertuples()
            ],
        }]
    }


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------

def verify(strategy_id: str) -> None:
    """Read back current vs target quantities so drift is visible in the run log."""
    quantities = _request(
        "GET", f"/api/v1/strategy/{strategy_id}/holding/strategy-quantity"
    )["strategy_quantities"]

    print(f"\n{'symbol':<8}{'target':>10}{'current':>10}")
    for row in quantities:
        print(f"{row['symbol']:<8}{row['targetQuantity']:>10}{row['currentQuantity']:>10}")

    unfilled = [r for r in quantities if r["targetQuantity"] != r["currentQuantity"]]
    if unfilled:
        print(f"\n{len(unfilled)} symbol(s) not yet reconciled - normal right after a "
              "rebalance, worth checking if it persists past the next session.")


def check_reconciliation(strategy_id: str, csv_path: Path) -> int:
    """
    Did the market actually execute what we published? Meant to run hours after
    the open, not seconds after the PATCH -- right after publishing every
    currentQuantity is 0 by definition, which tells you nothing.

    Returns a process exit code so CI fails loudly instead of drifting silently.
    """
    quantities = _request(
        "GET", f"/api/v1/strategy/{strategy_id}/holding/strategy-quantity"
    )["strategy_quantities"]
    orders = _request("GET", f"/api/v1/strategy/{strategy_id}/order")["orders"]

    expected = {str(row.ticker): int(row.shares)
                for row in pd.read_csv(csv_path).itertuples() if row.shares > 0}
    reported = {r["symbol"]: r for r in quantities}

    problems: list[str] = []

    if not quantities:
        problems.append("the platform reports no positions at all for this strategy - "
                        "targets were published but nothing was acted on")

    # A symbol the platform never echoes back is the signature of an unsupported
    # or restricted instrument (ADRs and foreign listings are the usual cause).
    for symbol in sorted(set(expected) - set(reported)):
        problems.append(f"{symbol}: published as a target but absent from the platform's "
                        "position list - possibly an unsupported instrument")

    print(f"\n{'symbol':<8}{'target':>10}{'current':>10}  status")
    for symbol in sorted(reported):
        row = reported[symbol]
        target, current = row["targetQuantity"], row["currentQuantity"]
        gap = current - target
        status = "ok" if gap == 0 else f"SHORT {-gap}" if gap < 0 else f"OVER {gap}"
        print(f"{symbol:<8}{target:>10}{current:>10}  {status}")
        if gap != 0:
            problems.append(f"{symbol}: holds {current} against a target of {target}")

    unfilled_orders = [o for o in orders if o.get("status") != "Filled"]
    for order in unfilled_orders:
        problems.append(f"{order['symbol']}: order {order.get('side')} "
                        f"{order.get('quantity')} is {order.get('status')}, not Filled")

    if not problems:
        print(f"\nAll {len(reported)} positions reconciled against target.")
        return 0

    print(f"\n{len(problems)} problem(s) found:")
    for problem in problems:
        print(f"  - {problem}")
    print("\nMost common cause is insufficient cash: the account is HKD-denominated, so "
          "a USD-sized basket can exceed available funds at the weak end of the peg.")
    return 1


# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true",
                        help="print portfolios and strategy ids, then exit")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the payload without sending it")
    parser.add_argument("--create", action="store_true",
                        help="POST a new holding strategy instead of PATCHing targets")
    parser.add_argument("--verify", action="store_true",
                        help="after publishing, read back current vs target quantities")
    parser.add_argument("--check", action="store_true",
                        help="publish nothing; check that live positions match the last "
                             "published targets. Exits non-zero on any drift. Run this "
                             "hours after the open, not right after publishing.")
    parser.add_argument("--csv", type=Path,
                        help="publish this CSV instead of the newest run")
    args = parser.parse_args()

    if args.list:
        list_strategies()
        return

    csv_path = args.csv or latest_csv()

    if args.check:
        print(f"checking against {csv_path.name}")
        sys.exit(check_reconciliation(resolve_strategy_id(), csv_path))

    payload = build_payload(csv_path)
    stocks = payload["holdings"][0]["stocks"]

    print(f"source     {csv_path.name}")
    print(f"effective  {payload['holdings'][0]['effective_datetime']}")
    print(f"positions  {len(stocks)}  ({sum(s['quantity'] for s in stocks)} shares total)")

    if args.dry_run:
        print("\n--dry-run, nothing sent:\n")
        print(json.dumps(payload, indent=2))
        return

    strategy_id = resolve_strategy_id()

    if args.create:
        payload = {"name": "Markowitz-Optimization",
                   "description": "Weekly GMV portfolio over NASDAQ-100 constituents",
                   **payload}

    data = _request(
        "POST" if args.create else "PATCH",
        f"/api/v1/strategy/{strategy_id}/holding",
        payload,
    )
    print(f"\npublished to strategy {data.get('strategy_id', strategy_id)}")

    if args.verify:
        verify(strategy_id)


if __name__ == "__main__":
    main()
