"""
Weekly NASDAQ-100 Portfolio Construction Engine
=================================================

Builds a 10-stock, $250,000 portfolio from NASDAQ-100 constituents on a
weekly cadence. This is a THEORY-DRIVEN allocator, not a signal-seeking
one: it does not forecast returns, time the market, or use any
information beyond price history. Its only job is to turn one of a few
classical portfolio-construction theories into a concrete list of 10
tickers and dollar amounts each week.

Three allocation schemes are computed side by side so the underlying
theory is auditable, not just the final output:

    1. Markowitz Global Minimum Variance (GMV) with Ledoit-Wolf
       shrinkage covariance     -- Markowitz (1952); Ledoit & Wolf (2004)
    2. Hierarchical Risk Parity (HRP)
                                 -- Lopez de Prado (2016), JPM 42(4), 59-69
    3. Equal Weight (1/N) benchmark
                                 -- DeMiguel, Garlappi & Uppal (2009), RFS 22(5)

STOCK SELECTION (which 10 of ~100 names) is done separately from
weighting: hierarchical clustering on the correlation matrix groups the
universe into 10 clusters, and the most "central" stock in each cluster
is picked as its representative. This guarantees the 10 names span
distinct return-correlation clusters before any weighting math happens,
rather than an optimizer arbitrarily concentrating in one corner of the
universe.

Data source: Yahoo Finance ONLY, via the `yfinance` package.

Usage
-----
    python portfolio_engine.py

Output
------
    outputs/portfolio_<YYYY-MM-DD>.csv   <- this week's 10-stock portfolio
    outputs/portfolio_history.csv        <- running log, appended to weekly

Suggested scheduling: run this once a week (e.g. a cron job or a
Windows Task Scheduler job set for Monday pre-market) so PRIMARY_METHOD
becomes this week's live monitoring list.
"""

import io
import os
import warnings
from datetime import date

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
BUDGET_USD = 245_000  # ~98% of the HKD 1.95m account. The account is denominated in
                      # HKD, which the peg holds only within 7.75-7.85, so a full
                      # 250k of orders can exceed available cash at the weak end and
                      # get the last legs rejected. The ~5k buffer also covers fees.
N_STOCKS = 10                 # final portfolio size
LOOKBACK_YEARS = 2            # price history window used to estimate covariance
MIN_HISTORY_WEEKS = 78        # ~1.5 years; tickers with less history are dropped
PRIMARY_METHOD = "markowitz_gmv"   # "markowitz_gmv" | "hrp" | "equal_weight"
                                    # -> this is the method actually used to size shares.
                                    # The other two are still computed and saved for comparison.
MIN_WEIGHT = 0.03      # per-position floor for the GMV optimizer (3%)
MAX_WEIGHT = 0.30      # per-position cap for the GMV optimizer (30%)
                        # Unconstrained long-only GMV frequently drives some weights to
                        # exactly zero ("corner solutions") -- fine in theory, but it
                        # silently turns a "10-stock portfolio" into an 8- or 9-stock one.
                        # These bounds are a standard practical fix (see e.g. Jagannathan &
                        # Ma, 2003, on how simple weight constraints help GMV out-of-sample)
                        # and guarantee the ask -- exactly 10 live positions -- is met.
OUTPUT_DIR = "outputs"

# Wikipedia blocks requests that use the default Python-urllib User-Agent and
# asks API/scraping clients to identify themselves (see their UA policy at
# https://foundation.wikimedia.org/wiki/Policy:User-Agent_policy).
WIKI_USER_AGENT = "portfolio-engine/1.0 (educational research script; contact: aidarkhanzhubatkan@gmail.com)"

# Static fallback universe, used only if the live Wikipedia pull fails.
# NOTE: NASDAQ-100 membership changes periodically (annual December
# reconstitution plus occasional intra-year swaps). Verify against
# https://en.wikipedia.org/wiki/Nasdaq-100 before relying on this list
# for anything beyond quick testing.
NASDAQ100_FALLBACK = [
    "AAPL","MSFT","AMZN","NVDA","GOOGL","GOOG","META","AVGO","TSLA","COST",
    "NFLX","ASML","AMD","PEP","ADBE","CSCO","LIN","TMUS","QCOM","INTU",
    "AMAT","TXN","CMCSA","AMGN","HON","BKNG","ISRG","VRTX","PANW","ADP",
    "SBUX","MU","REGN","GILD","LRCX","MDLZ","ADI","PYPL","KLAC","SNPS",
    "CDNS","MELI","CSX","MAR","ORLY","CRWD","ABNB","PDD","CTAS","FTNT",
    "NXPI","MRVL","WDAY","ROP","PCAR","MNST","DASH","AEP","ROST","ODFL",
    "PAYX","KDP","EXC","FAST","BKR","EA","VRSK","XEL","CHTR","CPRT",
    "DDOG","CTSH","GEHC","IDXX","LULU","KHC","CCEP","TTD","ANSS","ON",
    "GFS","MCHP","BIIB","ZS","TEAM","DXCM","WBD","ILMN","MDB","SIRI",
    "WBA","SMCI","ARM","APP","MSTR","AXON","CDW","DLTR","FANG","GLBE",
]


# --------------------------------------------------------------------------
# 1. UNIVERSE
# --------------------------------------------------------------------------
def get_nasdaq100_tickers() -> list:
    """Fetch current NASDAQ-100 constituents from Wikipedia; fall back to
    the static list above if that fails (no internet, page format change, etc.)."""
    # The constituents table lives on the "List of ..." page; the main index
    # page is kept as a secondary try in case the split is ever reverted.
    urls = [
        "https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies",
        "https://en.wikipedia.org/wiki/Nasdaq-100",
    ]
    for url in urls:
        try:
            # Wikipedia rejects the default urllib User-Agent that pandas.read_html
            # sends (HTTP 403), so fetch the page ourselves with a real UA and hand
            # the HTML to read_html directly.
            resp = requests.get(url, headers={"User-Agent": WIKI_USER_AGENT}, timeout=20)
            resp.raise_for_status()
            tables = pd.read_html(io.StringIO(resp.text))
            for t in tables:
                # Flatten any MultiIndex header (the "changes" table uses one).
                cols = [" ".join(str(p) for p in c).lower() if isinstance(c, tuple)
                        else str(c).lower() for c in t.columns]
                matches = [i for i, c in enumerate(cols) if "ticker" in c or "symbol" in c]
                if not matches:
                    continue
                tickers = t.iloc[:, matches[0]].astype(str).str.strip()
                # Keep only plausible symbols (1-5 letters, optional class suffix);
                # drops footnote rows and stray text if the table layout shifts.
                tickers = [tk for tk in tickers
                           if tk.replace(".", "").isalpha() and 1 <= len(tk) <= 6]
                if len(tickers) >= 90:
                    print(f"[universe] pulled {len(tickers)} tickers from Wikipedia")
                    return [tk.replace(".", "-") for tk in tickers]
            raise ValueError("no suitable table found")
        except Exception as e:
            print(f"[universe] live pull from {url.rsplit('/', 1)[-1]} failed ({e})")
    print(f"[universe] using static fallback list "
          f"({len(NASDAQ100_FALLBACK)} tickers, verify freshness manually)")
    return NASDAQ100_FALLBACK


# --------------------------------------------------------------------------
# 2. DATA
# --------------------------------------------------------------------------
def download_price_data(tickers: list, lookback_years: int) -> pd.DataFrame:
    """Weekly adjusted close prices for `tickers` over the lookback window.
    Drops any ticker with insufficient history."""
    raw = yf.download(
        tickers,
        period=f"{lookback_years}y",
        interval="1wk",
        auto_adjust=True,
        progress=False,
        group_by="column",
    )
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    prices = prices.dropna(axis=1, thresh=MIN_HISTORY_WEEKS)
    prices = prices.ffill().dropna(axis=0, how="any")
    dropped = set(tickers) - set(prices.columns)
    if dropped:
        print(f"[data] dropped {len(dropped)} tickers for insufficient history: {sorted(dropped)}")
    print(f"[data] {prices.shape[1]} tickers x {prices.shape[0]} weekly bars "
          f"({prices.index.min().date()} to {prices.index.max().date()})")
    return prices


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna(how="all").dropna(axis=1)


# --------------------------------------------------------------------------
# 3. COVARIANCE (Ledoit-Wolf shrinkage)
# --------------------------------------------------------------------------
def shrinkage_covariance(returns: pd.DataFrame):
    lw = LedoitWolf().fit(returns.values)
    cov = pd.DataFrame(lw.covariance_, index=returns.columns, columns=returns.columns)
    return cov, lw.shrinkage_


def corr_from_cov(cov: pd.DataFrame) -> pd.DataFrame:
    d = np.sqrt(np.diag(cov.values))
    corr = cov.values / np.outer(d, d)
    corr = np.clip(corr, -1.0, 1.0)
    return pd.DataFrame(corr, index=cov.index, columns=cov.columns)


# --------------------------------------------------------------------------
# 4. SELECTION: cluster the full universe, pick 1 representative / cluster
# --------------------------------------------------------------------------
def select_diversified_universe(corr: pd.DataFrame, n_stocks: int) -> list:
    dist = np.sqrt(0.5 * (1 - corr.values))
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)
    link = linkage(condensed, method="average")
    labels = fcluster(link, n_stocks, criterion="maxclust")

    tickers = corr.index.to_numpy()
    selected = []
    cluster_map = {}
    for c in sorted(set(labels)):
        members = tickers[labels == c]
        cluster_map[int(c)] = list(members)
        if len(members) == 1:
            rep = members[0]
        else:
            sub = dist[np.ix_(labels == c, labels == c)]
            avg_dist = sub.mean(axis=1)
            rep = members[np.argmin(avg_dist)]  # most "central" member of its cluster
        selected.append(rep)

    # fcluster can occasionally return fewer than n_stocks clusters for
    # highly correlated universes; top up from the largest remaining
    # cluster by second-most-central member if that happens.
    if len(selected) < n_stocks:
        print(f"[selection] only {len(selected)} clusters formed; topping up")
        remaining = [t for t in tickers if t not in selected]
        selected += remaining[: n_stocks - len(selected)]

    return selected[:n_stocks], cluster_map


# --------------------------------------------------------------------------
# 5a. WEIGHTING: Markowitz Global Minimum Variance
# --------------------------------------------------------------------------
def markowitz_gmv_weights(cov: pd.DataFrame) -> pd.Series:
    n = cov.shape[0]
    Sigma = cov.values

    def portfolio_var(w):
        return w @ Sigma @ w

    w0 = np.repeat(1 / n, n)
    bounds = [(MIN_WEIGHT, MAX_WEIGHT)] * n
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    result = minimize(portfolio_var, w0, method="SLSQP", bounds=bounds, constraints=constraints)
    if not result.success:
        raise RuntimeError(f"GMV optimization failed: {result.message}")
    w = np.clip(result.x, 0, None)
    w = w / w.sum()
    return pd.Series(w, index=cov.index, name="markowitz_gmv")


# --------------------------------------------------------------------------
# 5b. WEIGHTING: Hierarchical Risk Parity (Lopez de Prado, 2016)
# --------------------------------------------------------------------------
def _get_quasi_diag(link):
    """Recover the leaf order implied by a linkage matrix (quasi-diagonalization)."""
    link = link.astype(int)
    n = link.shape[0] + 1
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = link[-1, 3]
    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        df0 = sort_ix[sort_ix >= num_items]
        i = df0.index
        j = df0.values - num_items
        sort_ix[i] = link[j, 0]
        df1 = pd.Series(link[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df1]).sort_index()
        sort_ix.index = range(sort_ix.shape[0])
    return sort_ix.tolist()


def _cluster_var(cov, items):
    sub = cov.loc[items, items].values
    ivp = 1.0 / np.diag(sub)
    ivp /= ivp.sum()
    return ivp @ sub @ ivp


def _recursive_bisection(cov, sorted_tickers):
    weights = pd.Series(1.0, index=sorted_tickers)
    clusters = [sorted_tickers]
    while clusters:
        clusters = [c[start:end] for c in clusters
                    for start, end in ((0, len(c) // 2), (len(c) // 2, len(c)))
                    if len(c) > 1]
        for i in range(0, len(clusters), 2):
            left, right = clusters[i], clusters[i + 1]
            var_left = _cluster_var(cov, left)
            var_right = _cluster_var(cov, right)
            alpha = 1 - var_left / (var_left + var_right)
            weights[left] *= alpha
            weights[right] *= 1 - alpha
    return weights


def hrp_weights(cov: pd.DataFrame, corr: pd.DataFrame) -> pd.Series:
    dist = np.sqrt(0.5 * (1 - corr.values))
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)
    link = linkage(condensed, method="single")  # single linkage, per the original HRP paper
    sort_ix = _get_quasi_diag(link)
    sorted_tickers = corr.index[sort_ix].tolist()
    w = _recursive_bisection(cov, sorted_tickers)
    return w.reindex(cov.index).rename("hrp")


# --------------------------------------------------------------------------
# 5c. WEIGHTING: Equal weight benchmark
# --------------------------------------------------------------------------
def equal_weights(tickers: list) -> pd.Series:
    return pd.Series(np.repeat(1 / len(tickers), len(tickers)), index=tickers, name="equal_weight")


# --------------------------------------------------------------------------
# 6. CONVERT WEIGHTS -> SHARES WITHIN BUDGET
# --------------------------------------------------------------------------
def weights_to_shares(weights: pd.Series, prices: pd.Series, budget: float) -> pd.DataFrame:
    target_dollars = weights * budget
    shares = np.floor(target_dollars / prices).astype(int)
    spent = shares * prices
    leftover = budget - spent.sum()

    # Greedily spend leftover cash on additional whole shares, always
    # buying the name currently furthest below its target dollar weight.
    guard = 0
    while guard < 500:
        current_dollars = shares * prices
        shortfall = target_dollars - current_dollars
        affordable = shortfall[(prices <= leftover) & (shortfall > 0)]
        if affordable.empty:
            break
        pick = affordable.idxmax()
        shares[pick] += 1
        leftover -= prices[pick]
        guard += 1

    out = pd.DataFrame({
        "price": prices,
        "target_weight": weights,
        "shares": shares,
        "dollar_allocation": shares * prices,
    })
    out["actual_weight"] = out["dollar_allocation"] / out["dollar_allocation"].sum()
    return out, leftover


# --------------------------------------------------------------------------
# 7. REPORTING HELPERS
# --------------------------------------------------------------------------
def diversification_ratio(weights: pd.Series, cov: pd.DataFrame) -> float:
    w = weights.values
    vol = np.sqrt(np.diag(cov.values))
    weighted_avg_vol = w @ vol
    port_vol = np.sqrt(w @ cov.values @ w)
    return weighted_avg_vol / port_vol


def annualized_vol(weights: pd.Series, cov: pd.DataFrame) -> float:
    w = weights.values
    weekly_var = w @ cov.values @ w
    return np.sqrt(weekly_var) * np.sqrt(52)


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    today = date.today().isoformat()

    # -- universe & data --------------------------------------------------
    universe = get_nasdaq100_tickers()
    prices_full = download_price_data(universe, LOOKBACK_YEARS)
    returns_full = compute_returns(prices_full)

    # -- covariance on the FULL universe, used only for selection --------
    cov_full, shrink_full = shrinkage_covariance(returns_full)
    corr_full = corr_from_cov(cov_full)
    print(f"[cov] full-universe Ledoit-Wolf shrinkage intensity: {shrink_full:.3f}")

    # -- select 10 structurally diversified names -------------------------
    selected, cluster_map = select_diversified_universe(corr_full, N_STOCKS)
    print(f"[selection] chosen tickers: {selected}")

    # -- re-estimate covariance on JUST the 10 selected names -------------
    returns_sel = returns_full[selected]
    cov_sel, shrink_sel = shrinkage_covariance(returns_sel)
    corr_sel = corr_from_cov(cov_sel)
    print(f"[cov] selected-10 Ledoit-Wolf shrinkage intensity: {shrink_sel:.3f}")

    # -- compute all three weighting schemes -------------------------------
    w_gmv = markowitz_gmv_weights(cov_sel)
    w_hrp = hrp_weights(cov_sel, corr_sel)
    w_eq = equal_weights(selected)

    weights_table = pd.concat([w_gmv, w_hrp, w_eq], axis=1)

    # -- size the actual portfolio using PRIMARY_METHOD --------------------
    primary_weights = {"markowitz_gmv": w_gmv, "hrp": w_hrp, "equal_weight": w_eq}[PRIMARY_METHOD]
    last_prices = prices_full[selected].iloc[-1]
    portfolio, leftover = weights_to_shares(primary_weights, last_prices, BUDGET_USD)
    portfolio = portfolio.join(weights_table.drop(columns=[primary_weights.name]))
    portfolio.insert(0, "ticker", portfolio.index)
    portfolio.insert(1, "cluster", [
        next(c for c, members in cluster_map.items() if t in members) for t in portfolio.index
    ])
    portfolio["rebalance_date"] = today
    portfolio["primary_method"] = PRIMARY_METHOD
    comparison_cols = [c for c in ("markowitz_gmv", "hrp", "equal_weight") if c in portfolio.columns]
    portfolio = portfolio[[
        "ticker", "cluster", "price", "shares", "dollar_allocation", "actual_weight",
        "target_weight", *comparison_cols, "rebalance_date", "primary_method",
    ]]

    # -- console summary -----------------------------------------------
    print("\n=== Portfolio summary ===")
    print(portfolio[["ticker", "shares", "price", "dollar_allocation", "actual_weight"]]
          .to_string(index=False))
    print(f"\nBudget: ${BUDGET_USD:,.2f}  |  Deployed: ${portfolio['dollar_allocation'].sum():,.2f}"
          f"  |  Cash leftover: ${leftover:,.2f}")
    print(f"Diversification ratio ({PRIMARY_METHOD}): "
          f"{diversification_ratio(primary_weights, cov_sel):.3f}")
    print(f"Estimated annualized volatility ({PRIMARY_METHOD}): "
          f"{annualized_vol(primary_weights, cov_sel):.1%}")

    # -- save -------------------------------------------------------------
    out_path = os.path.join(OUTPUT_DIR, f"portfolio_{today}.csv")
    portfolio.to_csv(out_path, index=False)
    print(f"\n[output] saved this week's portfolio to {out_path}")

    history_path = os.path.join(OUTPUT_DIR, "portfolio_history.csv")
    header = not os.path.exists(history_path)
    portfolio.to_csv(history_path, mode="a", header=header, index=False)
    print(f"[output] appended to running history log at {history_path}")

    return portfolio


if __name__ == "__main__":
    main()
