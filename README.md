# Weekly NASDAQ-100 Portfolio Engine

Builds a 10-stock, $248,000 portfolio from NASDAQ-100 constituents each week,
grounded in portfolio theory (not signals). See the module docstring in
`portfolio_engine.py` for the full methodology and the papers behind each step.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python portfolio_engine.py
```

This will:
1. Pull the current NASDAQ-100 list (Wikipedia, with a static fallback baked in)
2. Download ~2 years of weekly prices for all ~100 names via `yfinance`
3. Cluster the universe and select 10 structurally diversified stocks
4. Compute three weighting schemes on those 10: Markowitz GMV (Ledoit-Wolf
   shrinkage), Hierarchical Risk Parity, and equal weight
5. Convert the chosen scheme (`PRIMARY_METHOD` in the config block) into
   integer share counts within the $248,000 budget
6. Write `outputs/portfolio_<date>.csv` and append to `outputs/portfolio_history.csv`

## How the Markowitz problem is actually solved

### 1. Which Markowitz problem — GMV, not tangency

The engine solves the **Global Minimum Variance (GMV)** portfolio, *not* the full
mean-variance / maximum-Sharpe portfolio. Expected returns are never estimated at all.

```
minimize    w' Σ w
   w
subject to  1' w = 1                     (fully invested)
            l ≤ w_i ≤ u   for all i      (l = MIN_WEIGHT = 3%, u = MAX_WEIGHT = 30%)
```

Note that `w_i ≥ l > 0` makes the portfolio long-only by construction — no short-sale
constraint has to be written separately.

Dropping the expected-return vector `μ` is a deliberate modelling choice and is the single
largest source of robustness in the whole pipeline. Mean-variance weights are notoriously
sensitive to `μ`, whose sample estimate converges far too slowly to be usable over a 2-year
window (Merton, 1980); the resulting "error maximization" is the standard critique of naive
Markowitz (Michaud, 1989). GMV is the one point on the efficient frontier that requires no
`μ` estimate whatsoever, which is why it is the frontier point that most reliably survives
out-of-sample (DeMiguel, Garlappi & Uppal, 2009).

### 2. How it is solved numerically

The unconstrained GMV problem has the well-known closed form

```
w* = Σ⁻¹ 1 / (1' Σ⁻¹ 1)
```

but that solution is only valid when the sum-to-one equality is the *only* constraint. It
routinely returns negative weights, and it cannot respect the 3%/30% box. So the engine
does **not** use it.

Instead the problem is solved numerically with **SLSQP** (Sequential Least Squares
Quadratic Programming) via `scipy.optimize.minimize` — see
[`markowitz_gmv_weights`](portfolio_engine.py#L232-L247):

- **Objective**: `w ↦ w' Σ w`, supplied directly; SLSQP builds its own gradient/Hessian
  approximations. Since Σ is positive definite (guaranteed by the shrinkage step below),
  the objective is strictly convex and the box + equality constraints define a convex
  feasible set, so the local optimum SLSQP converges to **is** the global optimum.
- **Equality constraint**: `1'w − 1 = 0`, passed as a `{"type": "eq"}` constraint.
- **Box constraints**: passed as `bounds`, so they are handled by the solver's active-set
  machinery rather than by penalization.
- **Start point**: equal weight `w₀ = 1/n`, which is always strictly feasible here
  (`n·l = 0.30 ≤ 1 ≤ n·u = 3.0`).
- **Failure handling**: if `result.success` is `False` the function raises rather than
  returning a silently-wrong allocation.
- A final `clip(0, None)` + renormalization cleans up float residue on the order of 1e-16
  so the weights sum to exactly 1 before being turned into share counts.

Problem size is tiny (n = 10 after selection), so SLSQP converges in milliseconds; there is
no need for a dedicated QP solver such as OSQP or CVXOPT.

### 3. What Σ is

Σ is the **Ledoit-Wolf shrinkage** covariance of *weekly* returns over a 2-year lookback
(≈104 bars), estimated in [`shrinkage_covariance`](portfolio_engine.py#L181-L184). It is
estimated twice: once on the full ~100-name universe (used only for the clustering /
selection step) and again on just the 10 selected names (used for the actual optimization).
Weekly rather than daily returns to match the weekly rebalance cadence and to avoid
non-synchronous-trading and microstructure noise in the correlation estimates.

## Regularization

**Yes — regularization is used, in three distinct places.** None of them is a `λ‖w‖²`
penalty bolted onto the objective; they are instead the structural forms of regularization
standard in the portfolio literature.

### (a) Shrinkage of the covariance matrix — Ledoit-Wolf

This is the explicit statistical regularizer. The sample covariance `S` is shrunk toward a
structured, low-variance target `F` (constant-correlation / scaled-identity):

```
Σ̂ = (1 − δ) S + δ F ,        δ ∈ [0, 1]
```

The shrinkage intensity `δ` is not tuned by hand or cross-validated — it is chosen
**analytically** to minimize the expected Frobenius loss `E‖Σ̂ − Σ‖²` (Ledoit & Wolf, 2004),
via `sklearn.covariance.LedoitWolf`. The script prints the fitted `δ` for both estimation
steps on every run, so the degree of regularization is auditable rather than hidden:

```
[cov] full-universe Ledoit-Wolf shrinkage intensity: 0.xxx
[cov] selected-10 Ledoit-Wolf shrinkage intensity: 0.xxx
```

Why it matters here, quantitatively: at the selection step there are p ≈ 100 assets and
T ≈ 103 return observations, so **p/T ≈ 1**. The sample covariance is at best catastrophically
ill-conditioned and at worst singular; its smallest eigenvalues are pure noise, and GMV — which
depends on `Σ⁻¹` — loads precisely on those smallest eigenvalues. Shrinkage guarantees a
well-conditioned, invertible, positive-definite Σ and is what makes the correlation matrix
fed into the clustering step meaningful. At the weighting step (p = 10, T ≈ 103) the ratio is
much healthier, so `δ` will typically be far smaller there — but the estimator adapts on its
own.

### (b) Weight constraints as implicit shrinkage — Jagannathan-Ma

The 3%/30% box is not just cosmetic housekeeping to keep exactly 10 live positions.
Jagannathan & Ma (2003) proved that for the GMV problem, **imposing weight constraints is
mathematically equivalent to shrinking the covariance matrix**. If `λ_i ≥ 0` are the KKT
multipliers on the upper bounds and `δ_i ≥ 0` those on the lower bounds, then the
constrained solution is exactly the *unconstrained* GMV solution for the modified matrix

```
σ̃_ij = σ_ij + (λ_i + λ_j) − (δ_i + δ_j)
```

i.e. binding a cap inflates that asset's estimated covariances (pushing weight away from
names whose low estimated variance was probably an estimation artifact), and binding a floor
deflates them. So the box constraints are a *second*, non-redundant layer of regularization
stacked on top of Ledoit-Wolf — and JM's empirical result is that this layer alone often
improves out-of-sample GMV performance more than a better covariance estimator does.

### (c) The long-only constraint as an L1 constraint

Because weights are forced non-negative *and* sum to one, the L1 norm of the weight vector
is pinned at `‖w‖₁ = 1` — the tightest possible L1 ball for a fully-invested portfolio. The
no-short constraint is therefore literally an L1 regularizer, which places this portfolio
inside the norm-constrained family that DeMiguel et al. (2009, *Management Science* 55(5))
study explicitly. The 30% cap additionally bounds `‖w‖∞`, and combined with the 3% floor it
pins `‖w‖₂²` to `[0.100, 0.222]` for n = 10 (the upper vertex being two positions at the 30%
cap, one at 19%, seven at the 3% floor). So the effective number of positions,
`1/‖w‖₂²`, is guaranteed to stay between 4.5 and 10 no matter what the optimizer wants to do
— the "10-stock portfolio" can never degenerate into a concentrated 2- or 3-stock bet.

### What is deliberately *not* done

- **No explicit ridge/lasso penalty term** on `w` in the objective. It would be redundant
  given (a)-(c), and it would add a hyperparameter with no principled way to set it here
  (there is no held-out validation criterion in a pure-theory allocator that makes no
  return forecast).
- **No resampled / bootstrap-averaged frontier** (Michaud, 1998). Defensible, and cheap to
  add, but it trades determinism for smoothness and would make week-to-week output
  non-reproducible without seed management.
- **No Black-Litterman**. It exists to blend *views on returns* into the prior, and this
  engine has no views and no returns.
- **No turnover penalty / no-trade region** across weekly rebalances. This is the most
  defensible thing to add next if the supervisor cares about implementability —
  `outputs/portfolio_history.csv` already logs everything needed to measure realized
  week-over-week turnover first.

### References

- Markowitz, H. (1952). "Portfolio Selection." *Journal of Finance* 7(1), 77-91.
- Merton, R. (1980). "On estimating the expected return on the market." *JFE* 8(4), 323-361.
- Michaud, R. (1989). "The Markowitz optimization enigma: is optimized optimal?" *FAJ* 45(1).
- Ledoit, O. & Wolf, M. (2004). "A well-conditioned estimator for large-dimensional
  covariance matrices." *Journal of Multivariate Analysis* 88(2), 365-411.
- Jagannathan, R. & Ma, T. (2003). "Risk reduction in large portfolios: why imposing the
  wrong constraints helps." *Journal of Finance* 58(4), 1651-1683.
- DeMiguel, V., Garlappi, L. & Uppal, R. (2009). "Optimal versus naive diversification."
  *Review of Financial Studies* 22(5), 1915-1953.
- DeMiguel, V., Garlappi, L., Nogales, F. & Uppal, R. (2009). "A generalized approach to
  portfolio optimization: improving performance by constraining portfolio norms."
  *Management Science* 55(5), 798-812.
- Lopez de Prado, M. (2016). "Building diversified portfolios that outperform out of
  sample." *Journal of Portfolio Management* 42(4), 59-69.

## Weekly scheduling

This is a script, not a daemon — point a scheduler at it once a week, e.g.:

```cron
# every Monday 7:00am, before market open
0 7 * * 1 cd /path/to/project && python3 portfolio_engine.py >> logs/run.log 2>&1
```

(Windows: Task Scheduler; macOS: `launchd` or just cron if enabled.)

## Config knobs (top of `portfolio_engine.py`)

| Variable | Meaning |
|---|---|
| `BUDGET_USD` | Total dollars to deploy (default $248,000, sized to an HKD 1.95m account) |
| `N_STOCKS` | Portfolio size (default 10) |
| `LOOKBACK_YEARS` | Price history window for covariance estimation |
| `PRIMARY_METHOD` | Which weighting scheme actually sizes the shares: `"markowitz_gmv"`, `"hrp"`, or `"equal_weight"` — the other two are still computed and saved as comparison columns |
| `MIN_WEIGHT` / `MAX_WEIGHT` | Position bounds for the GMV optimizer. Prevents corner solutions from zeroing out a selected stock, and doubles as implicit covariance shrinkage — see [Regularization (b)](#b-weight-constraints-as-implicit-shrinkage--jagannathan-ma) |

## Output CSV columns

`ticker, cluster, price, shares, dollar_allocation, actual_weight, target_weight,
markowitz_gmv/hrp/equal_weight (whichever weren't primary), rebalance_date, primary_method`

## Notes / known limitations

- Yahoo Finance data only, as specified — no fundamentals, no alt data.
- The NASDAQ-100 Wikipedia scrape can break if the page format changes; the
  script falls back to a static list embedded in the script. **Verify that
  list's freshness periodically** — index membership changes at the annual
  December reconstitution and occasionally intra-year.
- This is explicitly *not* trying to be profitable — no return forecasting,
  no momentum/value signals. Turnover between weekly runs is not currently
  penalized; if turnover matters for your write-up, `portfolio_history.csv`
  gives you what you need to compute week-over-week turnover yourself.
