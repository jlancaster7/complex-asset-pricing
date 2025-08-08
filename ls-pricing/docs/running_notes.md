# Running Notes and Future Additions

Date: 2025-08-07

## Regression diagnostics (callable bond LS fits)
Observed on 30Y, 10NC, 4000 paths, Laguerre degree=3:
- Early callable window (~10–16y): low R² (~0.08–0.25), higher MSE; continuation harder to learn far from maturity.
- R² improves steadily as maturity approaches; near-maturity fits are excellent (~0.95→1.00) with very low MSE.
- ITM samples are large throughout (good statistical power).
- Coefficient magnitudes early are very large (ill-conditioning risk with OLS + Laguerre).

## Recent engineering changes
- Unified discounting via Monte Carlo path discount factor ratios; verified convergence behavior improved and notebook parity maintained.
- Implemented analytic theta using spline derivatives; simulation drift stabilized (less noise in early times).
- Added include_coupon_on_call flag to reflect market convention differences.
- Added regression diagnostics capture per exercise time; exposed in methodology notebook with summary and R² plot.
- Factored out yield curve CSV parsing into utils/curve_io.py; updated tests and examples to use load_yield_curve_from_csv.

## Recommendations (short-term)
- Try ridge regression for stability (e.g., regression_type="ridge", alpha in [1e-4, 1e-2]).
- Standardize state (z-score short rate) or switch to a numerically friendlier basis in early window.
- Add bond price as an additional state feature (rate-only may be insufficient far from maturity).
- Consider adaptive basis degree: cap at 2 when R² < threshold; raise with more ITM support.
- Increase paths in early window if feasible to boost fit quality.

## Engine consistency and methodology
- Unify callable engine discounting with path discount factors (reuse MonteCarloEngine discount_factors ratios) and include credit spread via exp(-s·Δt) multipliers.
- Keep coupon-on-call convention configurable (done); document in API and notebook.
- Persist regression diagnostics in results (done) and add simple quality flags in outputs (R² low, small ITM sample).

## Testing and docs
- Add tests asserting diagnostics presence and reasonable monotonicity near maturity (high R², low MSE).
- Add example in notebook comparing OLS vs Ridge R² curves.
- Update code review doc once discounting is unified and ridge option evaluated.

## Next prioritized tasks (from review)
1. Callable discounting unification with path DF ratios + credit spread handling.
2. Add warning on unimplemented swaption calibration stub.
3. Optional: expose regression_type and alpha in notebook to compare OLS vs Ridge quickly.
