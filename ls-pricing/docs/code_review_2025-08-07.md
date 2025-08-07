# LS Pricing Codebase Review (2025-08-07)

## 1. Overview
The repository implements Monte Carlo pricing for interest-rate contingent instruments using a 1-factor Hull-White short rate model and the Longstaff–Schwartz (LS) regression framework for early exercise. A callable fixed-rate bond (investor short an issuer call) is the primary structured product currently supported. Package management uses Poetry.

High-level layers:
1. Core
   - `YieldCurve`: cubic-spline interpolated zero/spot rate curve with simple flat extrapolation at ends.
   - `HullWhiteModel`: path simulation (Euler) and zero coupon bond pricing formula (closed form) with a finite-difference constructed mean-reversion term θ(t). Calibration is a placeholder.
2. Engine
   - `MonteCarloEngine`: orchestrates short-rate path generation and builds approximate discount factors per path.
   - `LongstaffSchwartzEngine`: generic American (on a zero coupon bond) regression engine using short rate as the single state variable (Laguerre or polynomial basis; OLS or Ridge).
   - `CallableBondEngine`: specialization applying LS ideas to structural callable bond valuation from the issuer’s decision perspective with regression smoothing of continuation values at call dates.
3. Instruments
   - `CouponBond`: deterministic fixed coupon schedule with optional constant credit spread added to risk-free discounting.
   - `CallableBond`: extends `CouponBond` with first call date & call price.
4. Tests
   - Broad unit tests for curve integrity, Hull-White dynamics, MC engine correctness, LS pricing sanity (American ≥ European), callable bond economics, reproducibility, and data-driven notebook replication.

## 2. Data / Control Flow
```
YieldCurve -> HullWhiteModel.simulate_short_rate -> MonteCarloEngine.generate_paths
            -> (rates, times, discount_factors ≈ exp(-∫ r dt))
   |-> LongstaffSchwartzEngine: regress continuation (future CF PV) on basis(state=short rate)
   |-> CallableBondEngine: backward issuer PV recursion + optional regression smoothing at call nodes; derive investor value and option value (difference vs straight bond PV).
```

## 3. Key Components Summary
- Yield curve interpolation: `CubicSpline`, but endpoints clamped manually (flat extrapolation) per rate query; discount factors exp(-r * t) (implicitly assumes continuously compounded spot curve aligns with instantaneous short rate at t).
- Hull-White path simulation: Euler step
  r_{t+dt} = r_t + (θ(t) - a r_t) dt + σ dW
  with θ(t) approximated via forward differences of forward rates using tiny dt = 1e-6.
- Zero coupon pricing uses standard affine form P(t,T) = A(t,T) exp(-B(t,T) r_t) with internally reconstructed A through initial term structure & model parameters.
- LS Implementation: Only short rate as regressor; continuation estimation discounts future realized CFs using average path rates — a simplification.
- Callable Bond: Maintains issuer value array (negative of investor PV). At call dates: if `immediate issuer value` (call now) dominates smoothed continuation, exercise. Regression fitted only on in-the-money (bond value > call price) states.

## 4. Issues / Risks Identified
1. Missing attribute: `YieldCurve.__init__` never sets `self.curve_date`, yet tests reference it and `shift_curve` uses it → latent AttributeError.
2. `YieldCurve.shift_curve` passes `self.curve_date` which currently does not exist; also fails to preserve original day count if expectation expands later.
3. Callable bond regression logging block references variables (`exercise`, `immediate_issuer_value`) before definition if logging enabled → potential NameError.
4. Longstaff-Schwartz `_discount_future_cashflows` uses `dt = times[i+1] - times[0]` instead of time relative to current step; likely incorrect discount factor accumulation (time origin misuse) and mixes average historical short rate rather than integral over intervals.
5. Continuation discounting / PV logic throughout (LS + callable) relies on simple arithmetic averages of short rates — introduces bias under mean-reverting short-rate; model affords closed-form conditional expectations that are unused.
6. Monte Carlo discount factors: pathwise approximation exp(-∑ r Δt) but discount_factors currently computed only once (for rates[:, :-1]) and prepended with 1; no direct reuse in LS discounting (duplication of logic / inconsistency).
7. Hull-White θ(t): finite-difference over extremely small dt (1e-6) on a cubic spline of discrete tenors can amplify noise and produce numerical instability; derivative of spline could be used directly, or analytic θ given initial curve.
8. Zero bond pricing: retrieval of `f_0_t` uses spot at 0 when t=0 else forward(0,t); may not correspond to instantaneous forward exactly; potential small mismatch with market discount factor bootstrap.
9. CallableBondEngine economic convention: adding coupon to call payment if call occurs on coupon date — needs confirmation (market conventions vary; sometimes coupon is paid, sometimes not, depending on notice period and call style).
10. Regression state space: only short rate; callable value strongly depends on remaining coupon PV (bond price). Using bond price or multi-factor basis (rate, rate^2, bond price) may improve stability.
11. Option value sign handling: perspective toggle only flips option value but not other derived quantities — consistent now but document convention.
12. Calibration: `calibrate_to_swaptions` is a stub returning initial parameters — incomplete for production; no error or warning to user.
13. Performance: several Python loops (bond valuation per path, backward induction updating future cash flows) could be vectorized or JIT compiled (Numba already in deps but unused).
14. Hard-coded logging steps `[20,25,30]` assume certain grid size; brittle.
15. Potential double discount / inconsistency risk: issuer continuation values computed then regressed again; ensure no double application once corrected discount routine implemented.
16. Test dependency on data files: CSV parsing is replicated in multiple places (test duplication) — factor out utility to avoid divergence.
17. Lack of reproducible seeding strategy across nested pricing calls (e.g., re-pricing with same engine but different perspective) except where user explicitly seeds.
18. No validation ensuring `bond_maturity > option_maturity` in callable bond context at construction time; enforced locally in LS option pricer only.
19. No handling for negative rates (possible in low-rate regimes) beyond model allowing them; business logic around exercise may need guardrails if rates drop drastically (deep ITM early).
20. Limited diagnostics persistence: exercise boundary only stores mean statistics; no ability to inspect regression coefficients afterwards.

## 5. Recommended Remediation Plan
Phase 1 (Correctness / Stability)
1. Fix `YieldCurve`: add `self.curve_date`; ensure `shift_curve` copies day_count and date; update tests if needed.
2. Correct `_discount_future_cashflows` time handling — discount each future cash flow from its actual future time back to current step using accumulated (or precomputed) integrals of r, or reuse Monte Carlo discount factors ratio D(0,t_future)/D(0,t_current).
3. Refactor LS discounting to use precomputed path discount factors and avoid recomputation via average rate heuristics.
4. Stabilize θ(t): either derive analytic θ from initial curve or use spline derivative (`CubicSpline.derivative()`), remove tiny finite-difference; unit test θ smoothness.
5. Fix callable regression logging block variable ordering; guard behind a method or proper flag.
6. Confirm and codify coupon-on-call convention (add parameter like `call_includes_accrued=True`).
7. Add warnings for unimplemented calibration; optionally raise `NotImplementedError` until implemented.

Phase 2 (Model / Methodology Enhancements)
8. Expand regression state: include bond price and/or additional basis terms; benchmark variance reduction.
9. Provide analytic continuation value approximations (Hull-White conditional expectations) to reduce regression noise.
10. Implement swaption calibration using market normal vols: objective on pricing errors of European swaptions -> calibrate (a, σ).
11. Replace average-rate discounting in callable backward induction with proper discount factor ratios; unify discount logic.
12. Introduce vectorized bond valuation over rates with broadcasting; optionally Numba JIT for hot loops.
13. Add reproducibility manager (central RNG or `np.random.Generator`) to MC engine to avoid global seeding side-effects.
14. Store regression coefficients and R² diagnostics per exercise date for post-trade analysis.

Phase 3 (Usability / Maintenance)
15. Factor out yield curve CSV parsing into a utility (shared between notebooks and tests).
16. Add docstrings and developer docs clarifying economic conventions (perspective, sign, coupon handling at call).
17. Provide scenario / sensitivity utilities (parallel shift, key-rate shift, volatility bump) leveraging `shift_curve`.
18. Add benchmarks and profiling harness (pytest marker) to track performance gains.
19. Add optional fast path using exact Hull-White bond price formulas for discounting all deterministic coupon legs (semi-analytic callable approximation as control variate).
20. Expand test coverage: regression coefficients existence, exercise boundary monotonicity over time (qualitative), negative rate scenarios.

## 6. Open Questions (Need Product / Domain Clarification)
1. Coupon inclusion at call: Should a call on a coupon date deliver coupon + principal (current implementation) or just principal? Any notice period modeling required?
2. Calibration scope: Are we targeting normal vols (Bachelier), Black vols, or direct swaption prices? Which tenors / expiries needed?
3. Accuracy target: Acceptable MC standard error relative to price (<1 bp, <5 bps)? Drives number of paths and whether variance reduction (antithetic, control variates) is warranted now.
4. Negative rates: Do we need floor mechanisms or is allowing negative short rates acceptable?
5. Day count / accrual precision: Is ACT/365 sufficient, or do we need 30/360, ACT/ACT for coupon accrual alignment with market conventions?
6. Callable style: Only par calls, or future premium step-down schedule required later?
7. Performance budget: Is adding Numba acceptable across deployment environments? Any restrictions on JIT compilation (e.g., hosted environment)?
8. Output diagnostics: What level of transparency needed for audit (store pathwise exercise flags, regression residuals)?
9. Future instruments: Step-up coupons, sinkable, putable bonds, Bermudan swaptions? Prioritize design extensibility accordingly.
10. API expectations: Will this evolve into a library consumed elsewhere (need stable public surface) or remain research prototype?

## 7. Immediate Next Steps (If Approved)
1. Implement minimal fixes (Issues 1,3,4) + unit tests for corrected discounting logic.
2. Add regression diagnostics storage (Issue 14) to aid validation of subsequent changes.
3. Add configuration flag for call coupon inclusion and test both branches.
4. Draft analytic θ(t) replacement leveraging spline derivative; benchmark vs old path statistics.
5. Prepare calibration module scaffold (data structures + placeholder objective) before full implementation.

## 8. Risk Assessment
- Pricing bias risk from current discount approximation may mis-rank exercise decisions (material for high coupon / long maturity). Severity: Medium.
- θ(t) numerical instability could introduce subtle drift mis-specification. Severity: Medium.
- Missing `curve_date` could trigger failures when `shift_curve` used externally. Severity: High (latent bug).
- Economic convention ambiguity (coupon at call) risks misinterpretation by users. Severity: Medium.

## 9. Suggested Additional Tests
- Assert `shift_curve` preserves date and rates shift.
- Compare LS American price convergence against increased steps and paths (2D grid) to detect discounting fix impact.
- Validate callable bond price monotonic in coupon (holding rate curve constant) and decreasing in spread.
- Regression quality: R² >= threshold or warn; coefficient stability across seeds.
- Calibration (when implemented): recovered (a, σ) within tolerance on synthetic market data.

## 10. Tooling / Dependencies Note
- `numba` is currently unused; deliberate decision needed before introducing JIT to core loops.
- `statsmodels` currently unused; potential future use for richer regression diagnostics; could remove if not planned.

## 11. Summary
Core architecture is clear and modular. Main corrective priority: repair curve object, rationalize discounting/continuation logic, and stabilize θ(t). Then enhance model realism (calibration) and performance. Clarification of economic conventions will ensure correct interpretation of results.

Please review Open Questions and confirm priorities for Phase 1 before implementation.
