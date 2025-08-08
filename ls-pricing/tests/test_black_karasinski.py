import numpy as np
from datetime import datetime
from ls_pricing.core.curves import YieldCurve
from ls_pricing.core.black_karasinski import BlackKarasinskiModel
from ls_pricing.engine.monte_carlo import MonteCarloEngine


def test_bk_simulation_positive_rates_and_shape():
    tenors = np.array([0.25, 0.5, 1.0, 2.0, 5.0, 10.0])
    rates = np.array([0.03, 0.032, 0.035, 0.038, 0.04, 0.042])
    yc = YieldCurve(tenors, rates, datetime(2024, 1, 1))

    bk = BlackKarasinskiModel(a=0.05, sigma=0.01, yield_curve=yc)
    mc = MonteCarloEngine(model=bk, n_paths=500, n_steps=20)

    T = 5.0
    paths = mc.generate_paths(T=T, seed=123)

    r = paths["rates"]
    t = paths["times"]
    df = paths["discount_factors"]

    # Shape checks
    assert r.shape[0] == 500
    assert r.shape[1] == len(t)

    # Positivity of rates (lognormal)
    assert np.all(r > 0)

    # DF monotonic non-increasing per path
    assert np.all(np.diff(df, axis=1) <= 1e-12)


def test_bk_option_pricing_pathwise_works():
    tenors = np.array([0.25, 0.5, 1.0, 2.0, 5.0, 10.0])
    rates = np.array([0.03, 0.032, 0.035, 0.038, 0.04, 0.042])
    yc = YieldCurve(tenors, rates, datetime(2024, 1, 1))

    bk = BlackKarasinskiModel(a=0.05, sigma=0.01, yield_curve=yc)
    mc = MonteCarloEngine(model=bk, n_paths=2000, n_steps=50)

    from ls_pricing.engine.longstaff_schwartz import LongstaffSchwartzEngine

    ls = LongstaffSchwartzEngine(mc)
    res_eur = ls.price_european_option(
        strike=0.98, option_maturity=1.0, bond_maturity=2.0, option_type="put"
    )
    res_amer = ls.price_american_option(
        strike=0.98, option_maturity=1.0, bond_maturity=2.0, option_type="put"
    )

    assert res_eur["price"] > 0
    assert res_amer["price"] >= res_eur["price"] - 1e-6
