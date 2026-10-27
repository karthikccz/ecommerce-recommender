import pandas as pd
from src.pipeline import generate, validate, evaluate


def test_small_pipeline_integrity_and_scenarios():
    data = generate(customers=60, products=40, sessions=300, orders=80, seed=20261022)
    checks = validate(data)
    assert all(checks.values()), [name for name, passed in checks.items() if not passed]
    assert len(data["Customer"]) == 60
    assert len(data["Product"]) == 40
    assert len(data["Session"]) == 300
    assert len(data["Order"]) == 80
    assert data["Product"].is_cold_start.any()
    assert data["Session"].is_bot.any()
    assert data["Category"].is_seasonal.any()
    assert set(data["Recommendation"].sku).isdisjoint(
        set(data["Product"].loc[data["Product"].is_cold_start, "sku"])
    )
    metrics = evaluate(data)
    assert metrics["evaluated_customers"] >= 0


def test_seed_reproducibility_for_core_tables():
    a = generate(customers=30, products=30, sessions=100, orders=25, seed=123)
    b = generate(customers=30, products=30, sessions=100, orders=25, seed=123)
    for table in ["Customer", "Product", "Session", "PageView", "Order", "OrderLine", "Recommendation"]:
        pd.testing.assert_frame_equal(a[table].reset_index(drop=True), b[table].reset_index(drop=True))
