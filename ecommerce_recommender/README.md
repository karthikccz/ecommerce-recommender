# Synthetic E-commerce Data + Item-Item Recommender

A deterministic synthetic data generator and explainable item-item collaborative-filtering baseline for the e-commerce coding assignment.

## Requirements

- Python 3.10+
- Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Run the Project

Run the starter configuration from the project root:

```powershell
python -m src.pipeline
```

Default configuration:

- Customers: 200
- Products: 100
- Sessions: 2,000
- Orders: 400
- Random seed: `20261022`

### Run Assignment Data Volumes

```powershell
python -m src.pipeline --full
```

This generates:

- 20,000 customers
- 10,000 products
- 500,000 sessions
- 100,000 orders

The full run may take time and require sufficient memory and disk space. Generated files are written to `output/` by default. Existing files in that directory may be replaced.

### Custom Configuration

```powershell
python -m src.pipeline --customers 500 --products 250 --sessions 5000 --orders 1000 --seed 20261022 --out output_custom
```

The same generator is used for both starter and full configurations.

## Generated Data

The pipeline generates the following entities:

- **Customer, Product, Category:** Master data
- **Session, PageView:** Customer browsing activity
- **Cart, CartLine:** Shopping cart data
- **Order, OrderLine:** Purchase data
- **ProductRelation:** Product relationships learned from training-period co-purchases
- **Recommendation:** Ranked product recommendations

Each entity uses a unique primary key. Foreign keys connect related records, and relationships are validated to maintain data integrity. Bridge tables represent many-to-many relationships.

IDs are generated deterministically, for example:

- `CUS-000001`
- `SKU-000001`
- `SES-2026-000001`
- `ORD-2026-000001`

The data dictionary documents the entity fields, data types, required/optional fields, and validation rules.

## Data Generation Scenarios

The generator includes the following scenarios:

- **Cold-start products:** 5% of the catalog, when the catalog contains at least 20 products, is marked as cold-start and excluded from generated views and purchases.
- **Seasonality:** The `Seasonal Gifts` category receives a higher selection probability during Q4.
- **Sparse customers:** Customer sessions follow a long-tail distribution, so some customers have few or no sessions.
- **Bots and noise:** 4% of sessions are marked as bot sessions. These sessions may contain longer browsing activity and noise events.

Generated page views use products that are active and available at the event timestamp.

## Data Validation and Edge Cases

The pipeline validates:

- Primary-key uniqueness and foreign-key relationships
- Required fields and allowed values
- Event and order chronology
- Product lifecycle and availability
- Positive quantities and matching order totals
- Duplicate records and train/test separation

Labeled anomaly examples are provided in `anomalies.csv`. Ground-truth recommendation relevance labels are stored separately in `ground_truth.csv`.

The anomaly examples include:

- Purchase before a product view
- Missing product category
- Test-period information leaking into training
- Recommendation of a discontinued product

These examples are used to demonstrate data-quality and leakage checks.

## Recommender and Evaluation

The project uses **item-item collaborative filtering based on co-purchases**.

The model is trained using historical orders only. Products purchased together in an order receive a co-purchase count.

For each customer:

1. Use their historical purchases.
2. Find related products using co-purchase counts.
3. Score candidate products.
4. Exclude products already purchased in training.
5. Exclude products that are inactive or unavailable at the fixed cutoff (`2026-06-01`).
6. Rank candidates and return up to five recommendations.

Orders are split chronologically at `2026-06-01`.

- Earlier orders are used for training.
- Orders on or after the cutoff are used for evaluation.

Recommendation quality is measured using:

- Precision@5
- Recall@5
- NDCG@5

This is an explainable baseline, not a neural network or matrix-factorization model. Customers without purchase history may not receive personalized recommendations in this version.

## Tests

Run tests from the project root:

```powershell
python -m pytest -v
```

The tests check data integrity, scenario generation, and reproducibility for a fixed seed.

The validation summary reports row counts, integrity checks, and recommendation metrics.

## Output Files

The pipeline writes CSV and JSON files to the configured output directory, including:

- Generated entity tables
- `ground_truth.csv`
- `anomalies.csv`
- `validation_summary.json`
- `source_manifest.json`

The source manifest records SHA-256 hashes for exported CSV files.

## Limitations

- The dataset is synthetic; product names are generated placeholders.
- Seasonality is represented by a Q4 sampling boost, not a forecasting model.
- Anomaly examples are labeled fixtures and do not necessarily corrupt the valid baseline tables.
- The item-item recommender is a baseline, and its performance depends on the generated interaction data.
