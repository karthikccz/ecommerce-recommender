# Data Dictionary

IDs are synthetic keys. `SKU` means Stock Keeping Unit. Times are local-naive synthetic timestamps; the recommendation evaluation cutoff is `2026-06-01`.

| File / entity | Main key | Purpose / important fields |
|---|---|---|
| Customer | `customer_id` | Customer master; `created_at`, `customer_segment` |
| Product | `sku` | Catalog master; `product_name`, `category_id`, `price`, `status`, `available_from`, `available_to`, `is_seasonal`, `is_cold_start`, `base_popularity` |
| Category | `category_id` | Category hierarchy; `category_name`, optional `parent_category_id`, `is_seasonal` |
| Session | `session_id` | Visit; `customer_id`, `started_at`, `device`, `is_bot` |
| PageView | `event_id` | Event; `session_id`, `customer_id`, `sku`, `event_type` (`view`/`noise`), `event_at` |
| Cart | `cart_id` | Shopping cart; `customer_id`, `session_id`, `created_at`, `status` (`converted`/`abandoned`) |
| CartLine | `cart_line_id` | Cart item; `cart_id`, `sku`, positive `quantity`, `unit_price` |
| Order | `order_id` | Purchase header; customer/session/cart keys, `ordered_at`, `status`, `order_total` |
| OrderLine | `order_line_id` | Purchased item; `order_id`, `sku`, `quantity`, `unit_price`, `line_total` |
| ProductRelation | `relation_id` | Product pair from training orders; `source_sku`, `target_sku`, `relation_type`, `strength` |
| Recommendation | `recommendation_id` | Result; `customer_id`, `sku`, `rank`, `score`, `generated_at`, `model_version` |
| ground_truth | `case_id` | Expected labels for anomaly fixtures |
| anomalies | `case_id` | Isolated synthetic anomaly examples, intentionally separate from valid data |

Foreign-key links: Customer→Session/Cart/Order/Recommendation; Category→Product; Session→PageView/Cart/Order; Cart→CartLine/Order; Product→PageView/CartLine/OrderLine/ProductRelation/Recommendation; Order→OrderLine.

`order_total` is the sum of `line_total`; `line_total = quantity * unit_price`. Quantities are positive integers and prices/totals are numeric currency values rounded to two decimal places. Product lifecycle is checked against each page-view timestamp; recommendation candidates must be active and available at the fixed training cutoff. The generator uses a fixed seed by default and emits a SHA-256 manifest for CSV outputs.

## Field-level conventions

- Required primary keys are non-null and unique in each entity table.
- Foreign keys are required for the links listed above; `Category.parent_category_id` and `Product.available_to` are optional.
- `Product.status`: `active` or `discontinued`; `Cart.status`: `converted` or `abandoned`; `Order.status`: `completed`, `cancelled`, or `refunded`; `PageView.event_type`: `view` or `noise`.
- `Customer.customer_segment`: `new`, `regular`, or `high_value`; `Session.device`: `web`, `mobile`, or `tablet`.
- `PageView.event_at` must be at or after `Session.started_at`; product `available_from`/`available_to` define the lifecycle window. Generated valid views use products that are active and within that window.
- `CartLine.quantity` and `OrderLine.quantity` are positive integers. `OrderLine.line_total` equals quantity times unit price; `Order.order_total` reconciles to its order lines.
- `Recommendation.rank` is a positive rank (at most five per customer in this baseline); recommendation SKUs must be active and available at the training cutoff and must not be present in that customer's training purchase history.
- Anomaly fixtures are intentionally isolated from valid tables and are linked to expected labels by the unique `case_id` values in `ground_truth.csv` and `anomalies.csv`.
