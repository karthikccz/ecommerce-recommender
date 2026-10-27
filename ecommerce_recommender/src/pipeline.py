"""Reproducible synthetic e-commerce data and an explainable item-item recommender.

Starter: python -m src.pipeline
Full baseline: python -m src.pipeline --full
Custom scale: python -m src.pipeline --customers 500 --products 250 --sessions 5000 --orders 1000
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 20261022
OUT = Path("output")
CUTOFF = pd.Timestamp("2026-06-01")


def make_id(prefix, n, width=6):
    return f"{prefix}-{int(n):0{width}d}"


def _product_probabilities(products_df, month=None):
    """Product popularity with an explicit boost for seasonal-category items in Q4."""
    w = products_df["base_popularity"].to_numpy(dtype=float)
    if month in (10, 11, 12):
        w = w * np.where(products_df["is_seasonal"].to_numpy(), 5.0, 1.0)
    w = w / w.sum()
    return w


def generate(customers=200, products=100, sessions=2000, orders=400, seed=SEED):
    if min(customers, products, sessions, orders) < 1:
        raise ValueError("All scale parameters must be positive integers")
    if orders > sessions:
        raise ValueError("orders cannot exceed sessions in this generator")
    rng = np.random.default_rng(seed)

    categories = pd.DataFrame([
        {"category_id":"CAT-001","category_name":"Electronics","parent_category_id":None,"is_seasonal":False},
        {"category_id":"CAT-002","category_name":"Home","parent_category_id":None,"is_seasonal":False},
        {"category_id":"CAT-003","category_name":"Fashion","parent_category_id":None,"is_seasonal":False},
        {"category_id":"CAT-004","category_name":"Computers","parent_category_id":"CAT-001","is_seasonal":False},
        {"category_id":"CAT-005","category_name":"Accessories","parent_category_id":"CAT-001","is_seasonal":False},
        {"category_id":"CAT-006","category_name":"Kitchen","parent_category_id":"CAT-002","is_seasonal":False},
        {"category_id":"CAT-007","category_name":"Clothing","parent_category_id":"CAT-003","is_seasonal":False},
        {"category_id":"CAT-008","category_name":"Seasonal Gifts","parent_category_id":"CAT-002","is_seasonal":True},
    ])
    customers_df = pd.DataFrame({
        "customer_id":[make_id("CUS", i) for i in range(1, customers+1)],
        "created_at":pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 500, customers), unit="D"),
        "customer_segment":rng.choice(["new","regular","high_value"], customers, p=[.35,.50,.15]),
    })

    category_ids = categories.category_id.to_numpy()
    category_weights = np.array([.18,.15,.13,.18,.13,.10,.08,.05])
    p_cat = rng.choice(category_ids, products, p=category_weights/category_weights.sum())
    seasonal_cat = p_cat == "CAT-008"
    # Ensure a small, explicit catalog cold-start cohort with no generated interactions.
    cold_n = max(1, int(products * .05)) if products >= 2 else 0
    cold_mask = np.zeros(products, dtype=bool)
    if cold_n:
        cold_mask[-cold_n:] = True
    prod = pd.DataFrame({
        "sku":[make_id("SKU", i) for i in range(1, products+1)],
        "product_name":[f"{categories.set_index('category_id').loc[c, 'category_name']} Item {i:05d}" for i,c in enumerate(p_cat,1)],
        "category_id":p_cat,
        "price":np.round(rng.lognormal(3.5,.7,products),2),
        "status":rng.choice(["active","discontinued"], products, p=[.94,.06]),
        "available_from":pd.Timestamp("2024-10-01") + pd.to_timedelta(rng.integers(0, 300, products), unit="D"),
        "available_to":pd.NaT,
        "is_seasonal":seasonal_cat,
        "is_cold_start":cold_mask,
        "base_popularity":rng.pareto(1.8, products)+1.0,
    })
    # Cold-start items are active catalog items but are deliberately never selected for events/orders.
    prod.loc[cold_mask, "status"] = "active"
    interaction_pool = np.flatnonzero(~cold_mask)
    pool_df = prod.iloc[interaction_pool].reset_index(drop=True)

    # Long-tail customer activity: power-law-like weights produce sparse users and a few frequent users.
    customer_weights = 1 / np.arange(1, customers+1, dtype=float)**1.15
    customer_weights /= customer_weights.sum()
    starts = (pd.Timestamp("2025-01-01")
              + pd.to_timedelta(rng.integers(0, 600, sessions), unit="D")
              + pd.to_timedelta(rng.integers(0, 86400, sessions), unit="s"))
    session_customers = rng.choice(customers_df.customer_id.to_numpy(), sessions, p=customer_weights)
    bot_flags = rng.random(sessions) < .04
    sess = pd.DataFrame({
        "session_id":[f"SES-2026-{i:06d}" for i in range(1,sessions+1)],
        "customer_id":session_customers,
        "started_at":starts,
        "device":rng.choice(["web","mobile","tablet"], sessions, p=[.45,.48,.07]),
        "is_bot":bot_flags,
    }).sort_values("started_at").reset_index(drop=True)

    # Vectorized page-view generation; bots have noisier, longer browsing sessions.
    nviews = rng.integers(1, 6, sessions)
    sorted_bot_flags = sess.is_bot.to_numpy()
    nviews[sorted_bot_flags] = rng.integers(4, 11, int(sorted_bot_flags.sum()))
    session_idx = np.repeat(np.arange(sessions), nviews)
    within = np.concatenate([np.arange(1, n+1) for n in nviews])
    event_times = sess.started_at.to_numpy()[session_idx] + pd.to_timedelta(rng.integers(0, 1800, len(session_idx)), unit="s")
    event_month = pd.DatetimeIndex(event_times).month
    sampled_pool_positions = np.empty(len(session_idx), dtype=int)
    # Month-group sampling gives seasonal category demand a real, reproducible effect.
    # Only recommend/view products that are active and within their availability window
    # at the actual event timestamp. Keep the cold-start cohort out of interactions.
    pool_available_from = pool_df.available_from.to_numpy(dtype="datetime64[ns]")
    pool_available_to = pool_df.available_to.to_numpy(dtype="datetime64[ns]")
    pool_active = pool_df.status.eq("active").to_numpy()
    for month in np.unique(event_month):
        ix = np.flatnonzero(event_month == month)
        weights = _product_probabilities(pool_df, int(month))
        for event_ix in ix:
            event_time = np.datetime64(event_times[event_ix], "ns")
            eligible = pool_active & (pool_available_from <= event_time) & (np.isnat(pool_available_to) | (pool_available_to >= event_time))
            if not eligible.any():
                # Product availability begins before the synthetic event timeline, so this
                # fallback should be rare; use active products with the earliest availability.
                eligible = pool_active & (pool_available_from == pool_available_from.min())
            event_weights = weights * eligible
            event_weights = event_weights / event_weights.sum()
            sampled_pool_positions[event_ix] = rng.choice(len(pool_df), p=event_weights)
    sampled_skus = pool_df.sku.to_numpy()[sampled_pool_positions]
    pageviews = pd.DataFrame({
        "event_id":[f"EVT-{sess.session_id.iloc[i]}-{int(j):03d}" for i,j in zip(session_idx, within)],
        "session_id":sess.session_id.to_numpy()[session_idx],
        "customer_id":sess.customer_id.to_numpy()[session_idx],
        "sku":sampled_skus,
        "event_type":np.where(sess.is_bot.to_numpy()[session_idx] & (rng.random(len(session_idx)) < .12), "noise", "view"),
        "event_at":event_times,
    }).sort_values("event_at").reset_index(drop=True)

    # Carts: exactly `orders` converted carts plus 20% abandoned carts (bounded by sessions).
    cart_count = min(sessions, max(orders, int(np.ceil(orders*1.2))))
    selected_session_indices = rng.choice(sessions, cart_count, replace=False)
    selected_sessions = sess.iloc[selected_session_indices].reset_index(drop=True)
    cart_rows=[]; cartline_rows=[]
    # Build lookup from session to viewed product SKUs once, avoiding repeated full-table scans.
    views_by_session = pageviews.groupby("session_id").sku.apply(list).to_dict()
    product_price = prod.set_index("sku").price.to_dict()
    product_status = prod.set_index("sku").status.to_dict()
    product_available = prod.set_index("sku").available_from.to_dict()
    for i, row in enumerate(selected_sessions.itertuples(index=False), 1):
        cart_id=f"CART-2026-{i:06d}"
        converted = i <= orders
        cart_rows.append({"cart_id":cart_id,"customer_id":row.customer_id,"session_id":row.session_id,
                          "created_at":row.started_at,"status":"converted" if converted else "abandoned"})
        choices = list(dict.fromkeys(views_by_session.get(row.session_id, [])))
        if not choices:
            choices = [str(rng.choice(pool_df.sku))]
        # Cart has up to 3 unique viewed products; order generation uses the converted subset.
        for j, sku in enumerate(choices[:3], 1):
            cartline_rows.append({"cart_line_id":f"CL-{i:06d}-{j:07d}","cart_id":cart_id,"sku":sku,
                                  "quantity":int(rng.integers(1,3)),"unit_price":float(product_price[sku])})
    carts=pd.DataFrame(cart_rows)
    cartlines=pd.DataFrame(cartline_rows)

    converted_carts=carts.iloc[:orders].copy()
    order_rows=[]; orderline_rows=[]
    cartline_by_cart=cartlines.groupby("cart_id")
    for i, c in enumerate(converted_carts.itertuples(index=False), 1):
        order_id=f"ORD-2026-{i:06d}"
        ordered_at=c.created_at + pd.Timedelta(minutes=5)
        order_rows.append({"order_id":order_id,"customer_id":c.customer_id,"session_id":c.session_id,
                           "cart_id":c.cart_id,"ordered_at":ordered_at,"status":"completed"})
        if c.cart_id in cartline_by_cart.groups:
            lines=cartline_by_cart.get_group(c.cart_id)
            for j, line in enumerate(lines.itertuples(index=False), 1):
                orderline_rows.append({"order_line_id":f"OL-{i:06d}-{j:03d}","order_id":order_id,"sku":line.sku,
                                       "quantity":line.quantity,"unit_price":line.unit_price,
                                       "line_total":round(line.quantity*line.unit_price,2)})
    orders_df=pd.DataFrame(order_rows)
    orderlines=pd.DataFrame(orderline_rows)
    totals=orderlines.groupby("order_id").line_total.sum()
    orders_df["order_total"]=orders_df.order_id.map(totals).fillna(0).round(2)

    # Chronological split and item-item collaborative filtering from TRAIN orders only.
    train_orders=orders_df[orders_df.ordered_at < CUTOFF].copy()
    test_orders=orders_df[orders_df.ordered_at >= CUTOFF].copy()
    train_lines=orderlines[orderlines.order_id.isin(train_orders.order_id)]
    co={}
    for _, group in train_lines.groupby("order_id", sort=False):
        items=sorted(set(group.sku))
        for x in items:
            targets=co.setdefault(x,{})
            for y in items:
                if x != y:
                    targets[y]=targets.get(y,0)+1
    cust_hist={}
    order_customer=train_orders.set_index("order_id").customer_id.to_dict()
    for oid, group in train_lines.groupby("order_id", sort=False):
        cid=order_customer[oid]
        cust_hist.setdefault(cid,set()).update(group.sku)
    cutoff_available=set(prod.loc[(prod.status=="active") & (prod.available_from<=CUTOFF) &
                                  (prod.available_to.isna() | (prod.available_to>=CUTOFF)),"sku"])
    rec_rows=[]
    for cid in customers_df.customer_id:
        hist=cust_hist.get(cid,set())
        scores={}
        for item in hist:
            for candidate, score in co.get(item,{}).items():
                if candidate not in hist and candidate in cutoff_available:
                    scores[candidate]=scores.get(candidate,0)+score
        ranked=sorted(scores.items(),key=lambda x:(-x[1],x[0]))[:5]
        for rank,(sku,score) in enumerate(ranked,1):
            rec_rows.append({"recommendation_id":f"REC-{len(rec_rows)+1:08d}","customer_id":cid,"sku":sku,
                             "rank":rank,"score":float(score),"generated_at":CUTOFF,"model_version":"item_item_v2"})
    recs=pd.DataFrame(rec_rows,columns=["recommendation_id","customer_id","sku","rank","score","generated_at","model_version"])

    # Product relations are derived only from training purchase records.
    pair_counts={}
    for _, group in train_lines.groupby("order_id", sort=False):
        items=sorted(set(group.sku))
        for a in range(len(items)):
            for b in range(a+1,len(items)):
                pair=(items[a],items[b]); pair_counts[pair]=pair_counts.get(pair,0)+1
    relations=pd.DataFrame([{"relation_id":f"REL-{i:07d}","source_sku":a,"target_sku":b,
                             "relation_type":"co_purchase","strength":n}
                            for i,((a,b),n) in enumerate(sorted(pair_counts.items()),1)])

    truth=pd.DataFrame([
        {"case_id":"ANOM-001","entity":"PageView","issue_type":"purchase_before_view","ground_truth":"invalid_temporal_order"},
        {"case_id":"ANOM-002","entity":"Product","issue_type":"missing_category","ground_truth":"category_required"},
        {"case_id":"ANOM-003","entity":"PageView","issue_type":"test_event_in_train","ground_truth":"split_leakage"},
        {"case_id":"ANOM-004","entity":"Recommendation","issue_type":"discontinued_recommendation","ground_truth":"must_exclude"},
    ])
    anomalies=pd.DataFrame([
        {"case_id":"ANOM-001","event_at":"2026-07-01T10:00:00","session_view_at":"2026-07-01T10:05:00","issue_type":"purchase_before_view"},
        {"case_id":"ANOM-002","sku":"SKU-ANOM-000001","category_id":None,"issue_type":"missing_category"},
        {"case_id":"ANOM-003","event_split":"train","event_at":"2026-08-01T00:00:00","cutoff":"2026-06-01T00:00:00","issue_type":"test_event_in_train"},
        {"case_id":"ANOM-004","sku":str(prod.loc[prod.status=="discontinued","sku"].iloc[0]) if (prod.status=="discontinued").any() else "SKU-ANOM-000002","issue_type":"discontinued_recommendation"},
    ])
    return {"Customer":customers_df,"Product":prod,"Category":categories,"Session":sess,"PageView":pageviews,
            "Cart":carts,"CartLine":cartlines,"Order":orders_df,"OrderLine":orderlines,
            "ProductRelation":relations,"Recommendation":recs,"ground_truth":truth,"anomalies":anomalies,
            "train_orders":train_orders,"test_orders":test_orders,"train_lines":train_lines}


def validate(data):
    checks={}
    keys=[("Customer","customer_id"),("Product","sku"),("Category","category_id"),("Session","session_id"),
          ("PageView","event_id"),("Cart","cart_id"),("CartLine","cart_line_id"),("Order","order_id"),
          ("OrderLine","order_line_id"),("ProductRelation","relation_id"),("Recommendation","recommendation_id")]
    for table,key in keys:
        checks[f"{table}.pk_unique"]=bool(data[table][key].is_unique)
    def fk(name, child, col, parent, key):
        checks[name]=set(data[child][col].dropna()) <= set(data[parent][key].dropna())
    fk("Session.customer_fk","Session","customer_id","Customer","customer_id")
    fk("Product.category_fk","Product","category_id","Category","category_id")
    fk("PageView.session_fk","PageView","session_id","Session","session_id")
    fk("PageView.customer_fk","PageView","customer_id","Customer","customer_id")
    fk("PageView.product_fk","PageView","sku","Product","sku")
    fk("Cart.customer_fk","Cart","customer_id","Customer","customer_id")
    fk("Cart.session_fk","Cart","session_id","Session","session_id")
    fk("CartLine.cart_fk","CartLine","cart_id","Cart","cart_id")
    fk("CartLine.product_fk","CartLine","sku","Product","sku")
    fk("Order.customer_fk","Order","customer_id","Customer","customer_id")
    fk("Order.session_fk","Order","session_id","Session","session_id")
    fk("Order.cart_fk","Order","cart_id","Cart","cart_id")
    fk("OrderLine.order_fk","OrderLine","order_id","Order","order_id")
    fk("OrderLine.product_fk","OrderLine","sku","Product","sku")
    fk("ProductRelation.source_fk","ProductRelation","source_sku","Product","sku")
    fk("ProductRelation.target_fk","ProductRelation","target_sku","Product","sku")
    fk("Recommendation.product_fk","Recommendation","sku","Product","sku")
    fk("Recommendation.customer_fk","Recommendation","customer_id","Customer","customer_id")
    checks["enum.product_status"]=set(data["Product"].status)<= {"active","discontinued"}
    checks["enum.cart_status"]=set(data["Cart"].status)<= {"converted","abandoned"}
    checks["enum.order_status"]=set(data["Order"].status)<= {"completed","cancelled","refunded"}
    checks["PageView.event_timestamp_after_session"]=bool((data["PageView"].merge(data["Session"][ ["session_id","started_at"]],on="session_id").event_at >=
                                                               data["PageView"].merge(data["Session"][ ["session_id","started_at"]],on="session_id").started_at).all())
    sums=data["OrderLine"].groupby("order_id").line_total.sum()
    expected=data["Order"].set_index("order_id").order_total
    checks["Order.total_reconciles"]=bool(np.allclose(expected.sort_index(),sums.reindex(expected.index).fillna(0).sort_index()))
    checks["split.chronological"]=bool(data["train_orders"].empty or data["test_orders"].empty or
        data["train_orders"].ordered_at.max() < data["test_orders"].ordered_at.min())
    cutoff_active=set(data["Product"].loc[(data["Product"].status=="active") & (data["Product"].available_from<=CUTOFF) &
                                           (data["Product"].available_to.isna() | (data["Product"].available_to>=CUTOFF)),"sku"])
    checks["Recommendation.available_at_cutoff"]=set(data["Recommendation"].sku)<=cutoff_active
    checks["Recommendation.rank_unique_per_customer"]=bool(not data["Recommendation"].duplicated(["customer_id","rank"]).any())
    train_pair=set(zip(data["OrderLine"].loc[data["OrderLine"].order_id.isin(data["train_orders"].order_id),"order_id"],
                       data["OrderLine"].loc[data["OrderLine"].order_id.isin(data["train_orders"].order_id),"sku"]))
    cust_purchased={}
    train_order_customer=data["train_orders"].set_index("order_id").customer_id.to_dict()
    for oid, sku in train_pair:
        cust_purchased.setdefault(train_order_customer[oid],set()).add(sku)
    checks["Recommendation.not_previously_purchased"]=all(
        row.sku not in cust_purchased.get(row.customer_id,set()) for row in data["Recommendation"].itertuples(index=False))
    # Product lifecycle and temporal integrity for generated (non-anomaly) records.
    product_lookup = data["Product"].set_index("sku")
    view_products = data["PageView"].join(product_lookup[["status", "available_from", "available_to"]], on="sku")
    checks["PageView.product_active_at_event"] = bool((
        (view_products.status == "active") &
        (view_products.available_from <= view_products.event_at) &
        (view_products.available_to.isna() | (view_products.available_to >= view_products.event_at))
    ).all())
    checks["CartLine.quantity_positive"] = bool((data["CartLine"].quantity > 0).all())
    checks["OrderLine.quantity_positive"] = bool((data["OrderLine"].quantity > 0).all())
    checks["Order.uses_converted_cart"] = bool(data["Order"].cart_id.isin(
        data["Cart"].loc[data["Cart"].status == "converted", "cart_id"]
    ).all())
    checks["Anomalies.labels_unique"] = bool(data["anomalies"].case_id.is_unique and data["ground_truth"].case_id.is_unique)
    checks["Anomalies.truth_cases_match"] = set(data["anomalies"].case_id) == set(data["ground_truth"].case_id)
    checks["ColdStart.products_present"] = bool(data["Product"].is_cold_start.any())
    cold_skus=set(data["Product"].loc[data["Product"].is_cold_start,"sku"])
    interacted=set(data["PageView"].sku) | set(data["OrderLine"].sku)
    checks["ColdStart.no_interactions"]=not bool(cold_skus & interacted)
    checks["Seasonal.category_present"]=bool(data["Category"].is_seasonal.any())
    checks["Bots.present"]=bool(data["Session"].is_bot.any())
    checks["Sparse.users_present"]=bool((data["Session"].groupby("customer_id").size()<=2).any())
    return checks


def evaluate(data, k=5):
    test_lines=data["OrderLine"][data["OrderLine"].order_id.isin(data["test_orders"].order_id)]
    order_customer=data["test_orders"].set_index("order_id").customer_id.to_dict()
    relevant={}
    for oid,g in test_lines.groupby("order_id"):
        cid=order_customer.get(oid)
        if cid is not None: relevant.setdefault(cid,set()).update(g.sku)
    recs=data["Recommendation"]
    ps=[]; rs=[]; nd=[]
    for cid, truth in relevant.items():
        pred=recs.loc[recs.customer_id==cid].sort_values("rank").sku.head(k).tolist()
        hits=[int(s in truth) for s in pred]
        ps.append(sum(hits)/k)
        rs.append(sum(hits)/len(truth) if truth else 0)
        dcg=sum(h/np.log2(i+2) for i,h in enumerate(hits))
        ideal=sum(1/np.log2(i+2) for i in range(min(len(truth),k)))
        nd.append(dcg/ideal if ideal else 0)
    return {"precision_at_5":float(np.mean(ps)) if ps else 0.0,"recall_at_5":float(np.mean(rs)) if rs else 0.0,
            "ndcg_at_5":float(np.mean(nd)) if nd else 0.0,"evaluated_customers":len(relevant)}


def export(data, out=OUT, seed=SEED):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    counts={}
    for name,df in data.items():
        if name in ("train_orders","test_orders","train_lines"): continue
        df.to_csv(out/f"{name}.csv",index=False)
        counts[name]=len(df)
    metrics=evaluate(data); checks=validate(data)
    (out/"validation_summary.json").write_text(json.dumps({"seed":seed,"cutoff":str(CUTOFF),"row_counts":counts,"checks":checks,"metrics":metrics},indent=2))
    manifest=[]
    for path in sorted(out.glob("*.csv")):
        manifest.append({"file":path.name,"sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
    (out/"source_manifest.json").write_text(json.dumps(manifest,indent=2))
    return metrics,checks


def main():
    parser=argparse.ArgumentParser(description="Synthetic e-commerce data and item-item recommender")
    parser.add_argument("--customers",type=int,default=200); parser.add_argument("--products",type=int,default=100)
    parser.add_argument("--sessions",type=int,default=2000); parser.add_argument("--orders",type=int,default=400)
    parser.add_argument("--seed",type=int,default=SEED); parser.add_argument("--out",default="output")
    parser.add_argument("--full",action="store_true")
    args=parser.parse_args()
    if args.full: args.customers,args.products,args.sessions,args.orders=20000,10000,500000,100000
    data=generate(args.customers,args.products,args.sessions,args.orders,args.seed)
    metrics,checks=export(data,args.out,args.seed)
    print(json.dumps({"metrics":metrics,"checks":checks},indent=2))
    if not all(checks.values()): raise SystemExit("Validation failed; inspect validation_summary.json")

if __name__=="__main__": main()
