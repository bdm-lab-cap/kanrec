"""
KAN-REC Dashboard — interactive explorer for symbolic scoring results.

Usage:
    streamlit run dashboard/app.py
"""
import json
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import torch

from kanrec.model import KANRecModel
from kanrec.mongo_store import MongoSymbolicStore

st.set_page_config(page_title="KAN-REC Dashboard", layout="wide")
st.title("KAN-REC — Symbolic Scoring Explorer")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Experiment")
    dataset  = st.selectbox("Dataset", ["criteo", "avazu"])
    seeds    = st.multiselect("Seeds", [42, 123, 256, 512, 1024], default=[42, 123, 256])
    ckpt     = st.text_input("Checkpoint", "checkpoints/best_kan-bspline_criteo_s42.pt")
    load_btn = st.button("Load model")
    st.markdown("---")
    st.caption("MongoDB: localhost:27017 · kanrec.symbolic_results")

# ── Load model on button press ────────────────────────────────────────────────
if load_btn:
    try:
        with open("data/feature_selection.json") as f:
            sel = json.load(f)
        fields = sel["selected"]
        m = KANRecModel(num_numerical=len(fields), cat_cardinalities=[100]*26, embedding_dim=16)
        m.load_state_dict(torch.load(ckpt, map_location="cpu"))
        m.eval()
        st.session_state["model"]  = m
        st.session_state["fields"] = fields
        st.sidebar.success(f"Loaded ({len(fields)} numerical fields)")
    except Exception as e:
        st.sidebar.error(str(e))

tab1, tab2, tab3 = st.tabs(["Spline curves φ", "Symbolic formula", "MongoDB analysis"])

# ── Tab 1: Spline curves ──────────────────────────────────────────────────────
with tab1:
    st.subheader("Learned spline curves per numerical field")
    if "model" not in st.session_state:
        st.info("Load a model checkpoint from the sidebar.")
    else:
        model  = st.session_state["model"]
        fields = st.session_state["fields"]
        cols   = st.columns(3)
        for j, name in enumerate(fields):
            with cols[j % 3]:
                x, y = model.numerical_encoder.get_spline_curves(j, n_points=300)
                fig  = go.Figure(go.Scatter(
                    x=x.numpy(), y=y[:, 0].numpy(),
                    mode="lines", line=dict(color="#FF860D", width=2)
                ))
                fig.update_layout(title=f"φ_{name}(x)", height=200,
                                  margin=dict(l=10, r=10, t=35, b=10))
                st.plotly_chart(fig, use_container_width=True)

# ── Tab 2: Symbolic formula ───────────────────────────────────────────────────
with tab2:
    st.subheader("Extracted scoring formula")
    store = MongoSymbolicStore()
    if seeds:
        formula = store.scoring_formula_summary(dataset, seeds)
        st.code(formula, language="text")

        report = store.stability_report(dataset, seeds)
        if report:
            df = pd.DataFrame(report)
            df["stability"] = df["stability_pct"].map(lambda x: f"{x:.0%}")
            st.dataframe(
                df[["field", "operator", "seeds_count", "stability", "avg_r2", "avg_gap_rmse"]],
                use_container_width=True
            )
        else:
            st.info("No accepted results found in MongoDB. Run experiments/run_all.sh first.")
    store.close()

# ── Tab 3: MongoDB analysis ───────────────────────────────────────────────────
with tab3:
    st.subheader("Cross-dataset comparison & monotonicity audit")
    store = MongoSymbolicStore()
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Cross-dataset operator comparison")
        field_sel = st.selectbox("Field", [f"I{i}" for i in range(1, 14)])
        comp = store.cross_dataset_comparison(field_sel)
        if comp:
            df_c = pd.DataFrame([
                {"dataset": r["_id"]["dataset"], "operator": r["_id"]["op"],
                 "n_seeds": r["count"], "avg_r2": round(r["avg_r2"], 4)}
                for r in comp
            ])
            st.dataframe(df_c, use_container_width=True)
            ops = df_c["operator"].unique()
            if len(ops) > 1:
                st.warning(f"⚠ {field_sel} uses different operators across datasets — notable finding.")
            else:
                st.success(f"✓ {field_sel} is stable across datasets.")
        else:
            st.info("No results for this field.")

    with c2:
        st.markdown("#### Monotonicity audit")
        mono_docs = list(store.col.find(
            {"dataset": dataset, "seed": {"$in": seeds}, "is_accepted": True},
            {"field_name": 1, "is_monotone": 1, "operator": 1, "_id": 0}
        ))
        if mono_docs:
            df_m = pd.DataFrame(mono_docs)
            df_m["status"] = df_m["is_monotone"].map(
                lambda x: "✓ Monotone" if x else "⚠ Non-monotone"
            )
            st.dataframe(
                df_m.groupby(["field_name", "operator", "status"])
                    .size().reset_index(name="count"),
                use_container_width=True
            )
        else:
            st.info("No monotonicity data yet.")

    store.close()
