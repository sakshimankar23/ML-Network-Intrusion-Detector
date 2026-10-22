"""
CyberShield NIDS - Explainable ML Network Intrusion Detection Dashboard
IBM SkillsBuild Cybersecurity Bootcamp 2026 (BharatCares)

Run with:  streamlit run app.py
Needs model.pkl and columns.pkl (created by the notebook) next to this file.
Optional: data/KDDTest+.txt (demo traffic), images/ and results/ folders (Model Performance tab).
"""
import os
import time
from datetime import datetime

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

AUTHOR = "Your Name"          # <- change this to your name

st.set_page_config(page_title="CyberShield NIDS", page_icon="🛡️", layout="wide",
                   initial_sidebar_state="expanded")

# ----------------------------------------------------------------------------- styling
st.markdown("""
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
.hero {background: linear-gradient(120deg, #0f172a 0%, #1e3a8a 55%, #7f1d1d 100%);
       padding: 26px 32px; border-radius: 16px; margin-bottom: 18px;}
.hero h1 {margin: 0; font-size: 2.1rem; color: #ffffff;}
.hero p {margin: 6px 0 0 0; color: #e2e8f0; font-size: 1rem;}
.kpi {background: rgba(128,128,128,0.10); border-left: 5px solid #3b82f6;
      border-radius: 12px; padding: 14px 16px; margin-bottom: 8px;}
.kpi-label {font-size: .78rem; text-transform: uppercase; letter-spacing: .06em; opacity: .7;}
.kpi-value {font-size: 1.9rem; font-weight: 700; line-height: 1.25;}
.kpi-sub {font-size: .78rem; opacity: .65;}
.verdict {border-radius: 14px; padding: 18px 22px; margin: 8px 0 14px 0; border: 1px solid rgba(128,128,128,.3);}
.verdict h3 {margin: 0 0 6px 0;}
.pill {display: inline-block; padding: 2px 12px; border-radius: 999px; font-size: .8rem;
       font-weight: 700; color: #fff; margin-right: 6px;}
.section-title {font-size: 1.15rem; font-weight: 700; margin: 10px 0 4px 0;}
</style>
""", unsafe_allow_html=True)

SEV_ORDER = ["Low", "Medium", "High", "Critical"]
SEV_COLORS = {"Low": "#22c55e", "Medium": "#f59e0b", "High": "#f97316", "Critical": "#ef4444"}
SEV_ACTIONS = {
    "Low": "No action. Log to traffic database.",
    "Medium": "Monitor closely and flag for review in the SIEM.",
    "High": "Alert the SOC and rate-limit the source.",
    "Critical": "Block the source IP and isolate the target host.",
}
DEMO_PATHS = ["data/KDDTest+.txt", "KDDTest+.txt"]


# ----------------------------------------------------------------------------- artifacts
@st.cache_resource
def load_artifacts():
    return joblib.load("model.pkl"), joblib.load("columns.pkl")

try:
    model, meta = load_artifacts()
except Exception as e:
    st.error("Could not load model.pkl / columns.pkl. Put both files next to app.py. "
             "If the message mentions scikit-learn or an attribute error, run this app in the same "
             "environment (same library versions) where the model was trained, for example Google Colab.\n\n"
             f"Details: {e}")
    st.stop()

FEATURES = meta["raw_features"]
CHAMPION = meta.get("champion_name", "Tree ensemble")


# ----------------------------------------------------------------------------- helpers
def read_traffic(source):
    """Read NSL-KDD style traffic with 41, 42 or 43 columns, with or without a header row."""
    df = pd.read_csv(source, header=None)
    if str(df.iloc[0, 0]).strip().lower() == "duration":          # file already has a header row
        if hasattr(source, "seek"):
            source.seek(0)
        return pd.read_csv(source)
    all_cols = FEATURES + ["label", "difficulty"]
    if df.shape[1] == len(all_cols):
        df.columns = all_cols
    elif df.shape[1] == len(FEATURES) + 1:
        df.columns = FEATURES + ["label"]
    elif df.shape[1] == len(FEATURES):
        df.columns = FEATURES
    else:
        raise ValueError(f"Expected {len(FEATURES)} to {len(all_cols)} columns, found {df.shape[1]}.")
    return df


@st.cache_data(show_spinner=False)
def load_demo(path):
    return read_traffic(path)


@st.cache_data(show_spinner="Scoring connections ...")
def compute_probs(features_df):
    return model.predict_proba(features_df)[:, 1]


def severity_array(probs, thr):
    return np.select([probs < thr, probs < 0.70, probs < 0.90], ["Low", "Medium", "High"], default="Critical")


def build_results(data, thr):
    probs = compute_probs(data[FEATURES])
    keep = ["protocol_type", "service", "flag", "src_bytes", "dst_bytes"]
    if "label" in data.columns:
        keep.append("label")
    res = data[keep].copy()
    res.insert(0, "conn_id", data.index)
    res["attack_probability"] = probs
    res["verdict"] = np.where(probs >= thr, "Attack", "Normal")
    res["severity"] = severity_array(probs, thr)
    res["recommended_action"] = res["severity"].map(SEV_ACTIONS)
    return res


def style_table(df):
    def color(v):
        return f"background-color: {SEV_COLORS[v]}44; font-weight: 600" if v in SEV_COLORS else ""
    sty = df.style.format({"attack_probability": "{:.2%}"})
    apply_fn = sty.map if hasattr(sty, "map") else sty.applymap
    return apply_fn(color, subset=["severity"])


def kpi(col, label, value, color="#3b82f6", sub=""):
    col.markdown(f'<div class="kpi" style="border-left-color:{color}">'
                 f'<div class="kpi-label">{label}</div><div class="kpi-value">{value}</div>'
                 f'<div class="kpi-sub">{sub}</div></div>', unsafe_allow_html=True)


@st.cache_resource
def get_explainer():
    return shap.TreeExplainer(model.named_steps["clf"])


def explain_row(row_df, top_n=10):
    """SHAP values of one raw connection for the 'Attack' class."""
    prep = model.named_steps["prep"]
    X = prep.transform(row_df[FEATURES])
    names = [n.split("__", 1)[1] for n in prep.get_feature_names_out()]
    sv = get_explainer().shap_values(X)
    if isinstance(sv, list):
        sv = np.array(sv[1])
    elif np.ndim(sv) == 3:
        sv = sv[:, :, 1]
    else:
        sv = np.array(sv)
    out = pd.DataFrame({"feature": names, "impact": sv[0]})
    out["abs"] = out["impact"].abs()
    out = out.sort_values("abs", ascending=False).head(top_n).drop(columns="abs")
    out["direction"] = np.where(out["impact"] > 0, "Pushes toward ATTACK", "Pushes toward NORMAL")
    return out.reset_index(drop=True)


def explain_chart(exp_df):
    return (alt.Chart(exp_df).mark_bar().encode(
        x=alt.X("impact:Q", title="SHAP impact on the attack score"),
        y=alt.Y("feature:N", sort=list(exp_df["feature"]), title=None),
        color=alt.Color("direction:N",
                        scale=alt.Scale(domain=["Pushes toward ATTACK", "Pushes toward NORMAL"],
                                        range=["#ef4444", "#3b82f6"]),
                        legend=alt.Legend(title=None, orient="bottom")),
        tooltip=["feature", alt.Tooltip("impact:Q", format=".4f"), "direction"],
    ).properties(height=320))


def show_explanation(row_df):
    if not HAS_SHAP:
        st.info("Install the `shap` package to see explanations: `pip install shap`.")
        return
    try:
        exp = explain_row(row_df)
    except Exception as e:
        st.warning(f"Could not compute the SHAP explanation: {e}")
        return
    st.altair_chart(explain_chart(exp))
    top = exp.head(3)
    reasons = ", ".join(f"`{r.feature}` ({'raises' if r.impact > 0 else 'lowers'} the attack score)"
                        for r in top.itertuples())
    st.markdown(f"**In plain words:** the three biggest factors for this connection were {reasons}.")


def verdict_card(prob, thr):
    is_attack = prob >= thr
    sev = severity_array(np.array([prob]), thr)[0]
    color = SEV_COLORS[sev]
    title = "🚨 ATTACK DETECTED" if is_attack else "✅ NORMAL TRAFFIC"
    st.markdown(f'<div class="verdict" style="border-left: 6px solid {color}">'
                f'<h3>{title}</h3><span class="pill" style="background:{color}">{sev}</span>'
                f'<b>{prob * 100:.2f}%</b> attack probability<br>'
                f'<span style="opacity:.8">Recommended action: {SEV_ACTIONS[sev]}</span></div>',
                unsafe_allow_html=True)


# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("## 🛡️ CyberShield NIDS")
    st.caption("Explainable ML intrusion detection")
    source = st.radio("Traffic source", ["Upload a file", "Use demo data (KDDTest+)"])
    uploaded = None
    if source == "Upload a file":
        uploaded = st.file_uploader("NSL-KDD file (.txt / .csv)", type=["txt", "csv"])
    thr = st.slider("Detection threshold", 0.05, 0.95, 0.50, 0.05,
                    help="Lower = catch more attacks but raise more false alarms (see the threshold analysis in the notebook).")
    st.markdown("**Severity levels**")
    for s in SEV_ORDER:
        st.markdown(f'<span class="pill" style="background:{SEV_COLORS[s]}">{s}</span> {SEV_ACTIONS[s]}',
                    unsafe_allow_html=True)
    st.divider()
    st.caption(f"Model: **{CHAMPION}**")
    st.caption("Dataset: NSL-KDD")
    st.caption(f"Explainability (SHAP): {'on' if HAS_SHAP else 'off'}")

# ----------------------------------------------------------------------------- load traffic
data, load_error = None, None
try:
    if source == "Upload a file":
        if uploaded is not None:
            data = read_traffic(uploaded)
    else:
        demo_path = next((p for p in DEMO_PATHS if os.path.exists(p)), None)
        if demo_path:
            data = load_demo(demo_path)
        else:
            load_error = "Demo file not found. Put KDDTest+.txt in a folder named `data` (or next to app.py)."
    if data is not None:
        missing = [c for c in FEATURES if c not in data.columns]
        if missing:
            load_error, data = f"The file is missing required columns: {missing[:5]} ...", None
        elif len(data) > 20000:
            st.sidebar.info("Large file: using the first 20,000 connections.")
            data = data.head(20000).reset_index(drop=True)
except Exception as e:
    load_error, data = f"Could not read the file: {e}", None

res = build_results(data, thr) if data is not None else None

# ----------------------------------------------------------------------------- header
st.markdown('<div class="hero"><h1>🛡️ CyberShield NIDS</h1>'
            '<p>Explainable machine-learning network intrusion detection with SOC-style alerting · '
            'IBM SkillsBuild Cybersecurity Bootcamp 2026</p></div>', unsafe_allow_html=True)
if load_error:
    st.error(load_error)

tab_dash, tab_xai, tab_live, tab_test, tab_perf, tab_about = st.tabs(
    ["📊 Dashboard", "🧠 Explainability", "📡 Live Simulation", "🧪 Test a Connection",
     "📈 Model Performance", "ℹ️ About"])

EMPTY_MSG = "Choose **Upload a file** or **Use demo data** in the sidebar to analyse traffic."

# ----------------------------------------------------------------------------- dashboard
with tab_dash:
    if res is None:
        st.info(EMPTY_MSG)
    else:
        attacks = res[res["verdict"] == "Attack"]
        c1, c2, c3, c4 = st.columns(4)
        kpi(c1, "Connections analysed", f"{len(res):,}", "#3b82f6")
        kpi(c2, "Attacks flagged", f"{len(attacks):,}", "#f97316", f"{len(attacks) / len(res) * 100:.1f}% of traffic")
        kpi(c3, "Critical alerts", f"{int((res['severity'] == 'Critical').sum()):,}", "#ef4444", "immediate response")
        kpi(c4, "Average attack score", f"{res['attack_probability'].mean() * 100:.1f}%", "#8b5cf6")

        left, right = st.columns(2)
        with left:
            st.markdown('<div class="section-title">Severity breakdown</div>', unsafe_allow_html=True)
            sev_counts = res["severity"].value_counts().reindex(SEV_ORDER).fillna(0).reset_index()
            sev_counts.columns = ["severity", "count"]
            donut = alt.Chart(sev_counts).mark_arc(innerRadius=60).encode(
                theta="count:Q",
                color=alt.Color("severity:N", sort=SEV_ORDER,
                                scale=alt.Scale(domain=SEV_ORDER, range=[SEV_COLORS[s] for s in SEV_ORDER])),
                tooltip=["severity", "count"]).properties(height=300)
            st.altair_chart(donut)
        with right:
            st.markdown('<div class="section-title">Attack score distribution</div>', unsafe_allow_html=True)
            hist = alt.Chart(res[["attack_probability"]]).mark_bar(color="#3b82f6").encode(
                x=alt.X("attack_probability:Q", bin=alt.Bin(maxbins=30), title="Attack probability"),
                y=alt.Y("count()", title="Connections")).properties(height=300)
            st.altair_chart(hist)

        left, right = st.columns(2)
        with left:
            st.markdown('<div class="section-title">Flagged attacks by protocol</div>', unsafe_allow_html=True)
            if len(attacks):
                proto = attacks["protocol_type"].value_counts().reset_index()
                proto.columns = ["protocol", "count"]
                st.altair_chart(alt.Chart(proto).mark_bar(color="#f97316").encode(
                    x=alt.X("protocol:N", sort="-y", title=None), y="count:Q",
                    tooltip=["protocol", "count"]).properties(height=260))
            else:
                st.info("No attacks flagged at this threshold.")
        with right:
            st.markdown('<div class="section-title">Most targeted services</div>', unsafe_allow_html=True)
            if len(attacks):
                svc = attacks["service"].value_counts().head(8).reset_index()
                svc.columns = ["service", "count"]
                st.altair_chart(alt.Chart(svc).mark_bar(color="#ef4444").encode(
                    y=alt.Y("service:N", sort="-x", title=None), x="count:Q",
                    tooltip=["service", "count"]).properties(height=260))

        if "label" in res.columns:
            st.markdown('<div class="section-title">Ground-truth check (file contains true labels)</div>',
                        unsafe_allow_html=True)
            truth = (res["label"] != "normal").values
            pred = (res["verdict"] == "Attack").values
            tp, fn = int((truth & pred).sum()), int((truth & ~pred).sum())
            fp, tn = int((~truth & pred).sum()), int((~truth & ~pred).sum())
            g1, g2, g3, g4 = st.columns(4)
            kpi(g1, "Attack recall", f"{tp / max(tp + fn, 1) * 100:.1f}%", "#22c55e", "real attacks caught")
            kpi(g2, "Precision", f"{tp / max(tp + fp, 1) * 100:.1f}%", "#3b82f6", "alerts that were real")
            kpi(g3, "False-alarm rate", f"{fp / max(fp + tn, 1) * 100:.1f}%", "#f59e0b", "normal flagged as attack")
            kpi(g4, "Missed attacks", f"{fn:,}", "#ef4444", "false negatives")
            missed = res[truth & ~pred]["label"].value_counts().head(8)
            if len(missed):
                st.caption("Most frequently missed attack types: " + ", ".join(f"{k} ({v})" for k, v in missed.items()))

        st.markdown('<div class="section-title">Alert list (highest risk first)</div>', unsafe_allow_html=True)
        chosen = st.multiselect("Filter by severity", SEV_ORDER[1:], default=SEV_ORDER[1:])
        alert_df = res[(res["verdict"] == "Attack") & (res["severity"].isin(chosen))] \
            .sort_values("attack_probability", ascending=False)
        show_cols = ["conn_id", "protocol_type", "service", "flag", "attack_probability", "severity", "recommended_action"]
        if "label" in alert_df.columns:
            show_cols.insert(4, "label")
        st.dataframe(style_table(alert_df[show_cols].head(200)))
        st.download_button("⬇️ Download alerts as CSV", alert_df.to_csv(index=False).encode(),
                           "nids_alerts.csv", "text/csv")

# ----------------------------------------------------------------------------- explainability
with tab_xai:
    st.markdown("### Why was this connection flagged?")
    st.caption("SHAP shows how much each network feature pushed the model toward 'attack' (red) or 'normal' (blue).")
    if res is None:
        st.info(EMPTY_MSG)
    else:
        top_alerts = res.sort_values("attack_probability", ascending=False).head(50)
        if top_alerts.empty:
            st.info("No connections to explain.")
        else:
            lookup = top_alerts.set_index("conn_id")
            cid = st.selectbox(
                "Choose one of the 50 highest-risk connections", list(lookup.index),
                format_func=lambda i: f"#{i} · {lookup.loc[i, 'attack_probability']:.1%} · "
                                      f"{lookup.loc[i, 'protocol_type']}/{lookup.loc[i, 'service']}")
            verdict_card(float(lookup.loc[cid, "attack_probability"]), thr)
            show_explanation(data.loc[[cid]])

# ----------------------------------------------------------------------------- live simulation
with tab_live:
    st.markdown("### Live traffic simulation")
    st.caption("Replays connections one by one, like a SOC console receiving a live feed.")
    if res is None:
        st.info(EMPTY_MSG)
    else:
        s1, s2 = st.columns(2)
        n_stream = s1.slider("Connections to stream", 10, 100, 30)
        delay = s2.slider("Delay between connections (seconds)", 0.05, 1.0, 0.25)
        if st.button("▶ Start simulation", type="primary"):
            sample = res.sample(min(n_stream, len(res)), random_state=int(time.time()) % 10000)
            m1, m2, m3 = st.columns(3)
            ph_total, ph_att, ph_crit = m1.empty(), m2.empty(), m3.empty()
            bar = st.progress(0)
            table_ph = st.empty()
            feed, n_att, n_crit = [], 0, 0
            for i, (_, row) in enumerate(sample.iterrows(), start=1):
                n_att += int(row["verdict"] == "Attack")
                n_crit += int(row["severity"] == "Critical")
                feed.insert(0, {"time": datetime.now().strftime("%H:%M:%S"), "conn_id": row["conn_id"],
                                "protocol_type": row["protocol_type"], "service": row["service"],
                                "attack_probability": row["attack_probability"], "severity": row["severity"],
                                "recommended_action": row["recommended_action"]})
                ph_total.metric("Processed", i)
                ph_att.metric("Attacks", n_att)
                ph_crit.metric("Critical", n_crit)
                bar.progress(i / len(sample))
                table_ph.dataframe(style_table(pd.DataFrame(feed[:12])))
                time.sleep(delay)
            st.success(f"Simulation finished: {n_att} attacks flagged, {n_crit} critical.")

# ----------------------------------------------------------------------------- manual test
PRESETS = {
    "Normal web browsing": dict(protocol_type="tcp", service="http", flag="SF", src_bytes=232, dst_bytes=8153,
                                logged_in=1, count=5, srv_count=5, same_srv_rate=1.0,
                                dst_host_count=30, dst_host_srv_count=255, dst_host_same_srv_rate=1.0),
    "SYN flood (DoS-like)": dict(protocol_type="tcp", service="private", flag="S0", src_bytes=0, dst_bytes=0,
                                 count=255, srv_count=20, serror_rate=1.0, srv_serror_rate=1.0,
                                 same_srv_rate=0.08, diff_srv_rate=0.06, dst_host_count=255,
                                 dst_host_srv_count=20, dst_host_same_srv_rate=0.08,
                                 dst_host_serror_rate=1.0, dst_host_srv_serror_rate=1.0),
    "Port scan (Probe-like)": dict(protocol_type="tcp", service="private", flag="REJ", src_bytes=0, dst_bytes=0,
                                   count=150, srv_count=8, rerror_rate=1.0, srv_rerror_rate=1.0,
                                   same_srv_rate=0.05, diff_srv_rate=0.9, dst_host_count=255,
                                   dst_host_srv_count=10, dst_host_same_srv_rate=0.04,
                                   dst_host_diff_srv_rate=0.07, dst_host_rerror_rate=1.0,
                                   dst_host_srv_rerror_rate=1.0),
}
FLAGS = ["SF", "S0", "REJ", "RSTR", "RSTO", "SH", "S1", "S2", "S3", "RSTOS0", "OTH"]
PROTOS = ["tcp", "udp", "icmp"]

with tab_test:
    st.markdown("### Test a single connection")
    st.caption("Pick a scenario, adjust the key fields, and analyse. Unlisted features default to 0. "
               "Presets are simplified examples, so results can differ from real NSL-KDD records.")
    preset = st.selectbox("Scenario", list(PRESETS))
    base = {f: 0 for f in FEATURES}
    base.update(protocol_type="tcp", service="http", flag="SF")
    base.update(PRESETS[preset])

    a, b, c = st.columns(3)
    base["protocol_type"] = a.selectbox("Protocol", PROTOS, index=PROTOS.index(base["protocol_type"]))
    base["service"] = b.text_input("Service", base["service"])
    base["flag"] = c.selectbox("Flag", FLAGS, index=FLAGS.index(base["flag"]) if base["flag"] in FLAGS else 0)
    d, e, f = st.columns(3)
    base["src_bytes"] = d.number_input("src_bytes", 0, 10**9, int(base["src_bytes"]))
    base["dst_bytes"] = e.number_input("dst_bytes", 0, 10**9, int(base["dst_bytes"]))
    base["count"] = f.number_input("count (connections to same host)", 0, 600, int(base["count"]))
    g, h, i = st.columns(3)
    base["serror_rate"] = g.slider("serror_rate (SYN errors)", 0.0, 1.0, float(base["serror_rate"]))
    base["rerror_rate"] = h.slider("rerror_rate (REJ errors)", 0.0, 1.0, float(base["rerror_rate"]))
    base["same_srv_rate"] = i.slider("same_srv_rate", 0.0, 1.0, float(base["same_srv_rate"]))

    if st.button("🔍 Analyse connection", type="primary"):
        row_df = pd.DataFrame([base])[FEATURES]
        p = float(model.predict_proba(row_df)[0][1])
        verdict_card(p, thr)
        st.progress(min(max(p, 0.0), 1.0))
        show_explanation(row_df)

# ----------------------------------------------------------------------------- model performance
with tab_perf:
    st.markdown("### Model performance (official KDDTest+ set)")
    tables = [("Test-set results", "results/test_set_results.csv"),
              ("Cross-validation results", "results/cross_validation_results.csv"),
              ("Threshold analysis (recall vs false alarms)", "results/threshold_analysis.csv"),
              ("Multi-class report", "results/multiclass_report.csv")]
    found_any = False
    for title, path in tables:
        if os.path.exists(path):
            found_any = True
            st.markdown(f"**{title}**")
            st.dataframe(pd.read_csv(path, index_col=0 if "multiclass" in path else None).round(4))
    figs = [("ROC curves", "images/fig4_roc_curves.png"), ("Metric comparison", "images/fig5_metric_comparison.png"),
            ("Confusion matrix", "images/fig6_confusion_matrix.png"), ("Precision-recall", "images/fig7_precision_recall.png"),
            ("Multi-class confusion matrix", "images/fig8_multiclass_confusion.png"),
            ("SHAP summary", "images/fig10_shap_summary.png"), ("SHAP local explanation", "images/fig11_shap_local.png"),
            ("Miss rate by attack category", "images/fig12_miss_rate.png")]
    present = [(t, p) for t, p in figs if os.path.exists(p)]
    if present:
        found_any = True
        cols = st.columns(2)
        for k, (t, p) in enumerate(present):
            cols[k % 2].markdown(f"**{t}**")
            cols[k % 2].image(p)
    if not found_any:
        st.info("Copy the `results/` and `images/` folders created by the notebook next to app.py to see them here.")

# ----------------------------------------------------------------------------- about
with tab_about:
    st.markdown(f"""
### About this project
**CyberShield NIDS** is an explainable machine-learning intrusion detection system built for the
IBM SkillsBuild Cybersecurity Bootcamp 2026 (BharatCares). Author: **{AUTHOR}**

**Pipeline**
1. NSL-KDD dataset: 41 features per connection, official train and test sets
2. Leak-free preprocessing (scaling and one-hot encoding) inside a scikit-learn pipeline
3. Four models compared with stratified cross-validation, then tuned: Logistic Regression, Decision Tree, Random Forest, XGBoost
4. Evaluation on the official test set, which contains attack types never seen in training
5. SHAP explanations for every alert
6. Severity levels and recommended actions, like a lightweight SOAR playbook

**Technologies:** Python, scikit-learn, XGBoost, SHAP, pandas, Altair, Streamlit

**Limitations:** NSL-KDD is an older benchmark and real networks are noisier. Rare attack classes (R2L, U2R)
and unseen attack types are the hardest. Treat this as an educational prototype, not a production IDS.
""")
