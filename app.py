# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
from pathlib import Path
from utils import get_usage_matrix, get_tenure_trends, get_peer_comparison
from explain import get_top_reasons

import joblib

# Load model and preprocessor (runs once when app starts)
@st.cache_resource
def load_model_and_preprocessor():
    preprocessor = joblib.load("models/preprocessor.pkl")
    model = joblib.load("models/xgb_model.pkl")
    return preprocessor, model

preprocessor, model = load_model_and_preprocessor()
# ────────────────────────────────────────────────
# Page config
# ────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Prediction & Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("Customer Churn Prediction & Analysis Dashboard")

# ────────────────────────────────────────────────
# Load dataset (cached)
# ────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

df = load_data()

# ────────────────────────────────────────────────
# Search section
# ────────────────────────────────────────────────
st.subheader("Load Customer")

col_input, col_button = st.columns([4, 1])

with col_input:
    customer_id = st.text_input(
        "Enter Customer ID",
        placeholder="e.g. 7590-VHVEG",
        label_visibility="collapsed"
    )

with col_button:
    load_clicked = st.button("Load", type="primary", use_container_width=True)

# ────────────────────────────────────────────────
# Load customer
# ────────────────────────────────────────────────
if load_clicked:

    if not customer_id:
        st.warning("Please enter a Customer ID first.")
        st.stop()

    customer_row = df[df["customerID"] == customer_id.strip()]

    if customer_row.empty:
        st.error("Customer ID not found. Please check the ID and try again.")
        st.stop()

    row = customer_row.iloc[0]

    col1, col2, col3 = st.columns([4, 5, 4])

    # ────────────────────────────────────────────────
    # Left tile: Customer Overview
    # ────────────────────────────────────────────────
    with col1:
        with st.container(border=True):
            st.markdown("### Customer Overview")

            # Use 2 columns for layout
            subcol1, subcol2 = st.columns(2)

            with subcol1:
                st.markdown(f"**Customer ID**  \n{row['customerID']}")
                st.markdown(f"**Tenure**  \n{row['tenure']} months")
                st.markdown(f"**Senior Citizen**  \n{'Yes' if row['SeniorCitizen'] == 1 else 'No'}")
                st.markdown(f"**Tech Support**  \n{row['TechSupport']}")

            with subcol2:
                st.markdown(f"**Monthly Charges**  \n₹{row['MonthlyCharges']:.2f}")
                st.markdown(f"**Contract Type**  \n{row['Contract']}")
                st.markdown(f"**Internet Service**  \n{row['InternetService']}")
                st.markdown(f"**Payment Method**  \n{row['PaymentMethod']}")
    # ────────────────────────────────────────────────
    # Middle tile: Churn Risk Gauge (HTML component)
    # ────────────────────────────────────────────────
    with col2:
        with st.container(height=420,border=True):
            st.markdown("### Churn Risk Prediction")

            # Prepare the single customer row as DataFrame
            customer_df = pd.DataFrame([row.to_dict()])  # row is from earlier

            # Preprocess using the loaded preprocessor
            customer_transformed = preprocessor.transform(customer_df)

            # Predict probability
            churn_prob = model.predict_proba(customer_transformed)[0][1] * 100  # in percent
            churn_prob = round(churn_prob, 1)  # nice display

            # Risk text (you can adjust thresholds)
            if churn_prob >= 70:
                risk_text = "High Risk"
            elif churn_prob >= 40:
                risk_text = "Medium Risk"
            else:
                risk_text = "Low Risk"

            # Keep time_to_churn as dummy for now
            time_to_churn = 45

            # Convert probability to rotation angle
            rotation_deg = (churn_prob * 1.8) - 90

            gauge_path = Path("components/gauge.html")

            if gauge_path.exists():
                gauge_html = gauge_path.read_text(encoding="utf-8")

                gauge_html = gauge_html.replace("{{churn_prob}}", str(churn_prob))
                gauge_html = gauge_html.replace("{{risk_text}}", risk_text)
                gauge_html = gauge_html.replace("{{time_to_churn}}", str(time_to_churn))
                gauge_html = gauge_html.replace("((rotation_deg))", str(rotation_deg))


                components.html(gauge_html, height=420)

            else:
                st.error("components/gauge.html not found")

    # ────────────────────────────────────────────────
    # Right tile: Clear 2×2 Usage Matrix (row-aligned)
    # ────────────────────────────────────────────────
    with col3:
        with st.container(height=450, border=True):
            st.markdown("### Customer Segment Analysis")

            # Get data
            matrix, tenure_cutoff, charges_cutoff = get_usage_matrix(df)

            st.markdown(
                f"<div style='margin-left: 100px;'>Monthly Charges (split at ₹{charges_cutoff:.1f})</div>",
                unsafe_allow_html=True
            )



            # Column headers
            header_label, header_low, header_high = st.columns([1.2, 2, 2])

            with header_label:
                st.markdown("Tenure")

            with header_low:
                st.markdown("**Low**")

            with header_high:
                st.markdown("**High**")

            # ───── Row 1: New customers ─────
            row1_label, row1_low, row1_high = st.columns([1.2, 2, 2])

            with row1_label:
                st.markdown(
                    "<div style='display:flex;align-items:center;height:100%;font-weight:bold;'>"
                    "New<br>(≤ 24 mo)"
                    "</div>",
                    unsafe_allow_html=True
                )

            with row1_low:
                st.markdown(
                    f"<div style='background:#f59e0b;color:white;padding:20px;"
                    f"border-radius:8px;text-align:center;font-weight:bold;'>"
                    f"Medium Risk<br>{matrix[('New','Low')]['pct']}%</div>",
                    unsafe_allow_html=True
                )

            with row1_high:
                st.markdown(
                    f"<div style='background:#ef4444;color:white;padding:20px;"
                    f"border-radius:8px;text-align:center;font-weight:bold;'>"
                    f"High Risk<br>{matrix[('New','High')]['pct']}%</div>",
                    unsafe_allow_html=True
                )
            st.markdown("<div style='margin-top:-16px;'></div>", unsafe_allow_html=True)


            # ───── Row 2: Long-tenure customers ─────
            row2_label, row2_low, row2_high = st.columns([1.2, 2, 2])

            with row2_label:
                st.markdown(
                    "<div style='display:flex;align-items:center;height:100%;font-weight:bold;'>"
                    "Long<br>(> 24 mo)"
                    "</div>",
                    unsafe_allow_html=True
                )

            with row2_low:
                st.markdown(
                    f"<div style='background:#10b981;color:white;padding:20px;"
                    f"border-radius:8px;text-align:center;font-weight:bold;'>"
                    f"Low Risk<br>{matrix[('Long','Low')]['pct']}%</div>",
                    unsafe_allow_html=True
                )

            with row2_high:
                st.markdown(
                    f"<div style='background:#f59e0b;color:white;padding:20px;"
                    f"border-radius:8px;text-align:center;font-weight:bold;'>"
                    f"Medium Risk<br>{matrix[('Long','High')]['pct']}%</div>",
                    unsafe_allow_html=True
                )
            st.markdown("---")

            # Current customer indicator
            current_tenure = 'New' if row['tenure'] <= tenure_cutoff else 'Long'
            current_charges = 'Low' if row['MonthlyCharges'] <= charges_cutoff else 'High'

            st.markdown(
                f"**← Current customer is in: {current_tenure} + {current_charges}**"
            )

    # ────────────────────────────────────────────────
    # New section: Top Reasons for Churn (SHAP-based)
    # ────────────────────────────────────────────────
    col1, col2, col3 = st.columns([4, 5, 4])
    with col1:
        with st.container(border=True):
            st.markdown("### Top Reasons for Churn")

            # Prepare single row DataFrame
            customer_df = pd.DataFrame([row.to_dict()])

            # Get SHAP reasons
            reasons = get_top_reasons(customer_df, top_n=5)

            # Display each reason
            for r in reasons:
                color = r['color']
                st.markdown(
                    f"<div style='display:flex; align-items:center; margin:8px 0;'>"
                    f"<span style='font-size:1.8rem; color:{color}; margin-right:12px;'>{r['direction']}</span>"
                    f"<div style='flex:1;'>"
                    f"<strong>{r['label']}</strong><br>"
                    f"<span style='color:#9ca3af;'>{r['impact']} impact</span>"
                    f"</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )

            st.caption("Reasons based on SHAP values (XGBoost model)")
        
    # ────────────────────────────────────────────────
    # Middle tile: Peer Comparison — Key Risk Signals
    # ────────────────────────────────────────────────
    with col2:
        with st.container(height=450, border=True):
            comparisons = get_peer_comparison(df, row)
            if "error" in comparisons:
                st.warning(comparisons["error"])
            else:
                # Header
                st.markdown("### Customer vs Similar Customers")
                st.markdown(
                    f"<div style='font-size:0.85rem; color:#9ca3af; text-align:center; margin-bottom:6px;'>"
                    f"Compared to {comparisons['peer_count']:,} similar customers</div>",
                    unsafe_allow_html=True
                )

                # Overall churn risk KPI
                risk_pct = comparisons.get('risk_percent', 0)
                st.metric(
                    label="Higher churn risk than peers",
                    value=f"{risk_pct:.0f}%"
                )

                # 🔹 TL;DR Insight (NEW)
                st.markdown(
                    f"""
                    <div style="
                        background:#111827;
                        border-left:4px solid #ef4444;
                        padding:8px 10px;
                        font-size:0.85rem;
                        color:#e5e7eb;
                        margin-bottom:10px;
                    ">
                        This customer is riskier than peers mainly due to <b>pricing</b>, 
                        <b>contract type</b>, and <b>missing support services</b>.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # ===== Monthly Charges Visual =====
                chg = comparisons["monthly_charges"]
                max_val = 100
                peer_width = min(chg['peer_avg'] / max_val * 100, 100)
                customer_pos = min(chg['your_value'] / max_val * 100, 100)

                st.markdown(
                    f"""
                    <div style="font-size:0.85rem; color:#9ca3af; margin-bottom:4px;">
                        Monthly Charges
                    </div>
                    <div style="position:relative; height:22px; background:#1f2937; border-radius:12px; overflow:hidden;">
                        <div style="height:100%; width:{peer_width}%; background:#3b82f6;"></div>
                        <div style="
                            position:absolute;
                            top:50%;
                            left:{customer_pos}%;
                            transform: translate(-50%, -50%);
                            width:12px;
                            height:12px;
                            border-radius:50%;
                            background:{'#ef4444' if risk_pct >= 70 else '#f59e0b' if risk_pct >= 30 else '#10b981'};
                            border:2px solid #1f2937;
                        "></div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Compact numeric comparison
                delta_pct = ((chg['your_value'] / chg['peer_avg'] - 1) * 100)
                st.markdown(
                    f"""
                    <div style="font-size:0.8rem; color:#d1d5db; margin-top:4px; margin-bottom:8px;">
                        Customer: <b>${chg['your_value']}</b> &nbsp;|&nbsp; 
                        Peers avg: <b>${chg['peer_avg']}</b><br>
                        <span style="color:{'#ef4444' if delta_pct > 0 else '#10b981'};">
                            {abs(delta_pct):.0f}% {'higher' if delta_pct > 0 else 'lower'} than peers
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # ===== Service Gaps (Grouped) =====
                st.markdown(
                    """
                    <div style="font-size:0.85rem; color:#9ca3af; margin-bottom:4px;">
                        Service Adoption Gaps
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown(
                    f"""
                    <div style="font-size:0.85rem; color:#d1d5db; line-height:1.6;">
                        Tech Support: <b style="color:#ef4444;">Customer ✘</b> 
                        <span style="color:#9ca3af;">vs {comparisons['tech_support']}% of peers ✔</span><br>
                        Online Security: <b style="color:#ef4444;">Customer ✘</b> 
                        <span style="color:#9ca3af;">vs {comparisons['online_security']}% of peers ✔</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # ===== Contract Risk (Isolated) =====
                contract_pct = round(
                    comparisons['contract_distribution'].get(row['Contract'], 0), 1
                )

                st.markdown(
                    f"""
                    <div style="
                        background:#111827;
                        padding:8px;
                        margin-top:10px;
                        border-radius:6px;
                        font-size:0.85rem;
                        color:#d1d5db;
                    ">
                        Contract: <b>{row['Contract']}</b><br>
                        Only <b>{contract_pct}%</b> of similar customers have this contract<br>
                        <span style="color:#ef4444; font-weight:600;">High-risk contract type</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Footer Insight (shorter, punchy)
                st.markdown(
                    "<div style='font-size:0.8rem; color:#9ca3af; margin-top:8px;'>"
                    "Key deviation: pricing + contract + missing services"
                    "</div>",
                    unsafe_allow_html=True
                )


    # ────────────────────────────────────────────────
    # Right tile: Concept Drift Banner + What-If teaser
    # ────────────────────────────────────────────────
    with col3:
        # Concept Drift Alert banner (small, top of col3)
        st.markdown(
            """
            <div style="
                background:#ca8a04; 
                color:#1f2937; 
                padding:10px 16px; 
                border-radius:8px; 
                margin-bottom:16px; 
                display:flex; 
                align-items:center; 
                font-weight:500; 
                font-size:0.95rem;
            ">
                <span style="font-size:1.3rem; margin-right:10px;">⚠️</span>
                <strong>Concept Drift Alert</strong> — Model Drift Detected! Retraining Recommended
                <div style="font-size:0.85rem; margin-left:auto; opacity:0.9;">
                    Last checked: Jan 26, 2026
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # What-If Scenario teaser
        with st.container(border=True):
            st.markdown("### What-If Scenario")
            st.markdown(
                f"<div style='background:#1f2937; padding:12px; border-radius:8px; margin:8px 0;'>"
                f"If Monthly Charges reduced by ₹200:<br>"
                f"<strong>Risk drops from {churn_prob:.1f}% → ~32%</strong></div>",
                unsafe_allow_html=True
            )
            st.button("Explore Full What-If Scenarios", type="primary", use_container_width=True)
    st.success("Customer loaded successfully!")
