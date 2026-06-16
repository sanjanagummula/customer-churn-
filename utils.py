# utils.py
import pandas as pd

def get_usage_matrix(df):
    """
    Creates a 2x2 usage matrix: Tenure (New/Long) × Monthly Usage (Low/High)
    Returns data ready for rendering in Streamlit.
    """
    # Define cutoffs (feel free to adjust these later)
    tenure_cutoff = 24          # New: <=24 months, Long: >24
    charges_cutoff = df['MonthlyCharges'].median()  # Low/High split at median

    # Create categories
    df_temp = df.copy()
    df_temp['TenureGroup'] = df_temp['tenure'].apply(
        lambda x: 'New' if x <= tenure_cutoff else 'Long'
    )
    df_temp['ChargesGroup'] = df_temp['MonthlyCharges'].apply(
        lambda x: 'Low' if x <= charges_cutoff else 'High'
    )

    # Calculate percentages for each cell
    total = len(df_temp)
    matrix = {
        ('New', 'Low'):   {'pct': 0, 'risk': 'Medium', 'color': '#f59e0b'},   # 🟡
        ('New', 'High'):  {'pct': 0, 'risk': 'High',   'color': '#ef4444'},   # 🔴
        ('Long', 'Low'):  {'pct': 0, 'risk': 'Low',    'color': '#10b981'},   # 🟢
        ('Long', 'High'): {'pct': 0, 'risk': 'Medium', 'color': '#f59e0b'},   # 🟡
    }

    # Fill real percentages
    for tenure_g, charge_g in matrix:
        count = len(df_temp[(df_temp['TenureGroup'] == tenure_g) & (df_temp['ChargesGroup'] == charge_g)])
        matrix[(tenure_g, charge_g)]['pct'] = round((count / total) * 100, 1) if total > 0 else 0

    return matrix, tenure_cutoff, charges_cutoff

def get_tenure_trends(df):
    """
    Returns two DataFrames for Behavioral Trends:
    - Average Monthly Charges by tenure bucket
    - Churn rate by tenure bucket
    """
    df_temp = df.copy()

    # Define tenure buckets
    bins = [0, 12, 24, 36, 48, float('inf')]
    labels = ['0-12 mo', '13-24 mo', '25-36 mo', '37-48 mo', '49+ mo']
    df_temp['TenureBucket'] = pd.cut(df_temp['tenure'], bins=bins, labels=labels, include_lowest=True)

    # Average Monthly Charges per bucket
    avg_charges = df_temp.groupby('TenureBucket')['MonthlyCharges'].mean().reset_index()
    avg_charges.columns = ['Tenure Bucket', 'Avg Monthly Charges']

    # Churn rate per bucket
    churn_rate = (
    df_temp
    .assign(ChurnNumeric=df_temp['Churn'].map({'Yes': 1, 'No': 0}))
    .groupby('TenureBucket')['ChurnNumeric']
    .mean() * 100
    )

    #churn_rate = df_temp.groupby('TenureBucket')['Churn'].mean() * 100  # % churn
    churn_rate = churn_rate.reset_index()
    churn_rate.columns = ['Tenure Bucket', 'Churn Rate (%)']

    return avg_charges, churn_rate

def get_peer_comparison(df, customer_row):
    """
    Personalized comparison: this customer vs peers in same tenure & charge band.
    Returns dict with comparison data.
    """
    # Define bands (same as usage matrix for consistency)
    tenure_cutoff = 24
    charges_cutoff = df['MonthlyCharges'].median()  # ~70.3

    # Determine customer's bands
    tenure_group = 'New' if customer_row['tenure'] <= tenure_cutoff else 'Long'
    charges_group = 'Low' if customer_row['MonthlyCharges'] <= charges_cutoff else 'High'

    # Filter peers (same tenure AND same charge band)
    peers = df[
        (df['tenure'].apply(lambda x: (x <= tenure_cutoff) if tenure_group == 'New' else (x > tenure_cutoff))) &
        (df['MonthlyCharges'].apply(lambda x: (x <= charges_cutoff) if charges_group == 'Low' else (x > charges_cutoff)))
    ]

    if len(peers) == 0:
        return {"error": "No similar peers found"}

    # Build comparisons
    comparisons = {
        "peer_group": f"{tenure_group} tenure + {charges_group} charges",
        "peer_count": len(peers),
        "monthly_charges": {
            "your_value": round(customer_row['MonthlyCharges'], 1),
            "peer_avg": round(peers['MonthlyCharges'].mean(), 1),
            "difference_pct": round(((customer_row['MonthlyCharges'] - peers['MonthlyCharges'].mean()) / peers['MonthlyCharges'].mean()) * 100, 1)
        },
        "tech_support": round((peers['TechSupport'] == 'Yes').mean() * 100, 1),
        "online_security": round((peers['OnlineSecurity'] == 'Yes').mean() * 100, 1),
        "contract_distribution": peers['Contract'].value_counts(normalize=True).head(2) * 100  # top 2 types
    }

    return comparisons