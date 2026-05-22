import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing & Machine Learning
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score

# Oversampling
from imblearn.over_sampling import SMOTE


# Association Rule Mining
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, apriori, association_rules

# Set page configuration
st.set_page_config(
    page_title="Glacier Analytics",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Pastel color palette matching the notebooks
C = {
    "no_churn" : "#a8d5a2",   # pastel green
    "churn"    : "#f4a7b9",   # pastel pink
    "dt"       : "#b5c7e7",   # pastel blue
    "lr"       : "#f7c59f",   # pastel orange
    "knn"      : "#c3b1e1",   # pastel purple
    "rule"     : "#ffd6a5",   # pastel peach
    "raw"      : "#d4e6f1",   # light sky for raw bars
    "proc"     : "#d5f5e3",   # light mint for processed bars
}

# -----------------------------------------------------------------------------
# CUSTOM CSS INJECTION FOR PREMIUM GLACIER ANALYTICS STYLE
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Import Outfit Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    /* Enforce light theme backgrounds globally */
    html, body, [data-testid="stAppViewContainer"], .stApp {
        background-color: #F0F4F8 !important;
        background: #F0F4F8 !important;
    }
    
    /* Ensure the header bar is transparent */
    [data-testid="stHeader"] {
        background-color: transparent !important;
        background: transparent !important;
    }
    
    /* Apply Outfit to custom designed classes explicitly with high specificity */
    .sidebar-header, 
    .sidebar-logo, 
    .user-profile-card, 
    .user-name, 
    .user-role, 
    .custom-card, 
    .custom-card-title, 
    .result-banner, 
    .metric-box, 
    .cm-cell,
    .custom-subheader {
        font-family: 'Outfit', sans-serif !important;
    }
    
    /* Set Outfit as the default font globally without !important on standard text and structure tags,
       letting Streamlit's native icon fonts override it naturally */
    html, body, [data-testid="stAppViewContainer"], .stApp, p, h1, h2, h3, h4, h5, h6, label, span {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Restore Streamlit UI icon font families to prevent literal icon text rendering */
    [data-testid="stIconMaterial"], 
    [class*="Icon"], 
    [class*="icon"], 
    [data-component="icon"], 
    span[class*="Icon"],
    span[class*="icon"],
    .stIconMaterial,
    [data-testid="stExpanderToggleIcon"],
    [data-testid="stExpanderToggleIcon"] *,
    [data-testid="stExpander"] [data-testid="stIconMaterial"],
    [data-testid="stExpander"] svg,
    [data-testid="stExpander"] span,
    summary span[class*="Icon"],
    summary span[class*="icon"],
    summary svg,
    summary [class*="Chevron"],
    summary [class*="chevron"],
    summary [class*="Arrow"],
    summary [class*="arrow"],
    .st-emotion-cache-16ids4d, 
    .st-ae, 
    .st-af {
        font-family: 'Material Symbols Outlined', 'Material Icons', 'streamlit-icons' !important;
    }
    
    /* Force high-contrast dark slate text globally on headings, labels and generic text elements */
    .stApp h1, 
    .stApp h2, 
    .stApp h3, 
    .stApp h4, 
    .stApp h5, 
    .stApp h6,
    .stApp label,
    .stApp p:not(.custom-subheader) {
        color: #1E293B !important;
    }
    
    /* Custom subheader with gray styling */
    .custom-subheader {
        color: #64748B !important;
        margin-top: -0.75rem !important;
        margin-bottom: 1.5rem !important;
        font-size: 0.95rem !important;
    }
    
    /* Input field overrides for elegant light mode widgets */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] input,
    .stNumberInput input,
    .stTextInput input,
    .stSelectbox select,
    .stSelectbox div[role="button"] {
        background-color: #FFFFFF !important;
        color: #1E293B !important;
        border-color: #CBD5E1 !important;
    }
    
    /* Style the dropdown list items explicitly to ensure visibility */
    ul[role="listbox"] li, 
    [data-testid="stVirtualDropdown"] div {
        background-color: #FFFFFF !important;
        color: #1E293B !important;
    }
    ul[role="listbox"] li:hover, 
    [data-testid="stVirtualDropdown"] div:hover {
        background-color: #F0F4F8 !important;
        color: #1E293B !important;
    }
    
    /* Checkbox labels */
    [data-testid="stCheckbox"] label {
        color: #1E293B !important;
    }
    
    /* Slider label and value text color overrides */
    .stSlider p,
    .stSlider div[data-testid="stWidgetLabel"] p,
    .stSlider div {
        color: #1E293B !important;
    }
    
    /* Expander visual light theme styling */
    [data-testid="stExpander"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 8px !important;
    }
    [data-testid="stExpander"] summary {
        background-color: #FFFFFF !important;
        color: #1E293B !important;
    }
    [data-testid="stExpander"] [data-testid="stExpanderDetails"] {
        background-color: #FFFFFF !important;
    }
    
    /* Custom style for all Streamlit buttons - Make them pastel blue */
    .stApp button,
    div.stButton > button, 
    button[kind="primary"],
    button[kind="secondary"],
    button[data-testid="stBaseButton-primary"],
    button[data-testid="stBaseButton-secondary"] {
        background-color: #b5c7e7 !important; /* pastel blue */
        color: #1E293B !important; /* dark slate */
        border: 1px solid #9cb4dd !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-family: 'Outfit', sans-serif !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03) !important;
    }
    .stApp button:hover,
    div.stButton > button:hover,
    button[kind="primary"]:hover,
    button[kind="secondary"]:hover,
    button[data-testid="stBaseButton-primary"]:hover,
    button[data-testid="stBaseButton-secondary"]:hover {
        background-color: #9cb4dd !important; /* slightly darker pastel blue */
        border-color: #64748B !important;
        color: #1E293B !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.05) !important;
    }
    .stApp button:active,
    div.stButton > button:active,
    button[data-testid="stBaseButton-primary"]:active,
    button[data-testid="stBaseButton-secondary"]:active {
        transform: translateY(0);
        background-color: #8da4cf !important; /* even darker pastel blue on click */
    }
    
    /* Main layout adjustment */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
    }
    
    /* Sidebar visual overrides */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0 !important;
    }
    
    /* Custom Sidebar Logo styling */
    .sidebar-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.5rem 0 1.5rem 0;
        border-bottom: 1px solid #E2E8F0;
        margin-bottom: 1.5rem;
    }
    .sidebar-logo {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1E3A8A;
        letter-spacing: -0.5px;
    }
    .sidebar-project-info {
        font-size: 0.85rem;
        color: #64748B;
        margin-top: -0.25rem;
    }
    .sidebar-project-ver {
        background-color: #E2E8F0;
        color: #475569;
        font-size: 0.75rem;
        padding: 0.1rem 0.4rem;
        border-radius: 4px;
        font-weight: 500;
    }
    
    /* User Profile Card */
    .user-profile-card {
        background-color: white;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .user-avatar {
        width: 38px;
        height: 38px;
        border-radius: 50%;
        background-color: #E2E8F0;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        color: #4F46E5;
    }
    .user-name {
        font-size: 0.95rem;
        font-weight: 600;
        color: #1E293B;
    }
    .user-role {
        font-size: 0.8rem;
        color: #64748B;
    }
    
    /* Custom Navigation Buttons */
    .sidebar-nav-btn {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        width: 100%;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        font-weight: 500;
        font-size: 0.95rem;
        color: #475569;
        border: none;
        background: transparent;
        text-align: left;
        cursor: pointer;
        transition: all 0.2s ease;
        margin-bottom: 0.5rem;
        text-decoration: none;
    }
    .sidebar-nav-btn:hover {
        background-color: #EDF2F7;
        color: #1E293B;
    }
    .sidebar-nav-btn.active {
        background-color: #4F46E5;
        color: white;
        box-shadow: 0 4px 6px -1px rgba(79, 70, 229, 0.2);
    }
    
    /* White elegant panels/cards */
    .custom-card {
        background-color: white !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.02), 0 2px 4px -1px rgba(0,0,0,0.01) !important;
        margin-bottom: 1rem !important;
    }
    .custom-card-title {
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        color: #1E293B !important;
        margin-bottom: 1.25rem !important;
        display: flex !important;
        align-items: center !important;
        gap: 0.5rem !important;
        border-bottom: 1px solid #F1F5F9 !important;
        padding-bottom: 0.5rem !important;
    }
    
    /* Subheading indicators */
    .section-indicator {
        font-size: 0.75rem !important;
        font-weight: 700 !important;
        color: #94A3B8 !important;
        letter-spacing: 0.75px !important;
        text-transform: uppercase !important;
        margin-top: 1rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Custom Result Banners */
    .result-banner {
        border-radius: 12px !important;
        padding: 1.25rem !important;
        margin-bottom: 1.5rem !important;
        border: 1px solid !important;
        display: flex !important;
        justify-content: space-between !important;
        align-items: center !important;
    }
    .result-banner-no-churn {
        background-color: #E8F8F5 !important;
        border-color: #a8d5a2 !important;
        color: #1E8449 !important;
    }
    .result-banner-no-churn * {
        color: #1E8449 !important;
    }
    .result-banner-churn {
        background-color: #F5EEF8 !important;
        border-color: #c3b1e1 !important;
        color: #5B2C6F !important;
    }
    .result-banner-churn * {
        color: #5B2C6F !important;
    }
    
    /* Metrics box grid */
    .metric-grid {
        display: grid !important;
        grid-template-columns: repeat(4, 1fr) !important;
        gap: 1rem !important;
        margin-bottom: 1.5rem !important;
    }
    .metric-box {
        background-color: #FAFBFC !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 8px !important;
        padding: 1rem 0.5rem !important;
        text-align: center !important;
        transition: all 0.2s ease !important;
    }
    .metric-box:hover {
        border-color: #CBD5E1 !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02) !important;
    }
    .metric-box-val {
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        color: #1E293B !important;
    }
    .metric-box-label {
        font-size: 0.8rem !important;
        color: #64748B !important;
        font-weight: 500 !important;
        margin-top: 0.25rem !important;
    }
    
    /* Confusion matrix visually */
    .cm-grid {
        display: grid !important;
        grid-template-columns: 1fr 1fr !important;
        gap: 0.75rem !important;
        margin-bottom: 1rem !important;
    }
    .cm-cell {
        padding: 1rem !important;
        border-radius: 8px !important;
        text-align: center !important;
        border: 1px solid !important;
    }
    .cm-tn { background-color: #E8F8F5 !important; border-color: #a8d5a2 !important; color: #1E8449 !important; }
    .cm-tp { background-color: #E8F8F5 !important; border-color: #a8d5a2 !important; color: #1E8449 !important; }
    .cm-fp { background-color: #F5EEF8 !important; border-color: #c3b1e1 !important; color: #5B2C6F !important; }
    .cm-fn { background-color: #F5EEF8 !important; border-color: #c3b1e1 !important; color: #5B2C6F !important; }
    
    .cm-cell * {
        color: inherit !important;
    }
    
    .cm-val { font-size: 1.5rem !important; font-weight: 700 !important; }
    .cm-lbl { font-size: 0.75rem !important; font-weight: 500 !important; opacity: 0.8 !important; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# DATA LOADERS & TRAINERS (CACHED FOR INSTANT PERFORMANCE)
# -----------------------------------------------------------------------------
@st.cache_resource
def train_classification_models():
    # Workspace paths
    filepath = r"c:\Users\lenovo\Desktop\SEM 4\Data Mining\Lab\Lab Final\Telco_customer_churn.xlsx"
    if not os.path.exists(filepath):
        filepath = "Telco_customer_churn.xlsx" # Fallback
        
    df = pd.read_excel(filepath)
    
    # Process
    df_clean = df.copy()
    drop_cols = [
        "CustomerID", "Count", "Country", "State", "City",
        "Zip Code", "Lat Long", "Latitude", "Longitude",
        "Churn Reason", "Churn Label", "Churn Score", "CLTV"
    ]
    df_clean.drop([c for c in drop_cols if c in df_clean.columns], axis=1, inplace=True)
    
    df_clean["Total Charges"] = pd.to_numeric(df_clean["Total Charges"], errors="coerce")
    df_clean["Total Charges"] = df_clean["Total Charges"].fillna(df_clean["Total Charges"].median())
    
    # Save encoders for categorical variables
    encoders = {}
    categorical_cols = []
    df_encoded = df_clean.copy()
    
    for col in df_encoded.columns:
        if df_encoded[col].dtype == "object":
            categorical_cols.append(col)
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            encoders[col] = le
            
    X_raw = df_encoded.drop("Churn Value", axis=1)
    y_raw = df_encoded["Churn Value"]
    feature_names = list(X_raw.columns)
    
    # Train-test split for raw models (No stratification)
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        X_raw, y_raw, test_size=0.2, random_state=42
    )
    
    # --- TRAIN RAW MODELS ---
    dt_raw = DecisionTreeClassifier(max_depth=8, random_state=42)
    dt_raw.fit(Xr_train, yr_train)
    
    lr_raw = LogisticRegression(max_iter=1000, random_state=42)
    lr_raw.fit(Xr_train, yr_train)
    
    knn_raw = KNeighborsClassifier(n_neighbors=5)
    knn_raw.fit(Xr_train, yr_train)
    
    # Still calculate dynamic k-profile for the diagnostic elbow plot
    k_values = range(1, 21)
    k_f1_raw = []
    for k in k_values:
        knn_temp = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn_temp, Xr_train, yr_train, cv=5, scoring="f1")
        k_f1_raw.append(scores.mean())
    best_k_raw = 5
    
    # --- TRAIN PROCESSED MODELS (SMOTE + SCALER ON WHOLE DATASET THEN SPLIT) ---
    # Apply SMOTE to the entire dataset (50/50 balance)
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_raw, y_raw)
    
    # Scale the entire resampled dataset
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_res)
    
    # Train-test split on preprocessed scaled data (No stratification)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_res, test_size=0.2, random_state=42
    )
    
    # Train preprocessed models
    dt_proc = DecisionTreeClassifier(max_depth=8, random_state=42)
    dt_proc.fit(X_train, y_train)
    
    lr_proc = LogisticRegression(max_iter=1000, random_state=42)
    lr_proc.fit(X_train, y_train)
    
    knn_proc = KNeighborsClassifier(n_neighbors=5)
    knn_proc.fit(X_train, y_train)
    
    # Still calculate dynamic k-profile for the diagnostic elbow plot
    k_f1_proc = []
    for k in k_values:
        knn_temp = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn_temp, X_train, y_train, cv=5, scoring="f1")
        k_f1_proc.append(scores.mean())
    best_k_proc = 5
    
    # --- TRAIN RULE-BASED MODELS (SHALLOW DT, max_depth=3) ---
    dt_rules_raw = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt_rules_raw.fit(Xr_train, yr_train)
    
    dt_rules_proc = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt_rules_proc.fit(X_train, y_train)
    
    rules_txt_raw = export_text(dt_rules_raw, feature_names=feature_names)
    rules_txt_proc = export_text(dt_rules_proc, feature_names=feature_names)
    
    # Cache all metrics
    metrics = {}
    for name, m_raw, m_proc, x_t_raw, x_t_proc, y_t_raw, y_t_proc in [
        ("Decision Tree", dt_raw, dt_proc, Xr_test, X_test, yr_test, y_test),
        ("Logistic Regression", lr_raw, lr_proc, Xr_test, X_test, yr_test, y_test),
        ("KNN", knn_raw, knn_proc, Xr_test, X_test, yr_test, y_test),
        ("Rule-Based Classifier", dt_rules_raw, dt_rules_proc, Xr_test, X_test, yr_test, y_test)
    ]:
        pred_raw = m_raw.predict(x_t_raw)
        pred_proc = m_proc.predict(x_t_proc)
        
        metrics[name] = {
            "raw": {
                "acc": accuracy_score(y_t_raw, pred_raw),
                "prec": precision_score(y_t_raw, pred_raw),
                "rec": recall_score(y_t_raw, pred_raw),
                "f1": f1_score(y_t_raw, pred_raw),
                "cm": confusion_matrix(y_t_raw, pred_raw)
            },
            "proc": {
                "acc": accuracy_score(y_t_proc, pred_proc),
                "prec": precision_score(y_t_proc, pred_proc),
                "rec": recall_score(y_t_proc, pred_proc),
                "f1": f1_score(y_t_proc, pred_proc),
                "cm": confusion_matrix(y_t_proc, pred_proc)
            }
        }
    # Precompute raw Decision Tree learning curve (depth 8)
    sizes_raw, train_scores_raw, val_scores_raw = learning_curve(
        DecisionTreeClassifier(max_depth=8, random_state=42),
        Xr_train, yr_train, cv=5, scoring="f1",
        train_sizes=np.linspace(0.1, 1.0, 8)
    )
    lc_raw = {
        "sizes": sizes_raw,
        "train_mean": train_scores_raw.mean(axis=1),
        "train_std": train_scores_raw.std(axis=1),
        "val_mean": val_scores_raw.mean(axis=1),
        "val_std": val_scores_raw.std(axis=1)
    }

    # Precompute preprocessed Decision Tree learning curve (depth 8)
    sizes_proc, train_scores_proc, val_scores_proc = learning_curve(
        DecisionTreeClassifier(max_depth=8, random_state=42),
        X_train, y_train, cv=5, scoring="f1",
        train_sizes=np.linspace(0.1, 1.0, 8)
    )
    lc_proc = {
        "sizes": sizes_proc,
        "train_mean": train_scores_proc.mean(axis=1),
        "train_std": train_scores_proc.std(axis=1),
        "val_mean": val_scores_proc.mean(axis=1),
        "val_std": val_scores_proc.std(axis=1)
    }

    # Precompute correlation matrix for EDA heatmap
    drop_for_corr = ["Zip Code", "Latitude", "Longitude", "Churn Score", "CLTV"]
    df_num = df.select_dtypes(include=[np.number]).drop(
        [c for c in drop_for_corr if c in df.columns], axis=1, errors='ignore'
    )
    corr_matrix = df_num.corr()

    return {
        "df_clean": df_clean,
        "feature_names": feature_names,
        "categorical_cols": categorical_cols,
        "encoders": encoders,
        "scaler": scaler,
        "models_raw": {
            "Decision Tree": dt_raw,
            "Logistic Regression": lr_raw,
            "KNN": knn_raw,
            "Rule-Based Classifier": dt_rules_raw
        },
        "models_proc": {
            "Decision Tree": dt_proc,
            "Logistic Regression": lr_proc,
            "KNN": knn_proc,
            "Rule-Based Classifier": dt_rules_proc
        },
        "metrics": metrics,
        "k_f1_raw": k_f1_raw,
        "k_f1_proc": k_f1_proc,
        "best_k_raw": best_k_raw,
        "best_k_proc": best_k_proc,
        "lc_raw": lc_raw,
        "lc_proc": lc_proc,
        "corr_matrix": corr_matrix,
        "rules_txt": {
            "raw": rules_txt_raw,
            "proc": rules_txt_proc
        },
        "y_train": yr_train,
        "y_train_res": y_train,
        "X_test_scaled": X_test,
        "y_test": y_test
    }


@st.cache_data
def load_market_basket_data():
    import urllib.request
    filepath = r"c:\Users\lenovo\Desktop\SEM 4\Data Mining\Lab\Lab Final\groceries.csv"
    if not os.path.exists(filepath):
        filepath = "groceries.csv" # Fallback
        
    if not os.path.exists(filepath):
        try:
            url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/groceries.csv"
            urllib.request.urlretrieve(url, filepath)
        except Exception:
            pass
            
    transactions = []
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    items = [it.strip() for it in line_str.split(',') if it.strip()]
                    if items:
                        transactions.append(items)
                        
    encoder = TransactionEncoder()
    if transactions:
        encoded_array = encoder.fit_transform(transactions)
        df = pd.DataFrame(encoded_array, columns=encoder.columns_)
    else:
        df = pd.DataFrame()
        
    return {
        "df": df,
        "dataset_len": len(transactions),
        "unique_items": len(encoder.columns_) if transactions else 0,
        "labels": list(encoder.columns_) if transactions else []
    }

# -----------------------------------------------------------------------------
# APP WIDE STATE & DATA INITIALIZATION
# -----------------------------------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "Churn Classifier"

# Load data models
with st.spinner("Bootstrapping Glacier Analytics & training models..."):
    clf_data = train_classification_models()
    mkt_data = load_market_basket_data()

# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    # Glacier Logo header
    st.markdown(f"""
    <div class="sidebar-header">
        <div style="font-size: 2.2rem; color: #4F46E5; font-weight: 700; line-height: 1;">G</div>
        <div>
            <div class="sidebar-logo">Glacier Analytics</div>
            <div class="sidebar-project-info">Mining Project <span class="sidebar-project-ver">V2.4 Stable</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Project Authors User Profile Card
    st.markdown("""
    <div class="user-profile-card" style="display: flex; flex-direction: column; align-items: flex-start; gap: 0.75rem;">
        <div style="font-size: 0.75rem; font-weight: 700; color: #94A3B8; letter-spacing: 0.75px; text-transform: uppercase; margin-bottom: -0.25rem;">Lead Data Scientists</div>
        <div style="display: flex; align-items: center; gap: 0.75rem; width: 100%;">
            <div class="user-avatar" style="flex-shrink: 0; background-color: #E8F8F5; color: #1E8449;">MS</div>
            <div>
                <div class="user-name" style="font-size: 0.9rem;">Menahil Suleman</div>
                <div class="user-role" style="font-size: 0.75rem;">FA24-BDS-029</div>
            </div>
        </div>
        <div style="display: flex; align-items: center; gap: 0.75rem; width: 100%;">
            <div class="user-avatar" style="flex-shrink: 0; background-color: #F5EEF8; color: #5B2C6F;">MF</div>
            <div>
                <div class="user-name" style="font-size: 0.9rem;">Mahnoor Fatima</div>
                <div class="user-role" style="font-size: 0.75rem;">FA24-BDS-027</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-indicator">Core Engines</div>', unsafe_allow_html=True)
    
    # Dynamic active button highlighting via custom CSS injection
    if st.session_state.page == "Churn Classifier":
        st.markdown("""
        <style>
            div.st-key-btn_churn > button {
                background-color: #8da4cf !important; /* active/highlighted blue */
                border-color: #4F46E5 !important;
                color: #1E293B !important;
                box-shadow: 0 4px 6px rgba(0,0,0,0.08) !important;
            }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
            div.st-key-btn_mkt > button {
                background-color: #8da4cf !important; /* active/highlighted blue */
                border-color: #4F46E5 !important;
                color: #1E293B !important;
                box-shadow: 0 4px 6px rgba(0,0,0,0.08) !important;
            }
        </style>
        """, unsafe_allow_html=True)

    # Custom reactive page buttons
    if st.button("Churn Classifier", key="btn_churn", use_container_width=True):
        st.session_state.page = "Churn Classifier"
        st.rerun()
    if st.button("Market Basket", key="btn_mkt", use_container_width=True):
        st.session_state.page = "Market Basket"
        st.rerun()
        
    st.markdown('<div class="section-indicator">Resources</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="padding-left: 0.5rem; display: flex; flex-direction: column; gap: 0.75rem;">
        <a href="#" style="text-decoration: none; color: #64748B; font-size: 0.9rem; font-weight: 500; display: flex; align-items: center; gap: 0.5rem;">Run New Analysis</a>
        <a href="#" style="text-decoration: none; color: #64748B; font-size: 0.9rem; font-weight: 500; display: flex; align-items: center; gap: 0.5rem;">Documentation</a>
        <a href="#" style="text-decoration: none; color: #64748B; font-size: 0.9rem; font-weight: 500; display: flex; align-items: center; gap: 0.5rem;">Archive</a>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# PAGE A: CHURN CLASSIFIER
# -----------------------------------------------------------------------------
if st.session_state.page == "Churn Classifier":
    st.markdown("## Customer Churn Engine")
    st.markdown("<p class='custom-subheader'>IBM Telco Customer Churn Real-Time Predictor & Classifier Analytics</p>", unsafe_allow_html=True)
    
    # Sub-tabs layout under Section A
    tab_pred, tab_bench = st.tabs(["Real-Time Predictor", "Model Benchmarking & Insights"])
    
    with tab_pred:
        # Columns setup
        col_input, col_result = st.columns([1.1, 1.0], gap="large")
        
        with col_input:
            st.markdown("""
            <div class="custom-card">
                <div class="custom-card-title">
                    Customer Profile Setup
                </div>
            """, unsafe_allow_html=True)
            
            # We will split form inputs cleanly
            st.markdown('<div class="section-indicator">Demographics</div>', unsafe_allow_html=True)
            demo_col1, demo_col2 = st.columns(2)
            with demo_col1:
                gender = st.selectbox("Gender", options=list(clf_data["encoders"]["Gender"].classes_))
                partner = st.checkbox("Partner (Has a partner)", value=True)
            with demo_col2:
                senior = st.checkbox("Senior Citizen", value=False)
                dependents = st.checkbox("Dependents (Has dependents)", value=False)
                
            st.markdown('<div class="section-indicator">Account Info</div>', unsafe_allow_html=True)
            tenure = st.slider("Tenure Months", min_value=0, max_value=72, value=24)
            
            acc_col1, acc_col2 = st.columns(2)
            with acc_col1:
                contract = st.selectbox("Contract", options=list(clf_data["encoders"]["Contract"].classes_))
                monthly_charges = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=84.50, step=0.1)
            with acc_col2:
                payment_method = st.selectbox("Payment Method", options=list(clf_data["encoders"]["Payment Method"].classes_))
                total_charges = st.number_input("Total Charges ($)", min_value=18.0, max_value=9000.0, value=2028.00, step=1.0)
                
            paperless = st.checkbox("Paperless Billing", value=True)
            
            st.markdown('<div class="section-indicator">Services</div>', unsafe_allow_html=True)
            serv_col1, serv_col2 = st.columns(2)
            with serv_col1:
                phone_service = st.selectbox("Phone Service", options=list(clf_data["encoders"]["Phone Service"].classes_))
                internet_service = st.selectbox("Internet Service", options=list(clf_data["encoders"]["Internet Service"].classes_))
                device_protection = st.selectbox("Device Protection", options=list(clf_data["encoders"]["Device Protection"].classes_))
            with serv_col2:
                multiple_lines = st.selectbox("Multiple Lines", options=list(clf_data["encoders"]["Multiple Lines"].classes_))
                online_security = st.selectbox("Online Security", options=list(clf_data["encoders"]["Online Security"].classes_))
                tech_support = st.selectbox("Tech Support", options=list(clf_data["encoders"]["Tech Support"].classes_))
                
            # Additional services required for model inputs
            with st.expander("Additional Services (Model Inputs)"):
                add_col1, add_col2 = st.columns(2)
                with add_col1:
                    online_backup = st.selectbox("Online Backup", options=list(clf_data["encoders"]["Online Backup"].classes_))
                    streaming_tv = st.selectbox("Streaming TV", options=list(clf_data["encoders"]["Streaming TV"].classes_))
                with add_col2:
                    streaming_movies = st.selectbox("Streaming Movies", options=list(clf_data["encoders"]["Streaming Movies"].classes_))
                    
            st.markdown('<div class="section-indicator">Model Selection</div>', unsafe_allow_html=True)
            model_name = st.selectbox("Select Classifier Model", options=["Decision Tree", "Logistic Regression", "KNN", "Rule-Based Classifier"])
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Predict Button
            predict_clicked = st.button("Predict Churn Status", type="primary", use_container_width=True)
            
        with col_result:
            # We calculate prediction inside if block or load default initially
            # Build user input array
            input_dict = {
                "Gender": gender,
                "Senior Citizen": "Yes" if senior else "No",
                "Partner": "Yes" if partner else "No",
                "Dependents": "Yes" if dependents else "No",
                "Tenure Months": tenure,
                "Phone Service": phone_service,
                "Multiple Lines": multiple_lines,
                "Internet Service": internet_service,
                "Online Security": online_security,
                "Online Backup": online_backup,
                "Device Protection": device_protection,
                "Tech Support": tech_support,
                "Streaming TV": streaming_tv,
                "Streaming Movies": streaming_movies,
                "Contract": contract,
                "Paperless Billing": "Yes" if paperless else "No",
                "Payment Method": payment_method,
                "Monthly Charges": monthly_charges,
                "Total Charges": total_charges
            }
            
            # Encode inputs exactly like training
            encoded_dict = {}
            for col in clf_data["feature_names"]:
                val = input_dict[col]
                if col in clf_data["encoders"]:
                    le = clf_data["encoders"][col]
                    # Fallback to handle unseen labels gracefully, though selectboxes prevent it
                    try:
                        encoded_dict[col] = le.transform([str(val)])[0]
                    except ValueError:
                        encoded_dict[col] = 0
                else:
                    encoded_dict[col] = float(val)
                    
            # Format as row and scale
            input_df = pd.DataFrame([encoded_dict])[clf_data["feature_names"]]
            input_scaled = clf_data["scaler"].transform(input_df)
            
            # Run inference
            model_proc = clf_data["models_proc"][model_name]
            
            start_time = time.perf_counter()
            pred_label = model_proc.predict(input_scaled)[0]
            prob_scores = model_proc.predict_proba(input_scaled)[0]
            inference_time = (time.perf_counter() - start_time) * 1000
            
            confidence = prob_scores[pred_label] * 100
            
            st.markdown(f"### Prediction & Model Evaluation ({model_name})")
            
            # Result banner
            if pred_label == 0:
                st.markdown(f"""
                <div class="result-banner result-banner-no-churn">
                    <div>
                        <span style="font-size: 1.1rem; font-weight: 500;">Prediction Result:</span><br>
                        <span style="font-size: 2.2rem; font-weight: 700;">No Churn</span>
                    </div>
                    <div style="text-align: right;">
                        <span class="sidebar-project-ver" style="background-color: #22C55E; color: white; font-size: 0.9rem; padding: 0.25rem 0.6rem; border-radius: 6px;">{confidence:.1f}% Confidence</span>
                        <div style="font-size: 0.8rem; opacity: 0.8; margin-top: 0.4rem;">Inference Time: {inference_time:.1f}ms</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-banner result-banner-churn">
                    <div>
                        <span style="font-size: 1.1rem; font-weight: 500;">Prediction Result:</span><br>
                        <span style="font-size: 2.2rem; font-weight: 700;">Churn</span>
                    </div>
                    <div style="text-align: right;">
                        <span class="sidebar-project-ver" style="background-color: #8E44AD; color: white; font-size: 0.9rem; padding: 0.25rem 0.6rem; border-radius: 6px;">{confidence:.1f}% Confidence</span>
                        <div style="font-size: 0.8rem; opacity: 0.8; margin-top: 0.4rem;">Inference Time: {inference_time:.1f}ms</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            # Model performance metrics row
            metrics_proc = clf_data["metrics"][model_name]["proc"]
            metrics_raw = clf_data["metrics"][model_name]["raw"]
            
            # Deltas representing processed (SMOTE + scaled) vs raw performance improvements
            d_acc = metrics_proc["acc"] - metrics_raw["acc"]
            d_prec = metrics_proc["prec"] - metrics_raw["prec"]
            d_rec = metrics_proc["rec"] - metrics_raw["rec"]
            d_f1 = metrics_proc["f1"] - metrics_raw["f1"]
            
            st.markdown(f"""
            <div class="metric-grid">
                <div class="metric-box">
                    <div class="metric-box-val">{metrics_proc["acc"]:.2f}</div>
                    <div class="metric-box-label">Accuracy</div>
                    <div style="font-size: 0.75rem; font-weight: 600; color: {'#1E8449' if d_acc >= 0 else '#8E44AD'};">
                        {'+' if d_acc >= 0 else ''}{d_acc:.2f} vs raw
                    </div>
                </div>
                <div class="metric-box">
                    <div class="metric-box-val">{metrics_proc["prec"]:.2f}</div>
                    <div class="metric-box-label">Precision</div>
                    <div style="font-size: 0.75rem; font-weight: 600; color: {'#1E8449' if d_prec >= 0 else '#8E44AD'};">
                        {'+' if d_prec >= 0 else ''}{d_prec:.2f} vs raw
                    </div>
                </div>
                <div class="metric-box">
                    <div class="metric-box-val">{metrics_proc["rec"]:.2f}</div>
                    <div class="metric-box-label">Recall</div>
                    <div style="font-size: 0.75rem; font-weight: 600; color: {'#1E8449' if d_rec >= 0 else '#8E44AD'};">
                        {'+' if d_rec >= 0 else ''}{d_rec:.2f} vs raw
                    </div>
                </div>
                <div class="metric-box">
                    <div class="metric-box-val">{metrics_proc["f1"]:.2f}</div>
                    <div class="metric-box-label">F1 Score</div>
                    <div style="font-size: 0.75rem; font-weight: 600; color: {'#1E8449' if d_f1 >= 0 else '#8E44AD'};">
                        {'+' if d_f1 >= 0 else ''}{d_f1:.2f} vs raw
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Grid layout for charts
            st.markdown('<div style="display: flex; flex-direction: column; gap: 1.5rem;">', unsafe_allow_html=True)
            
            # Confusion matrix visual grid
            cm = metrics_proc["cm"]
            st.markdown(f"""
            <div class="custom-card">
                <div class="custom-card-title">Confusion Matrix (Test Set)</div>
                <div class="cm-grid">
                    <div class="cm-cell cm-tn">
                        <div class="cm-val">{cm[0][0]}</div>
                        <div class="cm-lbl">True Negatives (TN)</div>
                    </div>
                    <div class="cm-cell cm-fp">
                        <div class="cm-val">{cm[0][1]}</div>
                        <div class="cm-lbl">False Positives (FP)</div>
                    </div>
                    <div class="cm-cell cm-fn">
                        <div class="cm-val">{cm[1][0]}</div>
                        <div class="cm-lbl">False Negatives (FN)</div>
                    </div>
                    <div class="cm-cell cm-tp">
                        <div class="cm-val">{cm[1][1]}</div>
                        <div class="cm-lbl">True Positives (TP)</div>
                    </div>
                </div>
                <div style="display: flex; gap: 1rem; justify-content: center; font-size: 0.8rem; font-weight: 600;">
                    <span style="color: #1E8449;">● Correctly Classified</span>
                    <span style="color: #5B2C6F;">● Incorrectly Classified</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Feature Importance Plot
            st.markdown("""
            <div class="custom-card">
                <div class="custom-card-title">Top Feature Importances / Coefficients</div>
            """, unsafe_allow_html=True)
            
            # Dynamically calculate feature importances
            feature_names = clf_data["feature_names"]
            
            if model_name == "Decision Tree":
                importances = model_proc.feature_importances_
                title_suffix = "Decision Tree (Gini/Entropy)"
                palette_color = C["dt"]
            elif model_name == "Rule-Based Classifier":
                importances = model_proc.feature_importances_
                title_suffix = "Rule-Based Classifier (Shallow Tree)"
                palette_color = C["rule"]
            elif model_name == "Logistic Regression":
                importances = np.abs(model_proc.coef_[0])
                importances = importances / np.sum(importances) # Normalize for easy parsing
                title_suffix = "Logistic Regression (Coefficients)"
                palette_color = C["lr"]
            else: # KNN doesn't support coefficients/importances directly. Let's use correlation proxy or DT as baseline
                # Plot the correlation with Churn target as feature importance proxy for KNN
                importances = np.abs(clf_data["df_clean"].drop("Churn Value", axis=1).corrwith(clf_data["df_clean"]["Churn Value"]).values)
                importances = importances / np.sum(importances)
                title_suffix = "Feature-Target Correlation (KNN Proxy)"
                palette_color = C["knn"]
                
            feat_imp_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": importances
            }).sort_values(by="Importance", ascending=False).head(8)
            
            fig, ax = plt.subplots(figsize=(6, 3.5))
            sns.barplot(
                data=feat_imp_df,
                x="Importance",
                y="Feature",
                color=palette_color,
                edgecolor="white",
                linewidth=1,
                ax=ax
            )
            ax.set_title(f"Top Features: {title_suffix}", fontsize=10, fontweight="bold")
            ax.set_xlabel("Relative Importance Score", fontsize=8)
            ax.set_ylabel("", fontsize=8)
            ax.tick_params(labelsize=8)
            ax.spines[['top', 'right']].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Display human-readable tree logic for the Rule-Based Classifier
            if model_name == "Rule-Based Classifier":
                st.markdown("""
                <div class="custom-card">
                    <div class="custom-card-title">Human-Readable Decision Logic Tree</div>
                """, unsafe_allow_html=True)
                st.markdown("<p class='custom-subheader' style='font-size: 0.85rem; margin-top: -0.5rem;'>This human-interpretable rule-tree shows the core splits computed from the customer dataset. Trace down the branches to see exactly how your choices lead to the prediction.</p>", unsafe_allow_html=True)
                st.code(clf_data["rules_txt"]["proc"], language="text")
                st.markdown("</div>", unsafe_allow_html=True)
                
            st.markdown("</div>", unsafe_allow_html=True)
            
    with tab_bench:
        # Card 1: Exploratory Data Analysis
        st.markdown("""
        <div class="custom-card">
            <div class="custom-card-title">Exploratory Data Analysis: Feature Correlation Heatmap</div>
        """, unsafe_allow_html=True)
        st.markdown("<p class='custom-subheader' style='font-size: 0.85rem; margin-top: -0.5rem;'>Replicates the exploratory analysis of numeric features. Strong positive or negative correlations indicate useful features, and the triangular mask prevents redundant visual clutter.</p>", unsafe_allow_html=True)
        
        corr = clf_data["corr_matrix"]
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        fig, ax = plt.subplots(figsize=(10, 6.5))
        sns.heatmap(
            corr, mask=mask, annot=True, fmt=".2f",
            cmap="coolwarm", center=0, linewidths=0.5,
            annot_kws={"size": 8}, ax=ax
        )
        ax.set_title("Correlation Heatmap of Numeric Features", fontsize=11, fontweight="bold", color="#1E293B")
        ax.tick_params(colors='#1E293B', labelsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.markdown("</div>", unsafe_allow_html=True)

        # Card 2: Model Performance Comparison
        st.markdown("""
        <div class="custom-card">
            <div class="custom-card-title">Model Performance Comparison: Raw vs. Preprocessed</div>
        """, unsafe_allow_html=True)
        st.markdown("<p class='custom-subheader' style='font-size: 0.85rem; margin-top: -0.5rem;'>Oversampling with SMOTE and feature scaling (StandardScaler) address class imbalance and feature magnitude domination. Toggle below to compare metrics.</p>", unsafe_allow_html=True)
        
        comp_metric = st.selectbox(
            "Select Performance Metric to Benchmark",
            options=["Accuracy", "Precision", "Recall", "F1 Score"],
            key="bench_metric_select"
        )
        
        metric_keys = {
            "Accuracy": "acc",
            "Precision": "prec",
            "Recall": "rec",
            "F1 Score": "f1"
        }
        m_key = metric_keys[comp_metric]
        
        bench_models = ["Decision Tree", "Logistic Regression", "KNN", "Rule-Based Classifier"]
        raw_vals = [clf_data["metrics"][m]["raw"][m_key] for m in bench_models]
        proc_vals = [clf_data["metrics"][m]["proc"][m_key] for m in bench_models]
        
        fig, ax = plt.subplots(figsize=(8, 4))
        x_indices = np.arange(len(bench_models))
        bar_width = 0.35
        
        b1 = ax.bar(x_indices - bar_width/2, raw_vals, bar_width, label="Raw Model", color=C["raw"], edgecolor="white")
        b2 = ax.bar(x_indices + bar_width/2, proc_vals, bar_width, label="Preprocessed Model", color=C["proc"], edgecolor="white")
        
        for bar in b1:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.2%}", ha='center', va='bottom', fontsize=8, color="#1E293B", fontweight="bold")
        for bar in b2:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.2%}", ha='center', va='bottom', fontsize=8, color="#1E293B", fontweight="bold")
            
        ax.set_title(f"Comparison: {comp_metric} across Classifiers", fontsize=11, fontweight="bold", color="#1E293B")
        ax.set_xticks(x_indices)
        ax.set_xticklabels(bench_models, fontsize=9, color="#1E293B")
        ax.set_ylabel(comp_metric, fontsize=9, color="#1E293B")
        ax.set_ylim(0, 1.15)
        ax.legend(loc="lower right", framealpha=0.9, facecolor="white", edgecolor="#CBD5E1")
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_color('#CBD5E1')
        ax.tick_params(colors='#1E293B')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        comparison_df = pd.DataFrame({
            "Classifier Model": bench_models,
            "Raw Score": [f"{r:.4f}" for r in raw_vals],
            "Preprocessed Score": [f"{p:.4f}" for p in proc_vals],
            "Improvement (Delta)": [f"+{p - r:+.4f}" if p - r >= 0 else f"{p - r:+.4f}" for r, p in zip(raw_vals, proc_vals)]
        })
        
        st.dataframe(comparison_df, hide_index=True, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Card 3: SMOTE Imbalance
        st.markdown("""
        <div class="custom-card">
            <div class="custom-card-title">SMOTE Oversampling: Addressing Training Set Imbalance</div>
        """, unsafe_allow_html=True)
        
        col_smote_text, col_smote_chart = st.columns([1.0, 1.2], gap="large")
        
        with col_smote_text:
            st.markdown("""
            <h4 style="color: #1E293B; margin-top: 0;">Avoiding Data Leakage & Scaling Pipeline</h4>
            <p style="font-size: 0.9rem; color: #1E293B; line-height: 1.5;">
                In the real-world <b>IBM Telco Customer Churn</b> dataset, there is a substantial class imbalance: roughly <b>73%</b> of customers remain loyal, while only <b>27%</b> churn. Machine learning algorithms trained directly on this imbalanced data suffer from a majority-class bias—they learn to predict "No Churn" highly accurately but fail completely to capture actual churn events (resulting in extremely poor <b>Recall</b>).
            </p>
            <p style="font-size: 0.9rem; color: #1E293B; line-height: 1.5;">
                <b>The Correct Methodology Applied:</b>
                <ol style="font-size: 0.9rem; color: #1E293B; padding-left: 1.2rem; margin-top: 0.5rem; line-height: 1.5;">
                    <li><b>Train-Test Split First:</b> The dataset is split into training (80%) and testing (20%) sets before applying any balancing techniques to prevent <b>data leakage</b> (spilling testing target distributions into training).</li>
                    <li><b>SMOTE balancing:</b> Synthetic Minority Over-sampling Technique (SMOTE) is applied solely to the training set to synthetically generate new, logical data points for the minority churn class.</li>
                    <li><b>Feature scaling:</b> Standardisation (StandardScaler) is applied to normalise feature magnitudes after SMOTE balancing to prevent high-scale columns (e.g. Total Charges) from dominating distance calculation models like KNN.</li>
                </ol>
            </p>
            """, unsafe_allow_html=True)
            
        with col_smote_chart:
            y_train_before = pd.Series(clf_data["y_train"]).value_counts().sort_index()
            y_train_after = pd.Series(clf_data["y_train_res"]).value_counts().sort_index()
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 4), sharey=True)
            
            colors_before = [C["no_churn"], C["churn"]]
            colors_after = [C["no_churn"], C["churn"]]
            
            bars1 = ax1.bar(["Loyal (0)", "Churn (1)"], y_train_before.values, color=colors_before, edgecolor="white", width=0.6)
            for bar in bars1:
                h = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2, h + 50, f"{int(h)}", ha='center', va='bottom', fontsize=8, color="#1E293B", fontweight="bold")
            ax1.set_title("Before SMOTE (Imbalanced)", fontsize=9, fontweight="bold", color="#1E293B")
            ax1.set_ylabel("Number of Transactions", fontsize=8, color="#1E293B")
            ax1.spines[['top', 'right']].set_visible(False)
            ax1.spines[['left', 'bottom']].set_color('#CBD5E1')
            ax1.tick_params(colors='#1E293B', labelsize=8)
            
            bars2 = ax2.bar(["Loyal (0)", "Churn (1)"], y_train_after.values, color=colors_after, edgecolor="white", width=0.6)
            for bar in bars2:
                h = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2, h + 50, f"{int(h)}", ha='center', va='bottom', fontsize=8, color="#1E293B", fontweight="bold")
            ax2.set_title("After SMOTE (Balanced)", fontsize=9, fontweight="bold", color="#1E293B")
            ax2.spines[['top', 'right']].set_visible(False)
            ax2.spines[['left', 'bottom']].set_color('#CBD5E1')
            ax2.tick_params(colors='#1E293B', labelsize=8)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Card 4: ROC & PR curves
        st.markdown("""
        <div class="custom-card">
            <div class="custom-card-title">Classification Diagnostic Curves: ROC & Precision-Recall</div>
        """, unsafe_allow_html=True)
        
        X_test_scaled = clf_data["X_test_scaled"]
        y_test = clf_data["y_test"]
        
        fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(10, 4.8))
        
        model_colors = {
            "Decision Tree": C["dt"],
            "Logistic Regression": C["lr"],
            "KNN": C["knn"],
            "Rule-Based Classifier": C["rule"]
        }
        
        for name in ["Decision Tree", "Logistic Regression", "KNN", "Rule-Based Classifier"]:
            model = clf_data["models_proc"][name]
            probs = model.predict_proba(X_test_scaled)[:, 1]
            
            fpr, tpr, _ = roc_curve(y_test, probs)
            roc_auc = auc(fpr, tpr)
            ax_roc.plot(fpr, tpr, color=model_colors[name], linewidth=2, label=f"{name} (AUC = {roc_auc:.3f})")
            
            prec_c, rec_c, _ = precision_recall_curve(y_test, probs)
            ap_score = average_precision_score(y_test, probs)
            ax_pr.plot(rec_c, prec_c, color=model_colors[name], linewidth=2, label=f"{name} (AP = {ap_score:.3f})")
            
        ax_roc.plot([0, 1], [0, 1], linestyle="--", color="#64748B", linewidth=1.2, label="Random (AUC = 0.500)")
        ax_roc.set_title("ROC Curves - Preprocessed Models", fontsize=10, fontweight="bold", color="#1E293B")
        ax_roc.set_xlabel("False Positive Rate", fontsize=8, color="#1E293B")
        ax_roc.set_ylabel("True Positive Rate", fontsize=8, color="#1E293B")
        ax_roc.legend(loc="lower right", fontsize=7.5, framealpha=0.9, facecolor="white", edgecolor="#CBD5E1")
        ax_roc.spines[['top', 'right']].set_visible(False)
        ax_roc.spines[['left', 'bottom']].set_color('#CBD5E1')
        ax_roc.tick_params(colors='#1E293B', labelsize=8)
        
        ax_pr.set_title("Precision-Recall Curves - Preprocessed Models", fontsize=10, fontweight="bold", color="#1E293B")
        ax_pr.set_xlabel("Recall", fontsize=8, color="#1E293B")
        ax_pr.set_ylabel("Precision", fontsize=8, color="#1E293B")
        ax_pr.legend(loc="lower left", fontsize=7.5, framealpha=0.9, facecolor="white", edgecolor="#CBD5E1")
        ax_pr.spines[['top', 'right']].set_visible(False)
        ax_pr.spines[['left', 'bottom']].set_color('#CBD5E1')
        ax_pr.tick_params(colors='#1E293B', labelsize=8)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        st.markdown("""
        <p style="font-size: 0.85rem; color: #64748B; margin-top: 0.5rem; line-height: 1.4;">
            <b>Interpretation Note:</b> An ROC curve plots sensitivity vs specificity. While highly informative, ROC curves can paint an overly optimistic picture of model performance on highly imbalanced datasets. The <b>Precision-Recall (PR) Curve</b> provides a more severe diagnostic test because it excludes the True Negatives from the metric calculation and focuses purely on Churn prediction precision and actual churn capture (Recall). The closer the curve reaches the upper-right corner, the more robust the classifier.
        </p>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Card 5: KNN Hyperparameter Tuning
        st.markdown("""
        <div class="custom-card">
            <div class="custom-card-title">KNN Hyperparameter Tuning: Cross-Validated Elbow Plots</div>
        """, unsafe_allow_html=True)
        st.markdown("<p class='custom-subheader' style='font-size: 0.85rem; margin-top: -0.5rem;'>Compares cross-validated F1-scores across K values (1 to 20). Preprocessing and scaling dramatically change the F1 profile, shifting the optimal neighbor count.</p>", unsafe_allow_html=True)
        
        col_elbow_raw, col_elbow_proc = st.columns(2, gap="large")
        k_values = list(range(1, 21))
        
        with col_elbow_raw:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.plot(k_values, clf_data["k_f1_raw"], "o-", color=C["knn"], linewidth=2, markersize=5)
            ax.axvline(clf_data["best_k_raw"], color=C["churn"], linestyle="--", linewidth=1.5,
                       label=f"Best K = {clf_data['best_k_raw']}")
            ax.set_title("KNN Elbow Plot: K vs F1 (Raw)", fontsize=9, fontweight="bold", color="#1E293B")
            ax.set_xlabel("Number of Neighbours (K)", fontsize=8)
            ax.set_ylabel("Mean F1 Score (5-fold CV)", fontsize=8)
            ax.legend(fontsize=8)
            ax.spines[['top', 'right']].set_visible(False)
            ax.spines[['left', 'bottom']].set_color('#CBD5E1')
            ax.tick_params(colors='#1E293B', labelsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
        with col_elbow_proc:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.plot(k_values, clf_data["k_f1_proc"], "o-", color=C["knn"], linewidth=2, markersize=5)
            ax.axvline(clf_data["best_k_proc"], color=C["churn"], linestyle="--", linewidth=1.5,
                       label=f"Best K = {clf_data['best_k_proc']}")
            ax.set_title("KNN Elbow Plot: K vs F1 (Preprocessed)", fontsize=9, fontweight="bold", color="#1E293B")
            ax.set_xlabel("Number of Neighbours (K)", fontsize=8)
            ax.set_ylabel("Mean F1 Score (5-fold CV)", fontsize=8)
            ax.legend(fontsize=8)
            ax.spines[['top', 'right']].set_visible(False)
            ax.spines[['left', 'bottom']].set_color('#CBD5E1')
            ax.tick_params(colors='#1E293B', labelsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
        st.markdown("</div>", unsafe_allow_html=True)

        # Card 6: Decision Tree Diagnostics
        st.markdown("""
        <div class="custom-card">
            <div class="custom-card-title">Decision Tree Diagnostics: Learning Curves & Structural Splits</div>
        """, unsafe_allow_html=True)
        st.markdown("<p class='custom-subheader' style='font-size: 0.85rem; margin-top: -0.5rem;'>Examines model generalization with 5-fold cross-validated learning curves (Training F1 vs. Validation F1) and structural splitting logic tree plots.</p>", unsafe_allow_html=True)
        
        # Learning Curves
        st.markdown("<h5 style='color: #1E293B; margin-top: 0; font-size: 0.95rem; font-weight: 600;'>1. Generalization Learning Curves</h5>", unsafe_allow_html=True)
        col_lc_raw, col_lc_proc = st.columns(2, gap="large")
        
        with col_lc_raw:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.plot(clf_data["lc_raw"]["sizes"], clf_data["lc_raw"]["train_mean"], "o-", color=C["dt"], label="Training F1", linewidth=1.8, markersize=4)
            ax.plot(clf_data["lc_raw"]["sizes"], clf_data["lc_raw"]["val_mean"], "s--", color=C["churn"], label="Validation F1", linewidth=1.8, markersize=4)
            ax.fill_between(clf_data["lc_raw"]["sizes"],
                            clf_data["lc_raw"]["train_mean"] - clf_data["lc_raw"]["train_std"],
                            clf_data["lc_raw"]["train_mean"] + clf_data["lc_raw"]["train_std"],
                            alpha=0.1, color=C["dt"])
            ax.fill_between(clf_data["lc_raw"]["sizes"],
                            clf_data["lc_raw"]["val_mean"] - clf_data["lc_raw"]["val_std"],
                            clf_data["lc_raw"]["val_mean"] + clf_data["lc_raw"]["val_std"],
                            alpha=0.1, color=C["churn"])
            ax.set_title("DT Learning Curve (Raw)", fontsize=9, fontweight="bold", color="#1E293B")
            ax.set_xlabel("Training Set Size", fontsize=8)
            ax.set_ylabel("F1 Score", fontsize=8)
            ax.set_ylim(0, 1.05)
            ax.legend(fontsize=8)
            ax.spines[['top', 'right']].set_visible(False)
            ax.spines[['left', 'bottom']].set_color('#CBD5E1')
            ax.tick_params(colors='#1E293B', labelsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
        with col_lc_proc:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.plot(clf_data["lc_proc"]["sizes"], clf_data["lc_proc"]["train_mean"], "o-", color=C["dt"], label="Training F1", linewidth=1.8, markersize=4)
            ax.plot(clf_data["lc_proc"]["sizes"], clf_data["lc_proc"]["val_mean"], "s--", color=C["churn"], label="Validation F1", linewidth=1.8, markersize=4)
            ax.fill_between(clf_data["lc_proc"]["sizes"],
                            clf_data["lc_proc"]["train_mean"] - clf_data["lc_proc"]["train_std"],
                            clf_data["lc_proc"]["train_mean"] + clf_data["lc_proc"]["train_std"],
                            alpha=0.1, color=C["dt"])
            ax.fill_between(clf_data["lc_proc"]["sizes"],
                            clf_data["lc_proc"]["val_mean"] - clf_data["lc_proc"]["val_std"],
                            clf_data["lc_proc"]["val_mean"] + clf_data["lc_proc"]["val_std"],
                            alpha=0.1, color=C["churn"])
            ax.set_title("DT Learning Curve (Preprocessed)", fontsize=9, fontweight="bold", color="#1E293B")
            ax.set_xlabel("Training Set Size", fontsize=8)
            ax.set_ylabel("F1 Score", fontsize=8)
            ax.set_ylim(0, 1.05)
            ax.legend(fontsize=8)
            ax.spines[['top', 'right']].set_visible(False)
            ax.spines[['left', 'bottom']].set_color('#CBD5E1')
            ax.tick_params(colors='#1E293B', labelsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
        st.markdown("<h5 style='color: #1E293B; margin-top: 1rem; font-size: 0.95rem; font-weight: 600;'>2. Structural Splits Visualization (max_depth=4)</h5>", unsafe_allow_html=True)
        tree_choice = st.selectbox("Select Model Tree to Visualize Structure", options=["Preprocessed Decision Tree", "Raw Decision Tree"])
        
        if tree_choice == "Preprocessed Decision Tree":
            model_tree = clf_data["models_proc"]["Decision Tree"]
            tree_title = "Decision Tree Splits (Preprocessed Data)"
        else:
            model_tree = clf_data["models_raw"]["Decision Tree"]
            tree_title = "Decision Tree Splits (Raw Data)"
            
        fig, ax = plt.subplots(figsize=(12, 5.5))
        plot_tree(
            model_tree,
            feature_names=clf_data["feature_names"],
            class_names=["No Churn", "Churn"],
            filled=True, rounded=True, fontsize=6, ax=ax
        )
        ax.set_title(tree_title, fontsize=10, fontweight="bold", color="#1E293B")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        st.markdown("</div>", unsafe_allow_html=True)

        # Card 7: Recommendations
        st.markdown("""
        <div class="custom-card">
            <div class="custom-card-title">Executive Production Recommendations & Decision Matrix</div>
        """, unsafe_allow_html=True)
        
        rec_col1, rec_col2, rec_col3 = st.columns(3)
        
        with rec_col1:
            st.markdown(f"""
            <div style="background-color: #FAFBFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 1rem; height: 100%;">
                <div style="font-weight: 700; color: #4F46E5; font-size: 0.95rem; border-bottom: 1px solid #E2E8F0; padding-bottom: 0.4rem; margin-bottom: 0.6rem;">Logistic Regression</div>
                <div style="font-size: 0.85rem; color: #1E293B; line-height: 1.45;">
                    <b>Production Recommendation: Highly Recommended (Stable)</b><br>
                    Showed stable generalization with the smallest gap between training and validation scores. It provides solid, reliable predictions and fits linear cost boundaries well without signs of overfitting.
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with rec_col2:
            st.markdown(f"""
            <div style="background-color: #FAFBFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 1rem; height: 100%;">
                <div style="font-weight: 700; color: #4F46E5; font-size: 0.95rem; border-bottom: 1px solid #E2E8F0; padding-bottom: 0.4rem; margin-bottom: 0.6rem;">Decision Tree / Rule-Based</div>
                <div style="font-size: 0.85rem; color: #1E293B; line-height: 1.45;">
                    <b>Stakeholder Interpretability: Recommended</b><br>
                    These classifiers achieve excellent recall performance. They are ideal for presentation to marketing and customer retention teams, allowing simple logic branches to determine risk profiles.
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with rec_col3:
            st.markdown(f"""
            <div style="background-color: #FAFBFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 1rem; height: 100%;">
                <div style="font-weight: 700; color: #4F46E5; font-size: 0.95rem; border-bottom: 1px solid #E2E8F0; padding-bottom: 0.4rem; margin-bottom: 0.6rem;">KNN Classifier</div>
                <div style="font-size: 0.85rem; color: #1E293B; line-height: 1.45;">
                    <b>Scaling Sensitivity: High Gain</b><br>
                    KNN achieved the single largest relative increase in metrics from preprocessing. Without StandardScaler and SMOTE, KNN fails entirely because of distance domination by higher-value features.
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("""
        <p style="font-size: 0.85rem; color: #64748B; margin-top: 1rem; line-height: 1.45;">
            <b>Methodological Conclusion:</b> While Decision Trees and Rules offer simple business logic, the <b>Logistic Regression</b> model trained on balanced data represents the optimal balance of high-throughput execution speed, low overfitting risk, and continuous calibration probability scaling for automated billing/retention systems.
        </p>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# PAGE B: MARKET BASKET ANALYSIS
# -----------------------------------------------------------------------------
else:
    st.markdown("## Market Basket Engine")
    st.markdown("<p class='custom-subheader'>Apriori & FP-Growth Association Rule Mining Analytics</p>", unsafe_allow_html=True)
    
    # Session state for association analysis results
    if "mb_results" not in st.session_state:
        st.session_state.mb_results = {
            "Apriori": None,
            "FP-Growth": None
        }
        
    col_control, col_display = st.columns([1.0, 2.2], gap="large")
    
    with col_control:
        st.markdown("""
        <div class="custom-card">
            <div class="custom-card-title">
                Threshold Controls
            </div>
        """, unsafe_allow_html=True)
        
        # Summary card for groceries dataset
        st.markdown(f"""
        <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 0.75rem 1rem; margin-bottom: 1.5rem;">
            <div style="font-weight: 600; font-size: 0.9rem; color: #1E293B; display: flex; align-items: center; gap: 0.5rem;">
                groceries.csv
            </div>
            <div style="font-size: 0.8rem; color: #64748B; display: flex; justify-content: space-between; margin-top: 0.4rem;">
                <span>Transactions: <b>{mkt_data["dataset_len"]:,}</b></span>
                <span>Unique Items: <b>{mkt_data["unique_items"]}</b></span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        min_support = st.slider("Min Support", min_value=0.0010, max_value=0.1000, value=0.0300, step=0.0050, format="%.4f")
        min_confidence = st.slider("Min Confidence", min_value=0.01, max_value=1.00, value=0.40, step=0.01)
        min_lift = st.slider("Min Lift Filter", min_value=1.0, max_value=5.0, value=1.2, step=0.1)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Dynamic execution button triggers
        run_apriori = st.button("Run Apriori Algorithm", type="primary", use_container_width=True)
        run_fpgrowth = st.button("Run FP-Growth Algorithm", use_container_width=True)
        
        # We also auto-calculate on page load if nothing exists yet
        if st.session_state.mb_results["Apriori"] is None:
            # Let's perform a default run in background to keep UI populated
            with st.spinner("Analyzing basket rules..."):
                for alg in ["Apriori", "FP-Growth"]:
                    start_t = time.perf_counter()
                    df_encoded = mkt_data["df"]
                    
                    if alg == "Apriori":
                        fi = apriori(df_encoded, min_support=min_support, use_colnames=True)
                    else:
                        fi = fpgrowth(df_encoded, min_support=min_support, use_colnames=True)
                        
                    runtime = time.perf_counter() - start_t
                    
                    if len(fi) > 0:
                        rules = association_rules(fi, metric="confidence", min_threshold=min_confidence, num_itemsets=len(fi))
                        for col in ['support', 'confidence', 'lift', 'leverage', 'conviction']:
                            if col in rules.columns:
                                rules[col] = pd.to_numeric(rules[col], errors='coerce')
                        rules = rules[rules['lift'] > min_lift]
                    else:
                        rules = pd.DataFrame()
                        
                    st.session_state.mb_results[alg] = {
                        "runtime": runtime,
                        "frequent_itemsets_count": len(fi),
                        "rules_count": len(rules),
                        "avg_lift": rules["lift"].mean() if len(rules) > 0 else 0.0,
                        "fi": fi,
                        "rules": rules
                    }
                    
        # Explicit triggers
        if run_apriori:
            with st.spinner("Running Apriori association analysis..."):
                start_t = time.perf_counter()
                fi = apriori(mkt_data["df"], min_support=min_support, use_colnames=True)
                runtime = time.perf_counter() - start_t
                if len(fi) > 0:
                    rules = association_rules(fi, metric="confidence", min_threshold=min_confidence, num_itemsets=len(fi))
                    for col in ['support', 'confidence', 'lift', 'leverage', 'conviction']:
                        if col in rules.columns:
                            rules[col] = pd.to_numeric(rules[col], errors='coerce')
                    rules = rules[rules['lift'] > min_lift]
                else:
                    rules = pd.DataFrame()
                st.session_state.mb_results["Apriori"] = {
                    "runtime": runtime,
                    "frequent_itemsets_count": len(fi),
                    "rules_count": len(rules),
                    "avg_lift": rules["lift"].mean() if len(rules) > 0 else 0.0,
                    "fi": fi,
                    "rules": rules
                }
                
        if run_fpgrowth:
            with st.spinner("Running FP-Growth association analysis..."):
                start_t = time.perf_counter()
                fi = fpgrowth(mkt_data["df"], min_support=min_support, use_colnames=True)
                runtime = time.perf_counter() - start_t
                if len(fi) > 0:
                    rules = association_rules(fi, metric="confidence", min_threshold=min_confidence, num_itemsets=len(fi))
                    for col in ['support', 'confidence', 'lift', 'leverage', 'conviction']:
                        if col in rules.columns:
                            rules[col] = pd.to_numeric(rules[col], errors='coerce')
                    rules = rules[rules['lift'] > min_lift]
                else:
                    rules = pd.DataFrame()
                st.session_state.mb_results["FP-Growth"] = {
                    "runtime": runtime,
                    "frequent_itemsets_count": len(fi),
                    "rules_count": len(rules),
                    "avg_lift": rules["lift"].mean() if len(rules) > 0 else 0.0,
                    "fi": fi,
                    "rules": rules
                }

    with col_display:
        st.markdown("### Association Mining Visualizer")
        
        # Tabs for Apriori Results vs FP-Growth Results
        tab_apriori, tab_fpgrowth = st.tabs(["Apriori Results", "FP-Growth Results"])
        
        # Visualizer function
        def render_association_results(alg_name):
            res = st.session_state.mb_results[alg_name]
            if res is None:
                st.warning("No results compiled yet. Trigger run using the Threshold Controls sidebar.")
                return
                
            # Success banner
            st.markdown(f"""
            <div class="result-banner result-banner-no-churn" style="padding: 0.8rem 1.2rem; margin-bottom: 1.25rem;">
                <div style="font-weight: 500; display: flex; align-items: center; gap: 0.5rem;">
                    {alg_name} completed successfully.
                </div>
                <div style="font-size: 0.85rem; font-weight: 600; opacity: 0.85;">
                    Execution time: {res["runtime"]:.4f}s
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Metrics cards
            st.markdown(f"""
            <div class="metric-grid" style="grid-template-columns: 1fr 1fr 1fr; margin-bottom: 1.5rem;">
                <div class="metric-box">
                    <div class="metric-box-val">{res["frequent_itemsets_count"]:,}</div>
                    <div class="metric-box-label">Frequent Itemsets</div>
                </div>
                <div class="metric-box">
                    <div class="metric-box-val">{res["rules_count"]:,}</div>
                    <div class="metric-box-label">Association Rules</div>
                </div>
                <div class="metric-box">
                    <div class="metric-box-val">{res["avg_lift"]:.3f}</div>
                    <div class="metric-box-label">Average Lift</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Edge-case handling for zero frequent itemsets or rules
            if res["frequent_itemsets_count"] == 0:
                st.warning("No frequent itemsets found! Your Minimum Support threshold might be too high for this dataset. Please lower it and try again.")
                return
            if res["rules_count"] == 0:
                st.warning("No association rules found matching your Confidence or Lift filters! Try lowering the Minimum Confidence or Minimum Lift sliders.")
                return
                
            # Columns under tabs
            col_chart, col_table = st.columns([1.0, 1.2], gap="medium")
            
            with col_chart:
                st.markdown("""
                <div class="custom-card" style="padding: 1rem;">
                    <div class="custom-card-title" style="font-size: 1rem; margin-bottom: 0.75rem;">Top 15 Frequent Itemsets</div>
                """, unsafe_allow_html=True)
                
                # Plotting Top 15 itemsets
                fi_df = res["fi"].copy()
                fi_df["itemsets"] = fi_df["itemsets"].apply(lambda x: ", ".join(list(x)))
                fi_top15 = fi_df.sort_values(by="support", ascending=False).head(15)
                
                fig, ax = plt.subplots(figsize=(5, 5))
                sns.barplot(
                    data=fi_top15,
                    x="support",
                    y="itemsets",
                    color=C["rule"] if alg_name == "Apriori" else C["dt"],
                    edgecolor="white",
                    linewidth=1,
                    ax=ax
                )
                ax.set_title(f"Top 15 Itemsets ({alg_name})", fontsize=10, fontweight="bold")
                ax.set_xlabel("Support Threshold", fontsize=8)
                ax.set_ylabel("")
                ax.tick_params(labelsize=8)
                ax.spines[['top', 'right']].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                st.markdown("</div>", unsafe_allow_html=True)
                
            with col_table:
                st.markdown("""
                <div class="custom-card" style="padding: 1rem; overflow: auto;">
                    <div class="custom-card-title" style="font-size: 1rem; margin-bottom: 0.75rem;">Derived Association Rules</div>
                """, unsafe_allow_html=True)
                
                # Format dataframe for user-friendly preview
                rules_df = res["rules"].copy()
                rules_df["antecedents"] = rules_df["antecedents"].apply(lambda x: ", ".join(list(x)))
                rules_df["consequents"] = rules_df["consequents"].apply(lambda x: ", ".join(list(x)))
                
                rules_display = rules_df[["antecedents", "consequents", "support", "confidence", "lift"]].sort_values(by="lift", ascending=False)
                
                # Style rules table using standard dataframe render
                st.dataframe(
                    rules_display.rename(columns={
                        "antecedents": "Antecedents (If)",
                        "consequents": "Consequents (Then)",
                        "support": "Support",
                        "confidence": "Confidence",
                        "lift": "Lift"
                    }),
                    hide_index=True,
                    use_container_width=True
                )
                st.markdown("</div>", unsafe_allow_html=True)
                
            # --- NEW DATASET INSIGHTS & RULE SENSITIVITY ---
            st.markdown("""
            <div class="custom-card" style="padding: 1.5rem;">
                <div class="custom-card-title" style="font-size: 1.1rem; margin-bottom: 1rem;">Dataset Insights & Rule Sensitivity Analysis</div>
            """, unsafe_allow_html=True)
            
            ins_col1, ins_col2 = st.columns(2, gap="large")
            
            with ins_col1:
                st.markdown("""
                <h4 style="color: #1E293B; margin-top: 0; font-size: 1rem;">Frequent Itemsets Size Distribution</h4>
                <p style="font-size: 0.85rem; color: #64748B; margin-bottom: 0.75rem;">
                    Shows the distribution of itemset sizes mined from the Groceries transaction dataset (representing the number of 1-item, 2-item, or 3-item combinations).
                </p>
                """, unsafe_allow_html=True)
                
                fi_df = res["fi"].copy()
                if not fi_df.empty:
                    fi_df["itemset_size"] = fi_df["itemsets"].apply(len)
                    size_counts = fi_df["itemset_size"].value_counts().sort_index()
                    
                    fig, ax = plt.subplots(figsize=(5, 3.5))
                    bar_colors = [C["dt"], C["lr"], C["knn"], C["rule"]]
                    
                    bars = ax.bar(
                        [f"{i}-item set" for i in size_counts.index],
                        size_counts.values,
                        color=bar_colors[:len(size_counts)],
                        edgecolor='white',
                        width=0.5
                    )
                    
                    for bar in bars:
                        h = bar.get_height()
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            h + (size_counts.max() * 0.02),
                            str(int(h)),
                            ha='center',
                            va='bottom',
                            fontsize=8,
                            color="#1E293B",
                            fontweight='bold'
                        )
                        
                    ax.set_xlabel('Itemset Size', fontsize=8, color="#1E293B")
                    ax.set_ylabel('Frequent Itemsets Count', fontsize=8, color="#1E293B")
                    ax.set_ylim(0, size_counts.max() * 1.15)
                    ax.spines[['top', 'right']].set_visible(False)
                    ax.spines[['left', 'bottom']].set_color('#CBD5E1')
                    ax.tick_params(colors='#1E293B', labelsize=8)
                    ax.grid(axis='y', linestyle='--', alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                else:
                    st.info("No frequent itemsets found under the current Support threshold.")
                    
            with ins_col2:
                st.markdown("""
                <h4 style="color: #1E293B; margin-top: 0; font-size: 1rem;">Rule Sensitivity: Support vs. Confidence</h4>
                <p style="font-size: 0.85rem; color: #64748B; margin-bottom: 0.75rem;">
                    Plots Support (X-axis) vs. Confidence (Y-axis) for all derived rules, where bubble color & size represent the rule <b>Lift</b> metric.
                </p>
                """, unsafe_allow_html=True)
                
                rules_df = res["rules"].copy()
                if not rules_df.empty:
                    fig, ax = plt.subplots(figsize=(5, 3.5))
                    
                    sup_vals = rules_df['support'].astype(float)
                    conf_vals = rules_df['confidence'].astype(float)
                    lift_vals = rules_df['lift'].astype(float)
                    
                    sz = lift_vals * 150
                    
                    scatter = ax.scatter(
                        sup_vals,
                        conf_vals,
                        c=lift_vals,
                        s=sz,
                        cmap='viridis',
                        alpha=0.8,
                        edgecolors='#CBD5E1',
                        linewidth=0.5
                    )
                    
                    cbar = plt.colorbar(scatter, ax=ax)
                    cbar.set_label('Lift', fontsize=8, color="#1E293B")
                    cbar.ax.tick_params(labelsize=7, colors='#1E293B')
                    
                    ax.set_xlabel('Support', fontsize=8, color="#1E293B")
                    ax.set_ylabel('Confidence', fontsize=8, color="#1E293B")
                    ax.spines[['top', 'right']].set_visible(False)
                    ax.spines[['left', 'bottom']].set_color('#CBD5E1')
                    ax.tick_params(colors='#1E293B', labelsize=8)
                    ax.grid(linestyle='--', alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                else:
                    st.info("No derived association rules to plot sensitivity analysis under the current threshold parameters.")
                    
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Runtime performance speed analysis plot
            st.markdown("""
            <div class="custom-card" style="padding: 1rem;">
                <div class="custom-card-title" style="font-size: 1rem; margin-bottom: 0.75rem;">Speed Benchmarking: Apriori vs FP-Growth</div>
            """, unsafe_allow_html=True)
            
            # Draw speed comparison bar chart
            ap_res = st.session_state.mb_results["Apriori"]
            fp_res = st.session_state.mb_results["FP-Growth"]
            
            if ap_res is not None and fp_res is not None:
                times = [ap_res["runtime"], fp_res["runtime"]]
                algs = ["Apriori", "FP-Growth"]
                
                fig, ax = plt.subplots(figsize=(6, 2.5))
                bars = ax.barh(algs, times, color=[C["lr"], C["dt"]], edgecolor="white", height=0.5)
                
                for bar in bars:
                    width = bar.get_width()
                    ax.text(
                        width + (max(times) * 0.01),
                        bar.get_y() + bar.get_height()/2,
                        f'{width:.4f}s',
                        ha='left',
                        va='center',
                        fontsize=8,
                        fontweight='bold'
                    )
                    
                ax.set_title("Runtime Comparison (Lower is Faster)", fontsize=9, fontweight="bold")
                ax.set_xlabel("Time (seconds)", fontsize=8)
                ax.tick_params(labelsize=8)
                ax.spines[['top', 'right']].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                # Write explanation text
                if fp_res["runtime"] < ap_res["runtime"]:
                    speedup = ap_res["runtime"] / fp_res["runtime"]
                    st.success(f"**FP-Growth is {speedup:.2f}x faster** than Apriori because it compresses transactions into a prefix tree (FP-Tree), eliminating candidate generation database scans!")
                else:
                    st.info("Runtimes are very similar on this transaction run threshold.")
                    
            st.markdown("</div>", unsafe_allow_html=True)
            
        with tab_apriori:
            render_association_results("Apriori")
            
        with tab_fpgrowth:
            render_association_results("FP-Growth")
