# online_retail_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import joblib
from io import StringIO



@st.cache_data
def load_data():
    file_path = r"D:\ALL\ai/Online Retail.xlsx"
    df = pd.read_excel(file_path)
    df.drop(columns="CustomerID",inplace=True)
    df['StockCode'] = df['StockCode'].astype(str)
    df['Description'] = df['Description'].astype(str)

    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    # TotalPrice
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

    return df

df = load_data()

# PAGE TITLE
st.title("Online Retail Data Analysis Dashboard")

# RAW DATA
st.header("Raw Data Overview")
st.write("First look at the dataset:")
st.dataframe(df.head())

if st.checkbox("Show dataset info"):
    
    buffer = StringIO()
    df.info(buf=buffer)

    info_text = buffer.getvalue()
    st.text(info_text)
    info_str = "\n".join(buffer)
    st.text(info_str)

st.write("Missing Values:")
st.write(df.isna().sum())

# OUTLIERS
st.header("Outlier Visualization")

fig = plt.figure(figsize=(14, 6))
sns.boxplot(data=df[['Quantity', 'UnitPrice', 'TotalPrice']])
st.pyplot(fig)

# CLEANING
st.header("Data Cleaning")

df_clean = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)].copy()

def cap_outliers(col):
    Q1 = col.quantile(0.25)
    Q3 = col.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return np.clip(col, lower, upper)

df_clean['Quantity'] = cap_outliers(df_clean['Quantity'])
df_clean['UnitPrice'] = cap_outliers(df_clean['UnitPrice'])
df_clean['TotalPrice'] = cap_outliers(df_clean['TotalPrice'])

# re-plot cleaned boxplot
fig2 = plt.figure(figsize=(14, 6))
sns.boxplot(data=df_clean[['Quantity', 'UnitPrice', 'TotalPrice']])
st.pyplot(fig2)

# Remove cancelled invoices
df_clean = df_clean[~df_clean['InvoiceNo'].astype(str).str.startswith('C')]

# Remove NA & duplicates
df_clean['Description'] = df_clean['Description'].fillna('Unknown')
df_clean = df_clean.drop_duplicates()

# FEATURE ENGINEERING
st.header("Feature Engineering")

df_clean['Hour'] = df_clean['InvoiceDate'].dt.hour
df_clean['DayOfWeek'] = df_clean['InvoiceDate'].dt.dayofweek
df_clean['Month'] = df_clean['InvoiceDate'].dt.month

def get_season(month):
    if month in [12, 1, 2]: return 'Winter'
    if month in [3, 4, 5]: return 'Spring'
    if month in [6, 7, 8]: return 'Summer'
    return 'Autumn'

df_clean['Season'] = df_clean['Month'].apply(get_season)

st.write("Data after feature engineering:")
st.dataframe(df_clean.head())

# EDA
st.header("Exploratory Data Analysis")

# Country spending
country_spend = df_clean.groupby('Country')['TotalPrice'].sum().reset_index().sort_values('TotalPrice', ascending=False)

st.subheader("Top Countries by Spending")
st.dataframe(country_spend.head(20))

# Bar: top 15 countries by transactions
country_counts = df_clean['Country'].value_counts().head(15)

fig3 = plt.figure(figsize=(10, 5))
country_counts.plot(kind='bar')
plt.xticks(rotation=45)
plt.ylabel("Transactions")
st.pyplot(fig3)

# Monthly Trends
st.header("Monthly Sales Trends")

ts = df_clean.resample('M', on='InvoiceDate')['TotalPrice'].sum().reset_index()
ts['YearMonth'] = ts['InvoiceDate'].dt.to_period('M').dt.to_timestamp()

fig4 = plt.figure(figsize=(12, 5))
plt.plot(ts['YearMonth'], ts['TotalPrice'])
plt.xticks(rotation=45)
plt.ylabel("Total Sales")
st.pyplot(fig4)

# TOP PRODUCTS
st.header("Top Selling Products")

# Quantity
prod_qty = df_clean.groupby(['StockCode', 'Description'])['Quantity'].sum().reset_index()
prod_qty = prod_qty.sort_values('Quantity', ascending=False)

st.subheader("Top Products by Quantity")
st.dataframe(prod_qty.head(15))

fig5 = plt.figure(figsize=(10, 6))
plt.barh(prod_qty['Description'].head(15)[::-1], prod_qty['Quantity'].head(15)[::-1])
st.pyplot(fig5)

# Revenue
prod_sales = df_clean.groupby(['StockCode', 'Description'])['TotalPrice'].sum().reset_index()
prod_sales = prod_sales.sort_values('TotalPrice', ascending=False)

st.subheader("Top Products by Revenue")
st.dataframe(prod_sales.head(15))

fig6 = plt.figure(figsize=(10, 6))
plt.barh(prod_sales['Description'].head(15)[::-1], prod_sales['TotalPrice'].head(15)[::-1])
st.pyplot(fig6)

st.success("Dashboard loaded successfully.")


try:
    from xgboost import XGBRegressor
    xgb_available = True
except Exception:
    xgb_available = False

st.set_page_config(layout="wide", page_title="Online Retail - CLV Dashboard")

@st.cache_data
def load_data(path=r"D:\ALL\ai/Online Retail.xlsx"):
    df = pd.read_excel(path)
    # ensure column types
    df['StockCode'] = df['StockCode'].astype(str)
    df['Description'] = df['Description'].astype(str)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    # TotalPrice
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    return df

@st.cache_data
def basic_cleaning(df):
    df = df.copy()
    # remove cancelled & nonpositive values
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    # drop missing CustomerID (assignment asked to handle these -> we drop here for ML)
    df = df.dropna(subset=['CustomerID'])
    df['Description'] = df['Description'].fillna('Unknown')
    df = df.drop_duplicates()
    return df

def cap_outliers_iqr(col):
    Q1 = col.quantile(0.25)
    Q3 = col.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return np.clip(col, lower, upper)

def feature_engineer(df):
    df = df.copy()
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    df['Hour'] = df['InvoiceDate'].dt.hour
    df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek
    df['Month'] = df['InvoiceDate'].dt.month
    def get_season(month):
        if month in [12,1,2]: return 'Winter'
        if month in [3,4,5]: return 'Spring'
        if month in [6,7,8]: return 'Summer'
        return 'Autumn'
    df['Season'] = df['Month'].apply(get_season)
    # simple Category extraction
    df['Category'] = df['Description'].astype(str).str.split().str[0]
    # CLV per customer
    clv = df.groupby('CustomerID')['TotalPrice'].sum().reset_index().rename(columns={'TotalPrice':'CLV'})
    # Recency/Frequency/Monetary (RFM)
    max_date = df['InvoiceDate'].max()
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (max_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalPrice': 'sum'
    }).reset_index()
    rfm.columns = ['CustomerID','Recency','Frequency','Monetary']
    # Merge features back at customer level (we'll aggregate other features)
    # Session-like features (simplified)
    df = df.sort_values(['CustomerID','InvoiceDate'])
    df['TimeDiff'] = df.groupby('CustomerID')['InvoiceDate'].diff()
    df['NewSession'] = (df['TimeDiff'] > pd.Timedelta(minutes=30)).astype(int)
    df['SessionID'] = df.groupby('CustomerID')['NewSession'].cumsum()
    session_features = df.groupby('CustomerID').agg({
        'SessionID':'nunique',
        'TotalPrice':'mean',
        'Quantity':'mean'
    }).reset_index().rename(columns={'SessionID':'SessionCount','TotalPrice':'AvgSessionValue','Quantity':'AvgSessionQuantity'})
    active_days = df.groupby('CustomerID')['InvoiceDate'].nunique().reset_index().rename(columns={'InvoiceDate':'ActiveDays'})
    week = df['InvoiceDate'].dt.isocalendar().week
    df['Week'] = week
    max_week = df['Week'].max()
    week_recency = df.groupby('CustomerID')['Week'].max().reset_index()
    week_recency['WeekRecency'] = max_week - week_recency['Week']
    week_recency = week_recency[['CustomerID','WeekRecency']]
    customer_data = rfm.merge(clv, on='CustomerID', how='left') \
                       .merge(session_features, on='CustomerID', how='left') \
                       .merge(active_days, on='CustomerID', how='left') \
                       .merge(week_recency, on='CustomerID', how='left')
    # Fill NA where necessary
    customer_data = customer_data.fillna(0)
    return df, customer_data

def encode_and_scale(customer_data):
    cd = customer_data.copy()
    # Country & Season not present at aggregated customer table; if you want country distribution add aggregation
    # For this script we stick to engineered numeric features
    # Create target CLV
    X = cd.drop(['CustomerID','CLV'], axis=1)
    y = cd['CLV']
    # Simple scaling
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    return X_scaled, y, scaler

# -------------------------------
# App layout
# -------------------------------
st.title("Online Retail — CLV Modeling Dashboard")

st.markdown("""
**Assignment**: Preprocess 500k+ transactions and build CLV models (Linear baseline, Random Forest, XGBoost).
Upload dataset path or use the default local path.
""")

# Sidebar controls
st.sidebar.header("Controls")
path_input = st.sidebar.text_input("Excel file path", value=r"D:\ALL\ai/Online Retail.xlsx")
subsample_pct = st.sidebar.slider("Subsample for faster prototyping (%)", min_value=10, max_value=100, value=100, step=5)
train_models = st.sidebar.button("Run/train models")
random_state = st.sidebar.number_input("Random seed", min_value=0, value=42, step=1)
rf_n_estimators = st.sidebar.slider("Random Forest n_estimators", 50, 500, value=200, step=50)
test_size = st.sidebar.slider("Test size (%)", 5, 50, value=20, step=5)

# Load data
with st.spinner("Loading data..."):
    df_raw = load_data(path_input)

st.subheader("Raw data sample")
st.dataframe(df_raw.head())

st.markdown("Basic info and missing values:")
st.write(df_raw.info())
st.write(df_raw.isna().sum())

# Quick EDA: top countries by spend
st.subheader("Top countries by total spend (sample)")
df_raw['TotalPrice'] = df_raw['Quantity'] * df_raw['UnitPrice']
country_spend = df_raw.groupby('Country')['TotalPrice'].sum().reset_index().sort_values('TotalPrice', ascending=False)
st.dataframe(country_spend.head(10))

# Cleaning & feature engineering
with st.spinner("Cleaning and feature engineering..."):
    df_clean = basic_cleaning(df_raw)
    df_clean['TotalPrice'] = df_clean['Quantity'] * df_clean['UnitPrice']
    # cap numeric outliers (column-wise) for the primary numeric fields to avoid crazy values
    for c in ['Quantity','UnitPrice','TotalPrice']:
        if c in df_clean.columns:
            df_clean[c] = cap_outliers_iqr(df_clean[c])
    df_tx, customer_data = feature_engineer(df_clean)

st.subheader("Customer-level feature sample")
st.dataframe(customer_data.head())

# Allow subsampling of customers to speed up training if requested
if subsample_pct < 100:
    customer_data = customer_data.sample(frac=subsample_pct/100.0, random_state=random_state).reset_index(drop=True)
    st.info(f"Subsampled customers: {len(customer_data)} rows ({subsample_pct}%)")

# Prepare X,y
X, y, scaler = encode_and_scale(customer_data)

st.write("Feature matrix shape:", X.shape)
st.write("Target distribution (CLV):")
st.write(y.describe())

# Optionally show correlation heatmap
if st.checkbox("Show feature correlation heatmap"):
    fig_corr = plt.figure(figsize=(8,6))
    sns.heatmap(pd.concat([X, y.rename('CLV')], axis=1).corr(), annot=True, fmt=".2f")
    st.pyplot(fig_corr)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(test_size/100.0), random_state=random_state)

st.markdown("---")
st.header("Supervised ML — Regression models (predict CLV)")

if not train_models:
    st.info("Click **Run/train models** in the sidebar to train Linear Regression, Random Forest and (if available) XGBoost.")
else:
    # 1) Linear Regression basline
    with st.spinner("Training Linear Regression baseline..."):
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        lr_preds = lr.predict(X_test)
        lr_rmse = np.sqrt(mean_squared_error(y_test, lr_preds))
        lr_mae = mean_absolute_error(y_test, lr_preds)
        lr_r2 = r2_score(y_test, lr_preds)

    # 2) RandomForest
    with st.spinner("Training Random Forest..."):
        rf = RandomForestRegressor(n_estimators=rf_n_estimators, max_depth=10, random_state=random_state, n_jobs=-1)
        rf.fit(X_train, y_train)
        rf_preds = rf.predict(X_test)
        rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
        rf_mae = mean_absolute_error(y_test, rf_preds)
        rf_r2 = r2_score(y_test, rf_preds)

    # 3) XGBoost (optional)
    xgb_rmse = xgb_mae = xgb_r2 = None
    if xgb_available:
        with st.spinner("Training XGBoost (default params)..."):
            xgb = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
                               colsample_bytree=0.8, random_state=random_state, tree_method="auto", n_jobs=-1)
            xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
            xgb_preds = xgb.predict(X_test)
            xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_preds))
            xgb_mae = mean_absolute_error(y_test, xgb_preds)
            xgb_r2 = r2_score(y_test, xgb_preds)

    # Display metrics
    st.subheader("Regression results (test set)")
    metrics_df = pd.DataFrame([
        {"Model":"LinearRegression", "RMSE":lr_rmse, "MAE":lr_mae, "R2":lr_r2},
        {"Model":"RandomForest", "RMSE":rf_rmse, "MAE":rf_mae, "R2":rf_r2}
    ])
    if xgb_available:
        metrics_df = pd.concat([
            metrics_df,
            pd.DataFrame([{
                "Model": "XGBoost",
                "RMSE": xgb_rmse,
                "MAE": xgb_mae,
                "R2": xgb_r2
            }])
        ], ignore_index=True)
    st.dataframe(metrics_df)

    # Residual plot and parity plot for best model (choose best by RMSE)
    best_model_name = metrics_df.sort_values('RMSE').iloc[0]['Model']
    st.markdown(f"**Best model by RMSE:** {best_model_name}")

    if best_model_name == "RandomForest":
        best_preds = rf_preds
    elif best_model_name == "XGBoost" and xgb_available:
        best_preds = xgb_preds
    else:
        best_preds = lr_preds

    # Parity (predicted vs actual)
    fig_parity = plt.figure(figsize=(6,6))
    plt.scatter(y_test, best_preds, alpha=0.3)
    maxv = max(y_test.max(), np.max(best_preds))
    plt.plot([0,maxv], [0,maxv], linestyle='--')
    plt.xlabel("Actual CLV")
    plt.ylabel("Predicted CLV")
    plt.title("Parity Plot")
    st.pyplot(fig_parity)

    # Residual
    residuals = y_test - best_preds
    fig_resid = plt.figure(figsize=(8,4))
    sns.histplot(residuals, bins=50, kde=True)
    plt.title("Residuals Distribution (best model)")
    st.pyplot(fig_resid)

    # Feature importance (RandomForest)
    st.subheader("Feature importance (Random Forest)")
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    fig_imp = plt.figure(figsize=(8,6))
    sns.barplot(x=importances.values[:20], y=importances.index[:20])
    plt.title("Top features by importance")
    st.pyplot(fig_imp)
    st.dataframe(importances.reset_index().rename(columns={'index':'feature',0:'importance'}).head(20))

    # Save model optionally
    if st.button("Save trained RandomForest model to disk"):
        joblib.dump(rf, "rf_clv_model.joblib")
        joblib.dump(scaler, "scaler.joblib")
        st.success("Saved rf_clv_model.joblib and scaler.joblib in working directory.")

    st.success("Training and evaluation finished!")

# Additiona EDA & Pareto
st.markdown("---")
st.header("Pareto (80/20) CLV analysis")
clv_df = customer_data[['CustomerID','CLV']].sort_values('CLV', ascending=False).reset_index(drop=True)
clv_df['cum_spend'] = clv_df['CLV'].cumsum()
total_spend = clv_df['CLV'].sum()
clv_df['cum_perc'] = 100 * clv_df['cum_spend'] / total_spend
clv_df['customer_rank'] = np.arange(1, len(clv_df)+1)
clv_df['cum_customers_perc'] = 100 * clv_df['customer_rank'] / len(clv_df)

fig_pareto = plt.figure(figsize=(10,5))
plt.plot(clv_df['cum_customers_perc'], clv_df['cum_perc'])
plt.axhline(80, color='grey', linestyle='--')
plt.axvline(20, color='grey', linestyle='--')
plt.xlabel("Cumulative % of customers")
plt.ylabel("Cumulative % of revenue")
plt.title("Pareto: cumulative customers vs cumulative revenue")
st.pyplot(fig_pareto)

# show top CLV custome table
st.subheader("Top customer by CLV")
st.dataframe(clv_df.head(20))

st.text("end of dasboard.")
