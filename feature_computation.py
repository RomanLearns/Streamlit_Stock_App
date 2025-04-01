# Import necessary libraries
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import time
import ta
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from arch import arch_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.decomposition import PCA
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import pickle
import os

# Function to decompose and remove seasonality using STL
def decompose_and_remove_seasonality_stl(series, period, shift_value=None, model='multiplicative'):
    stl = STL(series, period=period, seasonal=13)
    decomposition = stl.fit()
    if model == 'multiplicative':
        deseasonalized = series / decomposition.seasonal
    else:
        deseasonalized = series - decomposition.seasonal
    
    if shift_value is not None:
        deseasonalized_unshifted = deseasonalized - shift_value
    else:
        deseasonalized_unshifted = deseasonalized
    
    return decomposition, deseasonalized_unshifted

# Function to calculate features
def calculate_features():
    sp500_ticker = "^GSPC"
    ibex35_ticker = "^IBEX"
    end_date = "2023-12-29"  # Match the date range shown in the app
    start_date = "2000-01-03"
    
    # Download historical data
    sp500 = pd.read_csv('sp500_data.csv')
    ibex35 = pd.read_csv('ibex35_data.csv')
    sp500['Date'] = pd.to_datetime(sp500['Date'])
    ibex35['Date'] = pd.to_datetime(ibex35['Date'])
    sp500.set_index('Date', inplace=True)
    ibex35.set_index('Date', inplace=True)
    
    # Add technical indicators using ta
    print("Adding technical indicators...")
    ibex35 = ta.add_all_ta_features(ibex35, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)
    sp500 = ta.add_all_ta_features(sp500, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)
    
    # Calculate percentage changes and add date features
    sp500['pct_change'] = sp500['Close'].pct_change() * 100
    sp500['month'] = sp500.index.month
    sp500['day'] = sp500.index.day
    sp500['year'] = sp500.index.year
    ibex35['pct_change'] = ibex35['Close'].pct_change() * 100
    ibex35['month'] = ibex35.index.month
    ibex35['day'] = ibex35.index.day
    ibex35['year'] = ibex35.index.year
    
    sp500 = sp500.dropna()
    ibex35 = ibex35.dropna()
    
    # Get economic data
    FRED_API_KEY = "144e929e4cb64cc8c55902991441556b"
    economic_factors = ["DFF", "T10YIE", "DCOILWTICO", "VIXCLS", "DEXUSEU", "NASDAQCOM", "DHHNGSP",
                        "BAMLC0A0CMEY", "USEPUINDXD", "DPRIME", "DEXCAUS", "DEXDNUS", "DEXJPUS", "DEXCHUS", "DEXSZUS",
                        "DEXUSAL", "DEXUSUK", "DEXKOUS", "INFECTDISEMVTRACKD", "DCOILBRENTEU", "DEXMXUS"]
    rename_map = {"DFF":"daily_fed_funds", "T10YIE":"ten_year_breakeven_inflation", "DCOILWTICO":"WTI_crude_oil_price",
                  "VIXCLS":"VIX", "DEXUSEU":"USD$EUR_spot", "NASDAQCOM":"NASDAQ", "DHHNGSP":"henry_hub_natrual_gas",
                  "BAMLC0A0CMEY":"corporate_bond_yield", "USEPUINDXD":"economic_policy_uncertainty", "DPRIME":"prime_rate",
                  "DEXCAUS":"USD$CAD_spot", "DEXDNUS":"USD$CNY_spot", "DEXJPUS":"USD$JPY_spot", "DEXCHUS":"USD$CHF_spot",
                  "DEXSZUS":"USD$SEK_spot", "DEXUSAL":"USD$AUD_spot", "DEXUSUK":"USD$GBP_spot", "DEXKOUS":"USD$KRW_spot",
                  "INFECTDISEMVTRACKD":"infectious_disease", "DCOILBRENTEU":"Brent_crude_oil_price_europe", "DEXMXUS":"USD$MXN_spot"}
    
    if os.path.exists('economic_data.csv'):
        print("Loading economic data from CSV file...")
        df_total = pd.read_csv('economic_data.csv')
        df_total['date'] = pd.to_datetime(df_total['date'])
    else:
        print("Downloading economic data from FRED...")
        df_total = pd.DataFrame()
        for series in economic_factors:
            curr_start, curr_end = start_date, "2005-01-01"
            series_df = pd.DataFrame()
            while curr_start <= end_date:
                base_series_url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series}&api_key={FRED_API_KEY}&file_type=json&observation_start={curr_start}&observation_end={curr_end}&units=lin&frequency=d"
                response = requests.get(base_series_url)
                if response.status_code != 200:
                    print(f"Failed to fetch FRED data for {series}")
                    continue
                data = response.json()
                df = pd.json_normalize(data)
                observations = pd.json_normalize(df['observations'][0])
                series_df = pd.concat([series_df, observations], axis=0)
                curr_start = (pd.to_datetime(curr_end) + relativedelta(days=1)).strftime("%Y-%m-%d")
                curr_end = min((pd.to_datetime(curr_end) + relativedelta(years=5, days=1), pd.to_datetime(end_date))).strftime("%Y-%m-%d")
                time.sleep(1)
            
            series_df.rename(columns={"date":"date", "value":rename_map[series]}, inplace=True)
            series_df.drop(columns=["realtime_start", "realtime_end"], inplace=True)
            if df_total.empty:
                df_total = series_df
            else:
                df_total = pd.merge(df_total, series_df, on=['date'], how='inner', validate='1:1')
        df_total.to_csv('economic_data.csv', index=False)
    
    # Deseasonalize the data
    shift_value_ibex = abs(ibex35['pct_change'].min()) + 1
    ibex35['pct_change_shifted'] = ibex35['pct_change'] + shift_value_ibex
    shift_value_sp500 = abs(sp500['pct_change'].min()) + 1
    sp500['pct_change_shifted'] = sp500['pct_change'] + shift_value_sp500
    
    ibex_decomp_close_yearly_stl, ibex_close_deseasonalized_yearly_stl = decompose_and_remove_seasonality_stl(
        ibex35['Close'], period=252, shift_value=None, model='additive'
    )
    ibex35['Close_deseasonalized'] = ibex_close_deseasonalized_yearly_stl
    sp500_decomp_close_yearly_stl, sp500_close_deseasonalized_yearly_stl = decompose_and_remove_seasonality_stl(
        sp500['Close'], period=252, shift_value=None, model='additive'
    )
    sp500['Close_deseasonalized'] = sp500_close_deseasonalized_yearly_stl
    
    ibex35['pct_change_deseasonalized'] = ibex35['Close_deseasonalized'].pct_change() * 100
    sp500['pct_change_deseasonalized'] = sp500['Close_deseasonalized'].pct_change() * 100
    
    ibex35 = ibex35.dropna()
    sp500 = sp500.dropna()
    
    # Prepare economic data
    econ = df_total.copy()
    econ.rename(columns={'date':'Date'}, inplace=True)
    econ['Date'] = pd.to_datetime(econ['Date'])
    econ.set_index('Date', inplace=True)
    for col in econ.select_dtypes(include='object'):
        econ[col] = pd.to_numeric(econ[col], errors='coerce')
    
    # Merge data
    df = pd.merge(sp500, ibex35, how='inner', left_index=True, right_index=True, suffixes=["_sp500", "_ibex"], validate='1:1')
    df = pd.merge(df, econ, how='inner', left_index=True, right_index=True, validate='1:1')
    
    # Clean up columns
    pct_change_cols = [col for col in df.columns if 'pct_change' in col and 'deseasonalized' in col]
    df.drop(columns=['Close_sp500', 'Close_ibex'] + pct_change_cols, inplace=True)
    df.rename(columns={
        'Close_deseasonalized_sp500': 'Close_sp500',
        'Close_deseasonalized_ibex': 'Close_ibex',
        'pct_change_sp500': 'SP500_Returns',
        'pct_change_ibex': 'IBEX35_Returns'
    }, inplace=True)
    
    # Deseasonalizing the returns
    df['SP500_Returns'] = df['Close_sp500'].pct_change() * 100
    df['IBEX35_Returns'] = df['Close_ibex'].pct_change() * 100
    
    # Create target variables (next day returns)
    df['SP500_Returns_Next'] = df['SP500_Returns'].shift(-1)
    df['IBEX35_Returns_Next'] = df['IBEX35_Returns'].shift(-1)
    
    # Create lagged features
    for lag in [1, 2, 3, 5, 7, 14]:
        df[f'SP500_Returns_Lag{lag}'] = df['SP500_Returns'].shift(lag)
        df[f'IBEX35_Returns_Lag{lag}'] = df['IBEX35_Returns'].shift(lag)
    
    # Create rolling statistics
    df['SP500_Rolling_Mean_5'] = df['SP500_Returns'].rolling(window=5).mean()
    df['SP500_Rolling_Vol_5'] = df['SP500_Returns'].rolling(window=5).std()
    df['IBEX35_Rolling_Mean_5'] = df['IBEX35_Returns'].rolling(window=5).mean()
    df['IBEX35_Rolling_Vol_5'] = df['IBEX35_Returns'].rolling(window=5).std()
    
    # Add BIAS indicator
    df['SP500_BIAS'] = (df['Close_sp500'] - df['Close_sp500'].rolling(window=12).mean()) / df['Close_sp500'].rolling(window=12).mean() * 100
    df['IBEX35_BIAS'] = (df['Close_ibex'] - df['Close_ibex'].rolling(window=12).mean()) / df['Close_ibex'].rolling(window=12).mean() * 100
    
    # Add lagged economic features
    econ_cols = econ.columns.tolist()
    for col in econ_cols:
        df[f'{col}_Lag1'] = df[col].shift(1)
    
    # Add PSY (Psychological Line Indicator)
    def calculate_psy(price, period=12):
        returns = price.pct_change()
        psy = (returns > 0).rolling(window=period).sum() / period * 100
        return psy
    
    df['SP500_PSY'] = calculate_psy(df['Close_sp500'])
    df['IBEX35_PSY'] = calculate_psy(df['Close_ibex'])
    
    # Fit GARCH model for volatility
    try:
        garch_model_sp500 = arch_model(df['SP500_Returns'].dropna(), vol='Garch', p=1, q=1)
        garch_fit_sp500 = garch_model_sp500.fit(disp='off')
        df['SP500_GARCH_Vol'] = pd.Series(garch_fit_sp500.conditional_volatility, index=df['SP500_Returns'].dropna().index)
        
        garch_model_ibex35 = arch_model(df['IBEX35_Returns'].dropna(), vol='Garch', p=1, q=1)
        garch_fit_ibex35 = garch_model_ibex35.fit(disp='off')
        df['IBEX35_GARCH_Vol'] = pd.Series(garch_fit_ibex35.conditional_volatility, index=df['IBEX35_Returns'].dropna().index)
    except Exception as e:
        print(f"Error fitting GARCH model: {e}. Using rolling volatility instead.")
        df['SP500_GARCH_Vol'] = df['SP500_Returns'].rolling(window=22).std()
        df['IBEX35_GARCH_Vol'] = df['IBEX35_Returns'].rolling(window=22).std()
    
    # Clean up the dataframe
    df = df.dropna(axis=0, how='any')
    # fix scale of data
    
    
    return df, sp500, ibex35

# Feature selection functions
def lasso_feature_selection(X, y, alpha=0.01):
    lasso = Lasso(alpha=alpha)
    lasso.fit(X, y)
    selected_features = X.columns[lasso.coef_ != 0].tolist()
    return selected_features, lasso.coef_

def ridge_feature_selection(X, y, alpha=1.0, top_n=20):
    ridge = Ridge(alpha=alpha)
    ridge.fit(X, y)
    coef_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': np.abs(ridge.coef_)})
    coef_df = coef_df.sort_values(by='Coefficient', ascending=False)
    selected_features = coef_df.head(top_n)['Feature'].tolist()
    return selected_features, ridge.coef_

def pca_feature_selection(X, variance_ratio=0.95):
    pca = PCA()
    pca.fit(X)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cumulative_variance >= variance_ratio) + 1
    selected_components = [f'PC{i+1}' for i in range(n_components)]
    X_pca = pd.DataFrame(pca.transform(X)[:, :n_components], columns=selected_components, index=X.index)
    return X_pca, pca

def xgboost_feature_selection(X, y, top_n=20):
    xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
    xgb_model.fit(X, y)
    
    importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': xgb_model.feature_importances_})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    selected_features = importance_df.head(top_n)['Feature'].tolist()
    
    return selected_features, importance_df

def evaluate_feature_selection(X_train, X_test, y_train, y_test, method_name, selected_features=None, X_train_transformed=None, X_test_transformed=None):
    if X_train_transformed is not None and X_test_transformed is not None:
        X_train_selected = X_train_transformed
        X_test_selected = X_test_transformed
    else:
        X_train_selected = X_train[selected_features]
        X_test_selected = X_test[selected_features]

    model = LinearRegression()
    model.fit(X_train_selected, y_train)
    y_pred = model.predict(X_test_selected)
    
    mse = mean_squared_error(y_test, y_pred)
    return mse, y_pred

# Main execution in Jupyter Notebook
# Step 1: Calculate features
df, sp500_raw, ibex35_raw = calculate_features()

overall_scaler = StandardScaler()
# Step 2: Prepare features and targets
X = df.drop(['Close_sp500', 'Close_ibex', 'SP500_Returns', 'IBEX35_Returns',
             'SP500_Returns_Next', 'IBEX35_Returns_Next'], axis=1)
# Ensure all columns are numeric before scaling
for col in X.select_dtypes(include='object'):
    X[col] = pd.to_numeric(X[col], errors='coerce')
X = X.dropna(axis=1) # Drop columns that became all NaN after coercion, if any

X_scaled = pd.DataFrame(overall_scaler.fit_transform(X), columns=X.columns, index=X.index)


y_sp500 = df['SP500_Returns_Next']
y_ibex = df['IBEX35_Returns_Next']

# Split data using the scaled features
X_train, X_test, y_train_sp, y_test_sp, y_train_ib, y_test_ib = train_test_split(
    X_scaled, y_sp500, y_ibex, test_size=0.2, random_state=42
)

# Step 3: Perform feature selection for S&P 500
print("Feature Selection for S&P 500 Returns_Next")
print("-" * 40)

# LASSO for S&P 500
lasso_features_sp, lasso_coef_sp = lasso_feature_selection(X_train, y_train_sp, alpha=0.0001)
print(f"LASSO Selected Features (S&P 500): {lasso_features_sp}")
if lasso_features_sp: # Check if list is not empty
    mse_lasso_sp, y_pred_lasso_sp = evaluate_feature_selection(X_train, X_test, y_train_sp, y_test_sp, "LASSO", selected_features=lasso_features_sp)
    print(f"LASSO MSE (S&P 500): {mse_lasso_sp:.6f}")
else:
    mse_lasso_sp = np.inf
    print("LASSO selected no features.")


# Ridge for S&P 500
ridge_features_sp, ridge_coef_sp = ridge_feature_selection(X_train, y_train_sp, alpha=1.0, top_n=20) # Reduced top_n
print(f"Ridge Selected Features (Top 20, S&P 500): {ridge_features_sp}")
mse_ridge_sp, y_pred_ridge_sp = evaluate_feature_selection(X_train, X_test, y_train_sp, y_test_sp, "Ridge", selected_features=ridge_features_sp)
print(f"Ridge MSE (S&P 500): {mse_ridge_sp:.6f}")

# PCA for S&P 500
X_pca_sp, pca_sp = pca_feature_selection(X_train, variance_ratio=0.95)
X_train_pca_sp = X_pca_sp # Already indexed correctly by train_test_split output
X_test_pca_sp = pd.DataFrame(pca_sp.transform(X_test)[:, :X_train_pca_sp.shape[1]], columns=X_train_pca_sp.columns, index=X_test.index)
print(f"PCA Components (S&P 500, 95% variance): {X_train_pca_sp.shape[1]} components")
mse_pca_sp, y_pred_pca_sp = evaluate_feature_selection(X_train, X_test, y_train_sp, y_test_sp, "PCA", X_train_transformed=X_train_pca_sp, X_test_transformed=X_test_pca_sp)
print(f"PCA MSE (S&P 500): {mse_pca_sp:.6f}")

# XGBoost for S&P 500
xgb_features_sp, xgb_importance_sp_df = xgboost_feature_selection(X_train, y_train_sp, top_n=20) # Reduced top_n
print(f"XGBoost Selected Features (Top 20, S&P 500): {xgb_features_sp}")
print("\nFeature Importance Scores (S&P 500):")
print(xgb_importance_sp_df.head(15))
mse_xgb_sp, y_pred_xgb_sp = evaluate_feature_selection(X_train, X_test, y_train_sp, y_test_sp, "XGBoost", selected_features=xgb_features_sp)
print(f"XGBoost MSE (S&P 500): {mse_xgb_sp:.6f}")

# Step 4: Perform feature selection for IBEX 35
print("\nFeature Selection for IBEX 35 Returns_Next")
print("-" * 40)

# LASSO for IBEX 35
lasso_features_ib, lasso_coef_ib = lasso_feature_selection(X_train, y_train_ib, alpha=0.0001)
print(f"LASSO Selected Features (IBEX 35): {lasso_features_ib}")
if lasso_features_ib:
    mse_lasso_ib, y_pred_lasso_ib = evaluate_feature_selection(X_train, X_test, y_train_ib, y_test_ib, "LASSO", selected_features=lasso_features_ib)
    print(f"LASSO MSE (IBEX 35): {mse_lasso_ib:.6f}")
else:
    mse_lasso_ib = np.inf
    print("LASSO selected no features.")


# Ridge for IBEX 35
ridge_features_ib, ridge_coef_ib = ridge_feature_selection(X_train, y_train_ib, alpha=1.0, top_n=20) # Reduced top_n
print(f"Ridge Selected Features (Top 20, IBEX 35): {ridge_features_ib}")
mse_ridge_ib, y_pred_ridge_ib = evaluate_feature_selection(X_train, X_test, y_train_ib, y_test_ib, "Ridge", selected_features=ridge_features_ib)
print(f"Ridge MSE (IBEX 35): {mse_ridge_ib:.6f}")

# PCA for IBEX 35
X_pca_ib, pca_ib = pca_feature_selection(X_train, variance_ratio=0.95)
X_train_pca_ib = X_pca_ib # Already indexed correctly
X_test_pca_ib = pd.DataFrame(pca_ib.transform(X_test)[:, :X_train_pca_ib.shape[1]], columns=X_train_pca_ib.columns, index=X_test.index)
print(f"PCA Components (IBEX 35, 95% variance): {X_train_pca_ib.shape[1]} components")
mse_pca_ib, y_pred_pca_ib = evaluate_feature_selection(X_train, X_test, y_train_ib, y_test_ib, "PCA", X_train_transformed=X_train_pca_ib, X_test_transformed=X_test_pca_ib)
print(f"PCA MSE (IBEX 35): {mse_pca_ib:.6f}")

# XGBoost for IBEX 35
xgb_features_ib, xgb_importance_ib_df = xgboost_feature_selection(X_train, y_train_ib, top_n=20) # Reduced top_n
print(f"XGBoost Selected Features (Top 20, IBEX 35): {xgb_features_ib}")
print("\nFeature Importance Scores (IBEX 35):")
print(xgb_importance_ib_df.head(15))
mse_xgb_ib, y_pred_xgb_ib = evaluate_feature_selection(X_train, X_test, y_train_ib, y_test_ib, "XGBoost", selected_features=xgb_features_ib)
print(f"XGBoost MSE (IBEX 35): {mse_xgb_ib:.6f}")

# Step 5: Compare the methods
print("\nSummary of Feature Selection Methods")
print("-" * 40)
print("S&P 500:")
print(f"LASSO MSE: {mse_lasso_sp:.6f}")
print(f"Ridge MSE: {mse_ridge_sp:.6f}")
print(f"PCA MSE: {mse_pca_sp:.6f}")
print(f"XGBoost MSE: {mse_xgb_sp:.6f}")
print("\nIBEX 35:")
print(f"LASSO MSE: {mse_lasso_ib:.6f}")
print(f"Ridge MSE: {mse_ridge_ib:.6f}")
print(f"PCA MSE: {mse_pca_ib:.6f}")
print(f"XGBoost MSE: {mse_xgb_ib:.6f}")

# Determine the best method for each target
best_method_sp = min([("LASSO", mse_lasso_sp), ("Ridge", mse_ridge_sp), ("PCA", mse_pca_sp), ("XGBoost", mse_xgb_sp)], key=lambda x: x[1])[0]
best_method_ib = min([("LASSO", mse_lasso_ib), ("Ridge", mse_ridge_ib), ("PCA", mse_pca_ib), ("XGBoost", mse_xgb_ib)], key=lambda x: x[1])[0]
print(f"\nBest method for S&P 500: {best_method_sp}")
print(f"Best method for IBEX 35: {best_method_ib}")

# Step 6: Apply the best feature selection method
# For S&P 500
if best_method_sp == "LASSO":
    selected_features_sp = lasso_features_sp
    X_selected_sp = X_scaled[selected_features_sp] # Use scaled X
    importance_data_sp = dict(zip(lasso_features_sp, lasso_coef_sp[lasso.coef_ != 0])) # Get coefficients for selected features
elif best_method_sp == "Ridge":
    selected_features_sp = ridge_features_sp
    X_selected_sp = X_scaled[selected_features_sp] # Use scaled X
    # Get coefficients for selected features
    ridge_coef_dict = dict(zip(X_train.columns, ridge_coef_sp))
    importance_data_sp = {feat: ridge_coef_dict[feat] for feat in selected_features_sp}
elif best_method_sp == "PCA":
    selected_features_sp = X_train_pca_sp.columns.tolist()
    X_selected_sp = pd.DataFrame(pca_sp.transform(X_scaled)[:, :X_train_pca_sp.shape[1]], columns=selected_features_sp, index=X_scaled.index) # Transform scaled X
    importance_data_sp = dict(zip(selected_features_sp, pca_sp.explained_variance_ratio_[:X_train_pca_sp.shape[1]])) # Use variance ratio as importance
else:  # XGBoost
    selected_features_sp = xgb_features_sp
    X_selected_sp = X_scaled[selected_features_sp] # Use scaled X
    # Convert DataFrame to dictionary
    importance_data_sp = xgb_importance_sp_df.set_index('Feature')['Importance'].to_dict()


# For IBEX 35
if best_method_ib == "LASSO":
    selected_features_ib = lasso_features_ib
    X_selected_ib = X_scaled[selected_features_ib] # Use scaled X
    importance_data_ib = dict(zip(lasso_features_ib, lasso_coef_ib[lasso.coef_ != 0]))
elif best_method_ib == "Ridge":
    selected_features_ib = ridge_features_ib
    X_selected_ib = X_scaled[selected_features_ib] # Use scaled X
    ridge_coef_dict_ib = dict(zip(X_train.columns, ridge_coef_ib))
    importance_data_ib = {feat: ridge_coef_dict_ib[feat] for feat in selected_features_ib}
elif best_method_ib == "PCA":
    selected_features_ib = X_train_pca_ib.columns.tolist()
    X_selected_ib = pd.DataFrame(pca_ib.transform(X_scaled)[:, :X_train_pca_ib.shape[1]], columns=selected_features_ib, index=X_scaled.index) # Transform scaled X
    importance_data_ib = dict(zip(selected_features_ib, pca_ib.explained_variance_ratio_[:X_train_pca_ib.shape[1]]))
else:  # XGBoost
    selected_features_ib = xgb_features_ib
    X_selected_ib = X_scaled[selected_features_ib] # Use scaled X
    # Convert DataFrame to dictionary
    importance_data_ib = xgb_importance_ib_df.set_index('Feature')['Importance'].to_dict()


# Step 7: Scale the selected features (Already done conceptually, just save the selected DFs)
# We've already selected from the scaled data (X_scaled)
# So X_selected_sp and X_selected_ib are the final dataframes to use for training
# Save these directly

X_selected_sp.to_csv("selected_features_sp500.csv")
X_selected_ib.to_csv("selected_features_ibex35.csv")


# Step 8: Save the feature lists and scalers (Save scalers used for the *selected* features)

# Create and fit scalers *only* for the selected features
scaler_sp = StandardScaler()
scaler_ib = StandardScaler()

# Fit scalers on the selected, unscaled training data to avoid data leakage from test set
scaler_sp.fit(X_train[selected_features_sp]) # Fit on selected columns of original train split
scaler_ib.fit(X_train[selected_features_ib]) # Fit on selected columns of original train split


with open("selected_features_sp500.pkl", "wb") as f:
    pickle.dump(selected_features_sp, f)
with open("selected_features_ibex35.pkl", "wb") as f:
    pickle.dump(selected_features_ib, f)
with open("scaler_sp500.pkl", "wb") as f:
    pickle.dump(scaler_sp, f) # Save the scaler fitted on selected features
with open("scaler_ibex35.pkl", "wb") as f:
    pickle.dump(scaler_ib, f) # Save the scaler fitted on selected features

# Save feature importance dictionary for visualization in the app
with open("rf_importance_sp500.pkl", "wb") as f:
    pickle.dump(importance_data_sp, f) # Save the dictionary

with open("rf_importance_ibex35.pkl", "wb") as f:
    pickle.dump(importance_data_ib, f) # Save the dictionary


print("Feature selection completed and saved to 'selected_features_sp500.csv', 'selected_features_ibex35.csv', 'selected_features_sp500.pkl', 'selected_features_ibex35.pkl', 'scaler_sp500.pkl', 'scaler_ibex35.pkl', 'rf_importance_sp500.pkl', and 'rf_importance_ibex35.pkl'.")