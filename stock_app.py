import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import pickle
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set page configuration
st.set_page_config(
    page_title="Enhanced Stock Index Prediction App",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #1E88E5; margin-bottom: 1rem; }
    .sub-header { font-size: 1.5rem; font-weight: bold; color: #424242; margin-bottom: 1rem; }
    .metric-card { background-color: #f8f9fa; border-radius: 5px; padding: 1rem; margin-bottom: 1rem; border-left: 5px solid #1E88E5; }
    .metric-value { font-size: 1.8rem; font-weight: bold; color: #1E88E5; }
    .metric-label { font-size: 1rem; color: #424242; }
    .stMetricValue { font-size: 1.8rem !important; }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">Enhanced Stock Index Prediction App</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown('<div class="sub-header">Model Selection</div>', unsafe_allow_html=True)
index_option = st.sidebar.radio("Select Index", ["SP500", "IBEX35"])
model_option = st.sidebar.radio("Select Model", ["LSTM", "GRU", "RNN"])
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    """
    This enhanced app uses deep learning models with precomputed features
    to predict stock index prices.

    Select an index and a model type from the options above to see predictions.

    Data is loaded from precomputed feature files.
    Scaling is applied dynamically based on historical data up to each point.
    """
)

# --- Model Classes (Ensure these match train_models.py EXACTLY) ---
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.3):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.3):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.3):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, nonlinearity='relu')
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out
# --- End Model Classes ---

# --- Data Loading and Preprocessing Functions (Match train_models.py) ---
@st.cache_data(ttl=3600) # Increased TTL as these files don't change often
def load_precomputed_data():
    try:
        # Load raw data
        sp500_raw = pd.read_csv("sp500_data.csv", index_col="Date", parse_dates=True)
        ibex35_raw = pd.read_csv("ibex35_data.csv", index_col="Date", parse_dates=True)

        # Load selected features dataframes (these already have 'Close' merged in train_models.py)
        df_sp500 = pd.read_csv("selected_features_sp500.csv", index_col="Date", parse_dates=True)
        df_ibex35 = pd.read_csv("selected_features_ibex35.csv", index_col="Date", parse_dates=True)
        df_sp500 = df_sp500.join(sp500_raw['Close'], how='inner') # Join Close price here for consistency
        df_ibex35 = df_ibex35.join(ibex35_raw['Close'], how='inner') # Join Close price here for consistency


        # Load feature lists
        with open("selected_features_sp500.pkl", "rb") as f:
            feature_columns_sp500 = pickle.load(f)
        with open("selected_features_ibex35.pkl", "rb") as f:
            feature_columns_ibex35 = pickle.load(f)

        # Load PRECOMPUTED scaling parameters
        with open("models/feature_scaling_params_sp500.pkl", "rb") as f:
            feature_scaling_params_sp500 = pickle.load(f) # Tuple: (means, stds)
        with open("models/feature_scaling_params_ibex35.pkl", "rb") as f:
            feature_scaling_params_ibex35 = pickle.load(f) # Tuple: (means, stds)
        with open("models/target_scaling_params_sp500.pkl", "rb") as f:
            target_scaling_params_sp500 = pickle.load(f)   # Tuple: (means, stds)
        with open("models/target_scaling_params_ibex35.pkl", "rb") as f:
            target_scaling_params_ibex35 = pickle.load(f)   # Tuple: (means, stds)

        # Load feature importance (assuming it's saved correctly as a dict)
        with open("rf_importance_sp500.pkl", "rb") as f:
            rf_importance_sp500 = pickle.load(f)
        with open("rf_importance_ibex35.pkl", "rb") as f:
            rf_importance_ibex35 = pickle.load(f)

        # Basic check for importance format
        if not isinstance(rf_importance_sp500, dict):
             st.warning("SP500 Feature importance data might not be in the expected dictionary format.")
        if not isinstance(rf_importance_ibex35, dict):
             st.warning("IBEX35 Feature importance data might not be in the expected dictionary format.")


        return (df_sp500, df_ibex35, sp500_raw, ibex35_raw,
                feature_columns_sp500, feature_columns_ibex35,
                feature_scaling_params_sp500, feature_scaling_params_ibex35,
                target_scaling_params_sp500, target_scaling_params_ibex35,
                rf_importance_sp500, rf_importance_ibex35)

    except Exception as e:
        st.error(f"Failed to load precomputed data: {e}")
        # Return None for all expected values to prevent downstream errors
        return (None,) * 12 # Adjust count based on return values


# Function to create sequences using PRECOMPUTED scaling parameters (from train_models.py)
def create_sequences_for_inference(data, seq_length, feature_columns, feature_scaling_params):
    """Creates sequences suitable for model inference using precomputed scaling."""
    X = []
    data_features = data[feature_columns].values
    feature_means, feature_stds = feature_scaling_params

    if len(data_features) < seq_length:
         st.error(f"Not enough data ({len(data_features)}) to create sequence of length {seq_length}")
         return np.array([]) # Return empty array

    for i in range(len(data_features) - seq_length + 1):
        # Get the sequence
        seq_features = data_features[i:i + seq_length]

        # Normalize using the scaling parameters relevant to the END of the sequence
        # The index `idx` should correspond to the index in the precomputed means/stds arrays
        # Note: The precomputed arrays should align with the original dataframe index used during precomputation
        idx = data.index.get_loc(data.index[i + seq_length - 1]) # Get integer position

        # Ensure idx is within the bounds of the precomputed scaling parameters
        if idx >= len(feature_means):
             st.warning(f"Scaling parameter index {idx} out of bounds (max: {len(feature_means)-1}). Using last available parameters.")
             idx = len(feature_means) - 1

        seq_features_normalized = (seq_features - feature_means[idx]) / feature_stds[idx]
        X.append(seq_features_normalized)

    return np.array(X)

# --- End Data Loading ---

# --- Model Loading ---
@st.cache_resource # Cache models
def load_models(feature_columns_sp500, feature_columns_ibex35):
    if not os.path.exists('models'):
        st.warning("Models directory not found. Using dummy predictions.")
        return None, None
    
    try:
        with open('models/model_params.pkl', 'rb') as f:
            model_params = pickle.load(f)
    except FileNotFoundError as e:
        st.error(f"Required file missing: {e}. Please run the training script first.")
        return None, None

    # Check if input_size matches the current number of features
    current_input_size_sp500 = len(feature_columns_sp500)
    current_input_size_ibex35 = len(feature_columns_ibex35)
    
    # Handle potential mismatch if model_params doesn't have input_size directly
    # Let's assume the number of features loaded is the correct input size if not in params
    expected_input_size_sp = model_params.get('input_size', current_input_size_sp500)
    expected_input_size_ib = model_params.get('input_size', current_input_size_ibex35)

    if expected_input_size_sp != current_input_size_sp500:
         st.error(f"SP500 input size mismatch: Model trained with {expected_input_size_sp}, current features {current_input_size_sp500}.")
         # Potentially return None or try to proceed if it's just a param file issue
    if expected_input_size_ib != current_input_size_ibex35:
         st.error(f"IBEX35 input size mismatch: Model trained with {expected_input_size_ib}, current features {current_input_size_ibex35}.")
         # Potentially return None

    # Use the determined input sizes for model initialization
    input_size_sp = current_input_size_sp500
    input_size_ib = current_input_size_ibex35


    models = {
        'SP500': {
            'LSTM': LSTMModel(input_size_sp, model_params['hidden_size'], model_params['num_layers'], model_params['output_size'], model_params['dropout']),
            'GRU': GRUModel(input_size_sp, model_params['hidden_size'], model_params['num_layers'], model_params['output_size'], model_params['dropout']),
            'RNN': RNNModel(input_size_sp, model_params['hidden_size'], model_params['num_layers'], model_params['output_size'], model_params['dropout'])
        },
        'IBEX35': {
            'LSTM': LSTMModel(input_size_ib, model_params['hidden_size'], model_params['num_layers'], model_params['output_size'], model_params['dropout']),
            'GRU': GRUModel(input_size_ib, model_params['hidden_size'], model_params['num_layers'], model_params['output_size'], model_params['dropout']),
            'RNN': RNNModel(input_size_ib, model_params['hidden_size'], model_params['num_layers'], model_params['output_size'], model_params['dropout'])
        }
    }
    try:
        # Load state dicts for SP500
        models['SP500']['LSTM'].load_state_dict(torch.load('models/lstm_sp500.pth', map_location=torch.device('cpu')))
        models['SP500']['GRU'].load_state_dict(torch.load('models/gru_sp500.pth', map_location=torch.device('cpu')))
        models['SP500']['RNN'].load_state_dict(torch.load('models/rnn_sp500.pth', map_location=torch.device('cpu')))

        # Load state dicts for IBEX35
        models['IBEX35']['LSTM'].load_state_dict(torch.load('models/lstm_ibex35.pth', map_location=torch.device('cpu')))
        models['IBEX35']['GRU'].load_state_dict(torch.load('models/gru_ibex35.pth', map_location=torch.device('cpu')))
        models['IBEX35']['RNN'].load_state_dict(torch.load('models/rnn_ibex35.pth', map_location=torch.device('cpu')))

    except FileNotFoundError as e:
        st.error(f"Model weight file missing: {e}. Please run the training script first.")
        return None, None
    except RuntimeError as e:
        # More specific error for state dict issues
        st.error(f"Error loading model state_dict: {e}. Architecture mismatch likely. Retrain models or ensure app model definitions match training.")
        return None, None

    for index in models:
        for model_type in models[index]:
            models[index][model_type].eval()

    return models, model_params
# --- End Model Loading ---

# --- Prediction Functions ---
def make_predictions(model, data, feature_scaling_params, target_scaling_params, model_params, feature_columns):
    """Makes predictions on historical data using time-aware scaling."""
    X = create_sequences_for_inference(data, model_params['seq_length'], feature_columns, feature_scaling_params)
    if X.size == 0: # Handle empty sequence case
        return np.array([])

    X_tensor = torch.FloatTensor(X)
    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(X_tensor).numpy() # Shape: (num_sequences, 1)

    # Inverse transform using the CORRECT precomputed target scaling parameters
    target_means, target_stds = target_scaling_params
    # Align the scaling parameters with the predictions
    # The scaling params for prediction i correspond to index i + seq_length - 1
    start_idx = model_params['seq_length'] - 1
    end_idx = start_idx + len(y_pred_scaled)

    # Ensure indices are valid
    if end_idx > len(target_means):
        st.error(f"Target scaling parameter index out of bounds during prediction. Required: {end_idx}, Available: {len(target_means)}")
        # Truncate prediction if necessary
        y_pred_scaled = y_pred_scaled[:len(target_means) - start_idx]
        end_idx = len(target_means)
        if start_idx >= end_idx:
             return np.array([]) # Cannot make predictions

    relevant_target_means = target_means[start_idx:end_idx]
    relevant_target_stds = target_stds[start_idx:end_idx]

    # Reshape means and stds to allow broadcasting with (num_sequences, 1) output
    y_pred = y_pred_scaled * relevant_target_stds.reshape(-1, 1) + relevant_target_means.reshape(-1, 1)

    return y_pred.flatten()


def make_future_prediction(model, data, feature_scaling_params, target_scaling_params, model_params, feature_columns):
    """Makes prediction for the next day using the last sequence and scaling params."""
    seq_length = model_params['seq_length']
    if len(data) < seq_length:
        st.error("Not enough data to create the last sequence for future prediction.")
        return None

    # Get the last sequence of actual features
    last_sequence_features = data[feature_columns].values[-seq_length:]

    # Normalize using the LAST precomputed scaling parameters
    feature_means, feature_stds = feature_scaling_params
    last_feature_mean = feature_means[-1]
    last_feature_std = feature_stds[-1]
    last_sequence_normalized = (last_sequence_features - last_feature_mean) / last_feature_std

    # Convert to tensor
    X_tensor = torch.FloatTensor(last_sequence_normalized).unsqueeze(0) # Add batch dimension

    # Predict
    model.eval()
    with torch.no_grad():
        next_day_pred_scaled = model(X_tensor).numpy()

    # Inverse transform using the LAST precomputed target scaling parameters
    target_means, target_stds = target_scaling_params
    last_target_mean = target_means[-1]
    last_target_std = target_stds[-1]
    next_day_pred = (next_day_pred_scaled * last_target_std + last_target_mean).flatten()[0]

    return next_day_pred

# (Keep calculate_metrics and calculate_confidence_interval as they are)
def calculate_metrics(y_true, y_pred):
    mse_value = mean_squared_error(y_true, y_pred)  # Use unique variable name
    rmse_value = np.sqrt(mse_value)
    mae_value = mean_absolute_error(y_true, y_pred)
    r2_value = r2_score(y_true, y_pred)
    return {'MSE': mse_value, 'RMSE': rmse_value, 'MAE': mae_value, 'R2': r2_value}

def calculate_confidence_interval(y_true, y_pred, future_pred, days=30):
    # Ensure arrays are aligned
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    # Calculate residuals (actual - predicted)
    residuals = y_true - y_pred

    # Initialize arrays for lower and upper bounds of the confidence interval
    # ci_lower = np.zeros(len(y_pred)) # Not needed for future only CI
    # ci_upper = np.zeros(len(y_pred))

    # Compute CI for the future prediction using the last `days` residuals
    if len(residuals) == 0:
        st.warning("Cannot calculate confidence interval: No historical residuals available.")
        return future_pred * 0.95, future_pred * 1.05 # Fallback

    last_residuals = residuals[-days:] if len(residuals) >= days else residuals
    if len(last_residuals) < 2:
         st.warning(f"Cannot calculate confidence interval: Not enough residuals ({len(last_residuals)}). Using fallback.")
         return future_pred * 0.95, future_pred * 1.05 # Fallback

    mean_residual = np.mean(last_residuals)
    std_residual = np.std(last_residuals)
    future_ci_lower = future_pred + mean_residual - 1.96 * std_residual
    future_ci_upper = future_pred + mean_residual + 1.96 * std_residual

    return future_ci_lower, future_ci_upper
# --- End Prediction Functions ---


# --- Main Execution ---
try:
    # Load all precomputed data at once
    with st.spinner("Loading precomputed data and models..."):
        (df_sp500, df_ibex35, sp500_raw, ibex35_raw,
         feature_columns_sp500, feature_columns_ibex35,
         feature_scaling_params_sp500, feature_scaling_params_ibex35,
         target_scaling_params_sp500, target_scaling_params_ibex35,
         rf_importance_sp500, rf_importance_ibex35) = load_precomputed_data()

        # Check if loading failed
        if df_sp500 is None:
             raise ValueError("Data loading failed, cannot proceed.")

        # Load models (pass feature columns for input size validation)
        models, model_params = load_models(feature_columns_sp500, feature_columns_ibex35)

        if models is None or model_params is None:
             # Fallback to dummy mode if model loading failed
             st.warning("Model loading failed. Falling back to dummy predictions.")
             models = None # Ensure models is None for the dummy logic below


    # Select data based on user choice
    if index_option == "SP500":
        df = df_sp500
        feature_columns = feature_columns_sp500
        feature_scaling_params = feature_scaling_params_sp500
        target_scaling_params = target_scaling_params_sp500
        rf_importance = rf_importance_sp500
        raw_data = sp500_raw
    else: # IBEX35
        df = df_ibex35
        feature_columns = feature_columns_ibex35
        feature_scaling_params = feature_scaling_params_ibex35
        target_scaling_params = target_scaling_params_ibex35
        rf_importance = rf_importance_ibex35
        raw_data = ibex35_raw

    target_col = 'Close' # Target is always 'Close' now

    # If models are not found OR model loading failed earlier, set up dummy parameters
    if models is None:
        st.warning("Running in demo mode with dummy predictions.")
        # Use loaded feature columns length if available, else a default
        input_size_dummy = len(feature_columns) if feature_columns else 50
        model_params = {
            'input_size': input_size_dummy, # Use actual feature count if possible
            'hidden_size': 64,
            'num_layers': 2,
            'output_size': 1,
            'dropout': 0.3,
            'seq_length': 10, # Match training if possible, else use default
        }

    # Display data info
    st.markdown(f'<div class="sub-header">{index_option} Data</div>', unsafe_allow_html=True)
    st.write(f"Data range: {raw_data.index[0].date()} to {raw_data.index[-1].date()}")

    # Plot historical close price
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(raw_data.index, raw_data['Close'], label=f'{index_option} Close Price', color='blue')

    # Calculate next date explicitly
    last_date = raw_data.index[-1]
    next_date = last_date + pd.Timedelta(days=1)

    # Make predictions and calculate metrics
    if models is not None:
        # Select the specific model based on user choice
        selected_model = models[index_option][model_option]

        y_pred = make_predictions(
            selected_model,
            df, # Pass the dataframe with selected features + 'Close'
            feature_scaling_params,
            target_scaling_params,
            model_params,
            feature_columns,
            # target_col is implicit now as it's handled via target_scaling_params
        )

        future_pred = make_future_prediction(
             selected_model,
             df,
             feature_scaling_params,
             target_scaling_params,
             model_params,
             feature_columns
        )

        if y_pred.size > 0 and future_pred is not None:
             # Align predictions with the dates
             # Predictions start from seq_length onwards
             pred_dates = df.index[model_params['seq_length']:] # Corrected index slicing

             # Ensure lengths match before plotting and metrics
             min_len = min(len(pred_dates), len(y_pred))
             pred_dates = pred_dates[:min_len]
             y_pred = y_pred[:min_len]

             ax.plot(pred_dates, y_pred, label=f'{model_option} Predicted Price', color='orange', linestyle='--')
             ax.scatter(next_date, future_pred, color='green', s=100, label=f'Next Day Predicted Price ({next_date.date()})', zorder=5)

             # Calculate metrics using aligned data
             y_true = df['Close'].values[model_params['seq_length']:]
             y_true = y_true[:min_len] # Align y_true as well
             metrics = calculate_metrics(y_true, y_pred)

             # Calculate confidence interval
             ci_lower, ci_upper = calculate_confidence_interval(y_true, y_pred, future_pred, days=30)

        else:
             st.warning("Could not generate predictions. Check data or model parameters.")
             # Set dummy values to avoid errors later
             y_pred = np.array([])
             future_pred = raw_data['Close'].iloc[-1] # Use last known value as fallback
             metrics = {'MSE': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan}
             ci_lower, ci_upper = future_pred * 0.95, future_pred * 1.05

    else: # Dummy mode
        seq_length_dummy = model_params['seq_length']
        pred_len = len(df) - seq_length_dummy
        if pred_len > 0:
            y_pred = df['Close'].values[seq_length_dummy:] * np.random.uniform(0.95, 1.05, size=pred_len)
            pred_dates = df.index[seq_length_dummy:]
            ax.plot(pred_dates, y_pred, label='Dummy Predicted Price', color='orange', linestyle='--')
        else:
            y_pred = np.array([])

        future_pred = raw_data['Close'].iloc[-1] * np.random.uniform(0.95, 1.05)
        ax.scatter(next_date, future_pred, color='green', s=100, label=f'Next Day Predicted Price ({next_date.date()})', zorder=5)

        if pred_len > 0 :
            y_true = df['Close'].values[seq_length_dummy:]
            metrics = calculate_metrics(y_true, y_pred)
        else:
             metrics = {'MSE': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan}

        ci_lower, ci_upper = future_pred * 0.95, future_pred * 1.05


    # --- Plotting and Metrics Display (largely unchanged, but ensure variables exist) ---
    # Finalize the graph
    ax.set_title(f'{index_option} Historical and Predicted Close Price')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Display model performance metrics
    st.markdown('<div class="sub-header">Model Performance Metrics</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    # Add checks for NaN metrics in case of errors
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Mean Squared Error</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{metrics["MSE"]:.2f}</div>' if not np.isnan(metrics["MSE"]) else '<div class="metric-value">N/A</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    # ... similar checks for RMSE, MAE, R2 ...
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Root Mean Squared Error</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{metrics["RMSE"]:.2f}</div>' if not np.isnan(metrics["RMSE"]) else '<div class="metric-value">N/A</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Mean Absolute Error</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{metrics["MAE"]:.2f}</div>' if not np.isnan(metrics["MAE"]) else '<div class="metric-value">N/A</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">R² Score</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{metrics["R2"]:.4f}</div>' if not np.isnan(metrics["R2"]) else '<div class="metric-value">N/A</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Display latest and next day prediction
    st.markdown('<div class="sub-header">Latest and Next Day Prediction</div>', unsafe_allow_html=True)
    last_actual = raw_data['Close'].iloc[-1]
    # Handle case where y_pred might be empty
    last_pred = y_pred[-1] if len(y_pred) > 0 else np.nan
    error = ((last_pred - last_actual) / last_actual) * 100 if not np.isnan(last_pred) else np.nan

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-label">Latest Actual ({last_date.date()})</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{last_actual:.2f}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-label">Latest Predicted</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{last_pred:.2f}</div>' if not np.isnan(last_pred) else '<div class="metric-value">N/A</div>', unsafe_allow_html=True)
        if not np.isnan(error):
            st.markdown(f'<div class="metric-label">{"↓" if error < 0 else "↑"} {abs(error):.2f}%</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-label">Next Day Prediction ({next_date.date()})</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{future_pred:.2f}</div>' if future_pred is not None else '<div class="metric-value">N/A</div>', unsafe_allow_html=True)
        if future_pred is not None:
             change = ((future_pred - last_actual) / last_actual) * 100
             st.markdown(f'<div class="metric-label">{"↑" if change > 0 else "↓"} {abs(change):.2f}%</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-label">95% Confidence Interval</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{ci_lower:.2f} - {ci_upper:.2f}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Feature Importance Display (Keep the structure, ensure rf_importance is a dict)
    st.markdown('<div class="sub-header">Feature Importance Analysis</div>', unsafe_allow_html=True)
    if isinstance(rf_importance, dict) and rf_importance: # Check if it's a non-empty dict
        # Convert importance DataFrame/Series potentially saved back to dict if needed
        if isinstance(rf_importance, (pd.Series, pd.DataFrame)):
             # Assuming DataFrame has 'Feature' and 'Importance' columns
             # Or Series index is 'Feature' and values are 'Importance'
             try:
                 if isinstance(rf_importance, pd.DataFrame):
                     rf_importance = rf_importance.set_index('Feature')['Importance'].to_dict()
                 else: # Is Series
                     rf_importance = rf_importance.to_dict()
             except Exception as imp_e:
                  st.error(f"Could not process feature importance data: {imp_e}")
                  rf_importance = {} # Reset to empty dict

        if rf_importance: # Proceed if conversion was successful or already a dict
            feature_importance_df = pd.DataFrame({
                'Feature': list(rf_importance.keys()),
                'Importance': list(rf_importance.values())
            }).sort_values(by='Importance', ascending=False).head(15) # Show top 15 descending

            fig_imp, ax_imp = plt.subplots(figsize=(10, 8)) # Adjusted size
            ax_imp.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
            ax_imp.set_title(f'Top 15 Feature Importance for {index_option} Prediction')
            ax_imp.set_xlabel('Importance')
            ax_imp.set_ylabel('Feature')
            plt.gca().invert_yaxis() # Show most important at the top
            st.pyplot(fig_imp)
        else:
              st.info("No feature importance data available to display.")

    else:
        st.info("Feature importance data not available or not in the expected format.")


    # Option to show raw data
    if st.checkbox("Show Raw Data"):
        st.markdown('<div class="sub-header">Raw Data</div>', unsafe_allow_html=True)
        st.dataframe(raw_data)

except Exception as e:
    st.error(f"An error occurred during app execution: {e}")
    st.exception(e) # Provides more detailed traceback in the app for debugging