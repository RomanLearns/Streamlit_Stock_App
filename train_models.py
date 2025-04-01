import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
import os
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import torch.optim.lr_scheduler as lr_scheduler

# Model classes
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

        # More complex, potentially beneficial RNN structure
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, nonlinearity='relu')
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # Forward pass through RNN
        out, _ = self.rnn(x, h0)
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

# Function to precompute time-aware scaling parameters
def precompute_scaling_params(data, target, feature_columns):
    feature_means = []
    feature_stds = []
    target_means = []
    target_stds = []
    
    data_features = data[feature_columns].values
    target_values = target.values.reshape(-1, 1)
    
    for i in range(len(data)):
        # Use all data up to index i for scaling
        window_features = data_features[:i + 1]
        window_target = target_values[:i + 1]
        
        # Compute mean and std
        feature_mean = window_features.mean(axis=0)
        feature_std = window_features.std(axis=0)
        target_mean = window_target.mean()
        target_std = window_target.std()
        
        # Handle zero std to avoid division by zero
        feature_std = np.where(feature_std == 0, 1, feature_std)
        target_std = 1 if target_std == 0 else target_std
        
        feature_means.append(feature_mean)
        feature_stds.append(feature_std)
        target_means.append(target_mean)
        target_stds.append(target_std)
    
    return (np.array(feature_means), np.array(feature_stds)), (np.array(target_means), np.array(target_stds))

# Function to create sequences with precomputed scaling parameters
def create_sequences(data, target, seq_length, feature_columns, feature_scaling_params, target_scaling_params):
    X, y = [], []
    data_features = data[feature_columns].values
    target_values = target.values
    
    feature_means, feature_stds = feature_scaling_params
    target_means, target_stds = target_scaling_params
    
    for i in range(len(data) - seq_length):
        # Get the sequence
        seq_features = data_features[i:i + seq_length]
        
        # Normalize using the scaling parameters at the end of the sequence
        idx = i + seq_length - 1
        seq_features_normalized = (seq_features - feature_means[idx]) / feature_stds[idx]
        
        # Normalize the target
        target_val = target_values[i + seq_length]
        target_normalized = (target_val - target_means[idx]) / target_stds[idx]
        
        X.append(seq_features_normalized)
        y.append(target_normalized)
    
    return np.array(X), np.array(y)

# Function to train a model
def train_model(model, X_train, y_train, X_val, y_val, epochs=200, batch_size=64, lr=0.01, patience=50, weight_decay=1e-5, clip=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val).to(device)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train_tensor[i:i + batch_size]
            batch_y = y_train_tensor[i:i + batch_size]
            
            optimizer.zero_grad()
            if isinstance(model, RNNModel):
                outputs = model(batch_X)  # Do not pass h0
            else:
                # Pass h0 forward, updated h0 returned
                # You would expect here to also pass h0, as the previous forward() method for RNNModel has it, but then it is not used in GRU and LSTM, so, let's not
                outputs = model(batch_X)

            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            
            # Gradient Clipping - Add clip to the parameter list
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / (len(X_train) // batch_size + 1)
        
        # Validation
        model.eval()
        with torch.no_grad():
            if isinstance(model, RNNModel):
                val_outputs = model(X_val_tensor) # Pass h0 forward, updated h0 returned
            else:
                val_outputs = model(X_val_tensor)
            
            val_loss = criterion(val_outputs.squeeze(), y_val_tensor)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss.item():.4f}')
        
        if epochs_no_improve >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    # Load the best model state
    model.load_state_dict(best_model_state)
    return model


# Main training script
def main():
    # Load precomputed features
    df_sp500 = pd.read_csv("selected_features_sp500.csv", index_col="Date", parse_dates=True)
    df_ibex35 = pd.read_csv("selected_features_ibex35.csv", index_col="Date", parse_dates=True)
    
    # Load raw data to get the closing prices
    sp500_raw = pd.read_csv("sp500_data.csv", index_col="Date", parse_dates=True)
    ibex35_raw = pd.read_csv("ibex35_data.csv", index_col="Date", parse_dates=True)
    
    # Merge raw data with precomputed features to get the target (Close price)
    df_sp500 = df_sp500.join(sp500_raw['Close'], how='inner')
    df_ibex35 = df_ibex35.join(ibex35_raw['Close'], how='inner')
    
    with open("selected_features_sp500.pkl", "rb") as f:
        feature_columns_sp500 = pickle.load(f)
    with open("selected_features_ibex35.pkl", "rb") as f:
        feature_columns_ibex35 = pickle.load(f)
    
    # Model parameters
    model_params = {
        'input_size': len(feature_columns_sp500),
        'hidden_size': 32,
        'num_layers': 3,
        'output_size': 1,
        'dropout': 0.0,
        'seq_length': 10,  # Kept to capture longer trends
        'ensemble_weights_sp500': [1/3, 1/3, 1/3],
        'ensemble_weights_ibex35': [1/3, 1/3, 1/3]
    }
    
    # Precompute scaling parameters
    print("Precomputing scaling parameters for SP500...")
    feature_scaling_params_sp500, target_scaling_params_sp500 = precompute_scaling_params(df_sp500, df_sp500['Close'], feature_columns_sp500)
    print("Precomputing scaling parameters for IBEX35...")
    feature_scaling_params_ibex35, target_scaling_params_ibex35 = precompute_scaling_params(df_ibex35, df_ibex35['Close'], feature_columns_ibex35)
    
    # Create sequences for SP500
    X_sp500, y_sp500 = create_sequences(df_sp500, df_sp500['Close'], model_params['seq_length'], feature_columns_sp500, feature_scaling_params_sp500, target_scaling_params_sp500)
    train_size = int(0.8 * len(X_sp500))
    X_train_sp500, X_val_sp500 = X_sp500[:train_size], X_sp500[train_size:]
    y_train_sp500, y_val_sp500 = y_sp500[:train_size], y_sp500[train_size:]
    
    # Create sequences for IBEX35
    X_ibex35, y_ibex35 = create_sequences(df_ibex35, df_ibex35['Close'], model_params['seq_length'], feature_columns_ibex35, feature_scaling_params_ibex35, target_scaling_params_ibex35)
    X_train_ibex35, X_val_ibex35 = X_ibex35[:train_size], X_ibex35[train_size:]
    y_train_ibex35, y_val_ibex35 = y_ibex35[:train_size], y_ibex35[train_size:]
    
    # Initialize models
    models = {
        'SP500': {
            'LSTM': LSTMModel(model_params['input_size'], model_params['hidden_size'], model_params['num_layers'], model_params['output_size'], model_params['dropout']),
            'GRU': GRUModel(model_params['input_size'], model_params['hidden_size'], model_params['num_layers'], model_params['output_size'], model_params['dropout']),
            'RNN': RNNModel(model_params['input_size'], model_params['hidden_size'], model_params['num_layers'], model_params['output_size'], model_params['dropout'])
        },
        'IBEX35': {
            'LSTM': LSTMModel(model_params['input_size'], model_params['hidden_size'], model_params['num_layers'], model_params['output_size'], model_params['dropout']),
            'GRU': GRUModel(model_params['input_size'], model_params['hidden_size'], model_params['num_layers'], model_params['output_size'], model_params['dropout']),
            'RNN': RNNModel(model_params['input_size'], model_params['hidden_size'], model_params['num_layers'], model_params['output_size'], model_params['dropout'])
        }
    }
    
    # Train models
    print("Training models for SP500...")
    models['SP500']['LSTM'] = train_model(models['SP500']['LSTM'], X_train_sp500, y_train_sp500, X_val_sp500, y_val_sp500)
    models['SP500']['GRU'] = train_model(models['SP500']['GRU'], X_train_sp500, y_train_sp500, X_val_sp500, y_val_sp500)
    models['SP500']['RNN'] = train_model(models['SP500']['RNN'], X_train_sp500, y_train_sp500, X_val_sp500, y_val_sp500, epochs=1000, patience=100, lr=.0001)
    
    print("Training models for IBEX35...")
    models['IBEX35']['LSTM'] = train_model(models['IBEX35']['LSTM'], X_train_ibex35, y_train_ibex35, X_val_ibex35, y_val_ibex35)
    models['IBEX35']['GRU'] = train_model(models['IBEX35']['GRU'], X_train_ibex35, y_train_ibex35, X_val_ibex35, y_val_ibex35)
    models['IBEX35']['RNN'] = train_model(models['IBEX35']['RNN'], X_train_ibex35, y_train_ibex35, X_val_ibex35, y_val_ibex35, epochs=1000, patience=100, lr=.0001)
    
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Save model weights
    torch.save(models['SP500']['LSTM'].state_dict(), 'models/lstm_sp500.pth')
    torch.save(models['SP500']['GRU'].state_dict(), 'models/gru_sp500.pth')
    torch.save(models['SP500']['RNN'].state_dict(), 'models/rnn_sp500.pth')
    torch.save(models['IBEX35']['LSTM'].state_dict(), 'models/lstm_ibex35.pth')
    torch.save(models['IBEX35']['GRU'].state_dict(), 'models/gru_ibex35.pth')
    torch.save(models['IBEX35']['RNN'].state_dict(), 'models/rnn_ibex35.pth')
    
    # Save model parameters
    with open('models/model_params.pkl', 'wb') as f:
        pickle.dump(model_params, f)
    
    # Save updated feature columns
    with open("selected_features_sp500.pkl", "wb") as f:
        pickle.dump(feature_columns_sp500, f)
    with open("selected_features_ibex35.pkl", "wb") as f:
        pickle.dump(feature_columns_ibex35, f)
    
    # # Save scaling parameters
    with open('models/feature_scaling_params_sp500.pkl', 'wb') as f:
        pickle.dump(feature_scaling_params_sp500, f)
    with open('models/feature_scaling_params_ibex35.pkl', 'wb') as f:
        pickle.dump(feature_scaling_params_ibex35, f)
    with open('models/target_scaling_params_sp500.pkl', 'wb') as f:
        pickle.dump(target_scaling_params_sp500, f)
    with open('models/target_scaling_params_ibex35.pkl', 'wb') as f:
        pickle.dump(target_scaling_params_ibex35, f)
    
    print("Models trained and saved successfully!")

if __name__ == "__main__":
    main()