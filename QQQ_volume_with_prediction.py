
#-----------------------------------------------------------------------
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import optuna
import torch
import torch.nn as nn
from torch.optim import Adam
from copy import deepcopy as dc
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device

historical_chart_QQQ = pd.read_pickle('dashboard_data/historical_chart_QQQ_1min.pkl')
historical_chart_QQQ['date'] = pd.to_datetime(historical_chart_QQQ['date'])
historical_chart_QQQ['average'] = (historical_chart_QQQ['open'] + historical_chart_QQQ['close']) / 2
historical_chart_QQQ['dollar_volume'] = historical_chart_QQQ['average'] * historical_chart_QQQ['volume']
historical_chart_QQQ = historical_chart_QQQ.sort_values(by='date')
historical_chart_QQQ['time'] = range(1, len(historical_chart_QQQ) + 1)
historical_chart_QQQ['day_date'] = historical_chart_QQQ['date'].dt.date
historical_chart_QQQ['day_timestamp'] = historical_chart_QQQ.groupby('day_date').cumcount() + 1
historical_chart_QQQ['price'] = historical_chart_QQQ['average']
historical_chart_QQQ = historical_chart_QQQ[['date','day_date', 'dollar_volume', 'price', 'day_timestamp']]




num_minutes_per_day = 390
num_days_for_training = 25
total_training_minutes = num_minutes_per_day * num_days_for_training

# Select the last 25*390 rows for the training data
historical_chart_QQQ_train = historical_chart_QQQ[-total_training_minutes:]
historical_chart_QQQ_train = historical_chart_QQQ_train[['date', 'dollar_volume']]


def prepare_dataframe_for_lstm(df, n_steps):
    df = dc(df)
    
    df.set_index('date', inplace=True)
    
    for i in range(1, n_steps+1):
        df[f'Close(t-{i})'] = df['dollar_volume'].shift(i)
        
    df.dropna(inplace=True)
    
    return df

lookback = 390
shifted_df = prepare_dataframe_for_lstm(historical_chart_QQQ_train, lookback)
val_df = shifted_df[-lookback:]
val_df

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

shifted_df_as_np = shifted_df.to_numpy()
shifted_df_as_np = scaler.fit_transform(shifted_df_as_np)

val_df_as_np = val_df.to_numpy()
val_df_as_np = scaler.fit_transform(val_df_as_np)

x_train = shifted_df_as_np[:, 1:]
x_train = dc(np.flip(x_train, axis=1))
y_train = shifted_df_as_np[:, 0]
x_test = val_df_as_np[:, 1:]
x_test = dc(np.flip(x_test, axis=1))
y_test = val_df_as_np[:, 0]
# x_train.shape, x_test.shape, y_train.shape,y_test.shape

x_train = x_train.reshape((-1, lookback,1))
x_test = x_test.reshape((-1, lookback, 1))
y_train = y_train.reshape((-1,1))
y_test = y_test.reshape((-1,1))
# x_train.shape, x_test.shape, y_train.shape, y_test.shape

x_train = torch.tensor(x_train).float()
x_test = torch.tensor(x_test).float()
y_train = torch.tensor(y_train).float()
y_test = torch.tensor(y_test).float()
# x_train.shape, x_test.shape, y_train.shape, y_test.shape

from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

train_dataset = TimeSeriesDataset(x_train, y_train)
test_dataset = TimeSeriesDataset(x_test, y_test)

from torch.utils.data import DataLoader
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

for _, batch in enumerate(train_loader):
    x_batch, y_batch = batch[0].to(device), batch[1].to(device)
    print(x_batch.shape, y_batch.shape)
    break

def train_one_epoch():
    model.train(True)
    print(f'Epoch: {epoch + 1}')
    running_loss = 0.0

    for batch_index, batch in enumerate(train_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)

        output = model(x_batch)
        loss = loss_function(output, y_batch)
        running_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_index % 100 == 99:  # print every 100 batches
            avg_loss_across_batches = running_loss / 100
            print('Batch {0}, Loss: {1:.3f}'.format(batch_index+1, avg_loss_across_batches))
            running_loss = 0.0

print()

def validate_one_epoch():
    model.train(False)
    running_loss = 0.0

    for batch_index, batch in enumerate(test_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)

        with torch.no_grad():
            output = model(x_batch)
            loss = loss_function(output, y_batch)
            running_loss += loss.item()

    avg_loss_across_batches = running_loss / len(test_loader)

    print('Val Loss: {:.3f}'.format(avg_loss_across_batches))
    print('************************************')
    print()




class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers
        # Update the LSTM layer to be bidirectional
        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers, batch_first=True, bidirectional=True)
        # Update the fully connected layer to handle input from both directions
        # self.fc = nn.Linear(hidden_size * 2, 1)  # Multiply hidden_size by 2 because of bidirectionality
        self.fc = nn.Linear(hidden_size * 2, 180)

    def forward(self, x):
        batch_size = x.size(0)
        # Initialize hidden and cell states for both forward and backward sequences
        h0 = torch.zeros(self.num_stacked_layers * 2, batch_size, self.hidden_size).to(device)  # Multiply by 2 for bidirectionality
        c0 = torch.zeros(self.num_stacked_layers * 2, batch_size, self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        # The output 'out' contains the concatenated hidden states from both directions
        out = self.fc(out[:, -1, :])  # Selecting the last time step output for prediction
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_one_epoch(model, optimizer, loss_function, data_loader, device):
    model.train()  # Set model to training mode
    total_loss = 0
    for batch in data_loader:
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = loss_function(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(data_loader)
    return avg_loss

def validate_one_epoch(model, loss_function, data_loader, device):
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    with torch.no_grad():
        for batch in data_loader:
            x_batch, y_batch = batch[0].to(device), batch[1].to(device)
            outputs = model(x_batch)
            loss = loss_function(outputs, y_batch)
            total_loss += loss.item()
    avg_loss = total_loss / len(data_loader)
    return avg_loss


def objective(trial):
    # Hyperparameters to be tuned
    hidden_size = trial.suggest_categorical('hidden_size', [20, 40, 60, 80, 100, 150, 200])
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)
    num_stacked_layers = trial.suggest_int('num_stacked_layers', 1, 3)  # Corrected

    # Model setup
    model = LSTM(1, hidden_size, num_stacked_layers).to(device)
    loss_function = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Training and validation
    num_epochs = 5
    total_val_loss = 0
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, optimizer, loss_function, train_loader, device)
        val_loss = validate_one_epoch(model, loss_function, test_loader, device)
        total_val_loss += val_loss

        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    avg_val_loss = total_val_loss / num_epochs
    return avg_val_loss

# Assuming train_loader and test_loader are defined elsewhere
# Example of creating and running the study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)  # Adjust the number of trials as needed



def train_one_epoch():
    model.train(True)
    print(f'Epoch: {epoch + 1}')
    running_loss = 0.0

    for batch_index, batch in enumerate(train_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)

        output = model(x_batch)
        loss = loss_function(output, y_batch)
        running_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_index % 100 == 99:  # print every 100 batches
            avg_loss_across_batches = running_loss / 100
            print('Batch {0}, Loss: {1:.3f}'.format(batch_index+1, avg_loss_across_batches))
            running_loss = 0.0
            
def validate_one_epoch():
    model.train(False)
    running_loss = 0.0

    for batch_index, batch in enumerate(test_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)

        with torch.no_grad():
            output = model(x_batch)
            loss = loss_function(output, y_batch)
            running_loss += loss.item()

    avg_loss_across_batches = running_loss / len(test_loader)

    print('Val Loss: {:.3f}'.format(avg_loss_across_batches))
    print('************************************')
    print()


# class BiLSTM(nn.Module):
#     def __init__(self, input_size, hidden_size, num_stacked_layers):
#         super().__init__()
#         self.hidden_size = hidden_size
#         self.num_stacked_layers = num_stacked_layers
#         self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers, batch_first=True, bidirectional=True)
#         self.fc = nn.Linear(hidden_size * 2, 1)  

#     def forward(self, x):
#         batch_size = x.size(0)
#         h0 = torch.zeros(self.num_stacked_layers * 2, batch_size, self.hidden_size).to(device)  
#         c0 = torch.zeros(self.num_stacked_layers * 2, batch_size, self.hidden_size).to(device)
#         out, _ = self.lstm(x, (h0, c0))
#         out = self.fc(out[:, -1, :]) 
#         return out



best_trial = study.best_trial
print(f"Best trial validation loss: {best_trial.value}")
print(f"Best trial parameters: {best_trial.params}")

# Store the best trial's hyperparameters into variables
best_hidden_size = best_trial.params['hidden_size']
best_learning_rate = best_trial.params['learning_rate']
best_weight_decay = best_trial.params['weight_decay']
best_num_stacked_layers = best_trial.params['num_stacked_layers']


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTM(1, best_hidden_size, 1).to(device)
model.to(device)
model


learning_rate = best_learning_rate
num_epochs = 5
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay=best_weight_decay)

for epoch in range(num_epochs):
    train_one_epoch()
    validate_one_epoch()




last_sequence = x_test[-1:]  # This extracts the last sequence with the shape [1, lookback, 1]

# Convert to a tensor and ensure it's of the correct type and device
last_sequence_tensor = torch.tensor(last_sequence).float().to(device)

# Ensure the model is in evaluation mode
model.eval()

# Make the prediction
with torch.no_grad():
    prediction = model(last_sequence_tensor)

prediction

prediction_np = prediction.cpu().numpy()
dummy_array = np.zeros((prediction_np.shape[0], scaler.scale_.shape[0]))

# Place your prediction data into the first 180 columns of the dummy array
dummy_array[:, :prediction_np.shape[1]] = prediction_np

# Now, apply inverse_transform using the dummy array
prediction_rescaled_dummy = scaler.inverse_transform(dummy_array)

# Extract the relevant part of the output (i.e., the first 180 columns, which correspond to your predictions)
prediction_rescaled = prediction_rescaled_dummy[:, :prediction_np.shape[1]]

# Flatten the array if necessary to match your desired output format
prediction_rescaled = prediction_rescaled.flatten()
prediction_rescaled







# make prediction
from datetime import time

num_predictions = 180

# Last timestamp in the original data
last_timestamp = historical_chart_QQQ['date'].max()

# Initialize the list to hold timestamps for the predictions
prediction_timestamps = []

# Helper function to check if the current time is within trading hours
def within_trading_hours(ts):
    return time(9, 30) <= ts.time() <= time(15, 59)

# Helper function to move timestamp to the next trading day's start
def move_to_next_trading_day_start(ts):
    # Add a day
    ts += pd.Timedelta(days=1)
    # Set time to 9:30 AM
    ts = pd.Timestamp(ts.date()).replace(hour=9, minute=30)
    return ts

# Generate timestamps for each prediction
current_timestamp = last_timestamp
for _ in range(num_predictions):
    # Increment by 1 minute
    current_timestamp += pd.Timedelta(minutes=1)
    
    # If it's past trading hours, move to next trading day start
    if not within_trading_hours(current_timestamp) or current_timestamp.weekday() >= 5:  # Saturday=5, Sunday=6
        # Move to next day if it's weekend or past trading hours
        current_timestamp = move_to_next_trading_day_start(current_timestamp)
        
        # Ensure the new timestamp is not on a weekend
        while current_timestamp.weekday() >= 5:  # Skip weekends
            current_timestamp += pd.Timedelta(days=1)
        
    prediction_timestamps.append(current_timestamp)

# Assuming 'prediction_rescaled' contains your volume predictions
predictions_df = pd.DataFrame({
    'date': prediction_timestamps,
    'dollar_volume': prediction_rescaled,  # Assuming you're not predicting volume
    'price': np.nan,
    'is_prediction': 1
})

# Add additional columns to match the original DataFrame
predictions_df['day_date'] = predictions_df['date'].dt.date
predictions_df['day_timestamp'] = predictions_df.groupby('day_date').cumcount() + 1

# Mark original data as not a prediction
historical_chart_QQQ['is_prediction'] = 0

# Concatenate the original DataFrame with the predictions
concatenated_df = pd.concat([historical_chart_QQQ, predictions_df], ignore_index=True)
# concatenated_df.tail(180+3)






# output data 
output_df = concatenated_df.tail(num_days_for_training + num_predictions)
output_df.to_pickle('dashboard_data/QQQ_volume_with_prediction.pkl')
print('QQQ_volume_with_prediction.pkl generated and stored successfully in dashboard_data directory')




#-----------------------------------------------------------------

