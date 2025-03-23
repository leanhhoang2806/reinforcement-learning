import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Clear GPU memory
torch.cuda.empty_cache()

# Set seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the LSTM model
activations = []


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate):
        super(LSTMModel, self).__init__()
        self.lstm_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for i in range(num_layers):
            lstm_layer = nn.LSTM(
                input_size if i == 0 else hidden_size * 2,
                hidden_size,
                num_layers=1,
                batch_first=True,
                dropout=dropout_rate if num_layers > 1 else 0,
                bidirectional=True
            )
            self.lstm_layers.append(lstm_layer)
            self.batch_norms.append(nn.BatchNorm1d(hidden_size * 2))  # BatchNorm for stability
        self.fc = nn.Linear(hidden_size * 2, 1)  # Adjust for bidirectionality
        self.leaky_relu = nn.LeakyReLU(0.01)  # Replace ReLU with LeakyReLU
    
    def forward(self, x):
        for i, (lstm, batch_norm) in enumerate(zip(self.lstm_layers, self.batch_norms)):
            x, _ = lstm(x)
            x = self.leaky_relu(x)  # Apply LeakyReLU after each LSTM layer
            x = batch_norm(x.permute(0, 2, 1)).permute(0, 2, 1)  # Apply BatchNorm
            activations.append(x.detach().cpu().numpy())
        x = x[:, -1, :]
        return torch.sigmoid(self.fc(x))

# Generate dataset for odd/even classification
def generate_odd_even_data(num_samples, sequence_length):
    sequences = np.random.randint(0, 2, (num_samples, sequence_length))
    labels = np.sum(sequences, axis=1) % 2
    return (
        torch.tensor(sequences, dtype=torch.float32).unsqueeze(-1).to(device), 
        torch.tensor(labels, dtype=torch.float32).unsqueeze(-1).to(device)
    )

# Function to train model
def train_model(model, x_train, y_train, x_val, y_val, epochs, batch_size, criterion, optimizer, clip_value, pbar):
    loss_history, val_loss_history = [], []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = len(x_train) // batch_size + (len(x_train) % batch_size != 0)
        
        for i in range(0, len(x_train), batch_size):
            x_batch = x_train[i:i+batch_size].to(device)
            y_batch = y_train[i:i+batch_size].to(device)
            output = model(x_batch)
            loss = criterion(output, y_batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()
            total_loss += loss.item()
            pbar.update(1)
        
        loss_history.append(total_loss / num_batches)
        model.eval()
        with torch.no_grad():
            val_outputs = model(x_val.to(device))
            val_loss = criterion(val_outputs, y_val.to(device)).item()
        val_loss_history.append(val_loss)
    return loss_history, val_loss_history

# Function to plot losses
def plot_losses(losses_dict):
    plt.figure(figsize=(10, 6))
    for seq_len, (train_loss, val_loss) in losses_dict.items():
        plt.plot(train_loss, label=f"Train Seq {seq_len}")
        plt.plot(val_loss, linestyle='dashed', label=f"Val Seq {seq_len}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training & Validation Loss Across Sequence Lengths")
    plt.grid()
    plt.show()

# Define parameter sets
params = {
    "sequence_lengths": [10],
    "hidden_sizes": [512 // 2],
    "num_layers": [3],
    "dropouts": [0.05],  # Reduced dropout
    "epochs": 100,
    "batch_size": 16,
    "clip_value": 5.0,
    "learning_rates": [0.001]
}

losses_dict = {}
total_iterations = sum(params["epochs"] * (1000 // params["batch_size"] + (1000 % params["batch_size"] != 0)) for _ in params["sequence_lengths"])

with tqdm(total=total_iterations, desc="Overall Training Progress", unit="batch") as pbar:
    for i, seq_len in enumerate(params["sequence_lengths"]):
        print(f"Training model with sequence length {seq_len}")
        x_train, y_train = generate_odd_even_data(1000, seq_len)
        x_val, y_val = generate_odd_even_data(200, seq_len)
        model = LSTMModel(input_size=1, hidden_size=params["hidden_sizes"][i],
                          num_layers=params["num_layers"][i], dropout_rate=params["dropouts"][i]).to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=params["learning_rates"][i])
        train_loss, val_loss = train_model(model, x_train, y_train, x_val, y_val, 
                                           epochs=params["epochs"], batch_size=params["batch_size"], 
                                           criterion=criterion, optimizer=optimizer, 
                                           clip_value=params["clip_value"], pbar=pbar)
        losses_dict[seq_len] = (train_loss, val_loss)

plot_losses(losses_dict)
