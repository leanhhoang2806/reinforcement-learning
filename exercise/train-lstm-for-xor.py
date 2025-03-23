import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
# Detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the LSTM model with regularization
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate=0):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.lstm1 = nn.LSTM(hidden_size, hidden_size // 2, num_layers, batch_first=True, dropout=dropout_rate)
        self.lstm2 = nn.LSTM(hidden_size // 2, hidden_size // 4, num_layers, batch_first=True, dropout=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)  # Additional dropout
        self.fc = nn.Linear(hidden_size // 4, 1)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out, _ = self.lstm1(out)
        out, _ = self.lstm2(out)
        out = self.dropout(out[:, -1, :])  # Apply dropout before final layer
        out = self.fc(out)
        return torch.sigmoid(out)

# Generate binary sequences and labels
def generate_data(num_samples, sequence_length):
    sequences = np.random.randint(0, 2, (num_samples, sequence_length))
    labels = np.sum(sequences, axis=1) % 2  # 0 for even, 1 for odd
    return torch.tensor(sequences, dtype=torch.float32).unsqueeze(-1).to(device), torch.tensor(labels, dtype=torch.float32).unsqueeze(-1).to(device)

# Training function with a single progress bar
def train_model(model, x_train, y_train, x_val, y_val, epochs, batch_size, criterion, optimizer, clip_value=5.0):
    model.to(device)
    loss_history, val_loss_history, grad_norms = [], [], []
    total_steps = epochs * (len(x_train) // batch_size + (len(x_train) % batch_size != 0))
    
    with tqdm(total=total_steps, desc="Training Progress", unit="batch") as pbar:
        for _ in range(epochs):
            model.train()
            total_loss = 0
            num_batches = 0
            epoch_grad_norms = []
            
            for i in range(0, len(x_train), batch_size):
                x_batch = x_train[i:i+batch_size].to(device)
                y_batch = y_train[i:i+batch_size].to(device)
                
                output = model(x_batch)
                loss = criterion(output, y_batch)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)  # Apply gradient clipping
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                # Compute gradient norms
                total_norm = 0
                for param in model.parameters():
                    if param.grad is not None:
                        param_norm = param.grad.norm().item()
                        total_norm += param_norm
                epoch_grad_norms.append(total_norm)
                
                pbar.set_postfix(loss=total_loss / num_batches)
                pbar.update(1)
            
            loss_history.append(total_loss / num_batches)
            grad_norms.append(np.mean(epoch_grad_norms))
            model.eval()
            with torch.no_grad():
                val_outputs = model(x_val.to(device))
                val_loss = criterion(val_outputs, y_val.to(device)).item()
            val_loss_history.append(val_loss)
    
    return loss_history, val_loss_history, grad_norms

# Function to evaluate model
def evaluate_model(model, x_val, y_val):
    model.eval()
    with torch.no_grad():
        outputs = model(x_val.to(device))
        predictions = torch.round(outputs)
        accuracy = (predictions == y_val.to(device)).float().mean().item()
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")
    return accuracy

# Function to plot training loss
def plot_training_loss(training_curves):
    plt.figure(figsize=(10,6))
    for (hidden_size, num_layers), loss_history in training_curves.items():
        plt.plot(loss_history, label=f"Train H{hidden_size}-L{num_layers}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid()
    plt.show()

# Function to plot gradient norms
def plot_gradient_norms(grad_norms):
    plt.figure(figsize=(10,6))
    plt.plot(grad_norms, label="Gradient Norms")
    plt.xlabel("Epochs")
    plt.ylabel("Gradient Norm")
    plt.title("Gradient Norms Over Epochs (Vanishing/Exploding Detection)")
    plt.legend()
    plt.grid()
    plt.show()

# Experiment settings
hidden_sizes = [128 * 2 * 2]
num_layers_list = [20]
loss_results, training_curves, validation_curves = {}, {}, {}
gradient_norms_curves = {}

# Generate dataset
sequence_length = 50
epochs = 25
num_samples = 10000
x_train, y_train = generate_data(num_samples, sequence_length)
x_val, y_val = generate_data(num_samples // 5, sequence_length)

for hidden_size in hidden_sizes:
    for num_layers in num_layers_list:
        print(f"Training model with hidden_size={hidden_size}, num_layers={num_layers}")
        
        model = LSTMModel(input_size=1, hidden_size=hidden_size, num_layers=num_layers).to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-2)  # L2 Regularization (Weight Decay)
        
        loss_history, val_loss_history, grad_norms = train_model(model, x_train, y_train, x_val, y_val, epochs, batch_size=16, criterion=criterion, optimizer=optimizer)
        loss_results[(hidden_size, num_layers)] = loss_history[-1]
        training_curves[(hidden_size, num_layers)] = loss_history
        validation_curves[(hidden_size, num_layers)] = val_loss_history
        gradient_norms_curves[(hidden_size, num_layers)] = grad_norms
        evaluate_model(model, x_val, y_val)

# Plot training loss
plot_training_loss(training_curves)

# Plot gradient norms
plot_gradient_norms(gradient_norms_curves[(hidden_sizes[0], num_layers_list[0])])
