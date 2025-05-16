
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)

## Part 1: Tensor Operations (20 points)

def tensor_operations():
    """
    Perform various tensor operations as specified in Part 1 of the assignment.
    Returns a dictionary containing all the required tensors and results.
    """
    # Create tensors
    random_tensor = torch.randn(3, 4)  # 3x4 tensor with normal distribution
    ones_tensor = torch.ones(3, 4)     # 3x4 tensor filled with ones
    zeros_tensor = torch.zeros(3, 4)   # 3x4 tensor filled with zeros
    
    # Perform operations
    add_result = random_tensor + ones_tensor  # Element-wise addition
    mul_result = random_tensor * ones_tensor  # Element-wise multiplication
    mean_rows = torch.mean(random_tensor, dim=1)  # Mean along each row
    reshaped_tensor = random_tensor.reshape(6, 2)  # Reshaped to 6x2
    
    return {
        'random_tensor': random_tensor,
        'ones_tensor': ones_tensor,
        'zeros_tensor': zeros_tensor,
        'add_result': add_result,
        'mul_result': mul_result,
        'mean_rows': mean_rows,
        'reshaped_tensor': reshaped_tensor
    }

## Part 2: Custom Dataset Creation (30 points)

class TemperatureDataset(Dataset):
    """
    Custom Dataset class for temperature conversion between Celsius and Fahrenheit.
    """
    
    def __init__(self, num_samples=1000, min_temp=-40, max_temp=40):
        """
        Initialize the dataset with random temperature values in Celsius.
        
        Args:
            num_samples (int): Number of samples to generate
            min_temp (float): Minimum temperature in Celsius
            max_temp (float): Maximum temperature in Celsius
        """
        super().__init__()
        # Generate random Celsius temperatures
        self.celsius = torch.FloatTensor(num_samples).uniform_(min_temp, max_temp)
        # Convert to Fahrenheit
        self.fahrenheit = (self.celsius * 9/5) + 32
        
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.celsius)
    
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            tuple: (celsius_temperature, fahrenheit_temperature)
        """
        return self.celsius[idx].unsqueeze(0), self.fahrenheit[idx].unsqueeze(0)

def create_dataloaders():
    """
    Create training and validation dataloaders from the custom dataset.
    
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Create full dataset
    dataset = TemperatureDataset(num_samples=1000)
    
    # Split into training and validation sets (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    return train_loader, val_loader

## Part 3: Neural Network Implementation (50 points)

class TemperatureModel(nn.Module):
    """
    Neural network for temperature conversion from Celsius to Fahrenheit.
    """
    
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 10),  # Input layer (1 neuron) to hidden layer (10 neurons)
            nn.ReLU(),         # ReLU activation
            nn.Linear(10, 1)   # Hidden layer to output layer (1 neuron)
        )
    
    def forward(self, x):
        """Forward pass through the network."""
        return self.layers(x)

def train_model(train_loader, val_loader, num_epochs=100):
    """
    Train the temperature conversion model.
    
    Args:
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        num_epochs (int): Number of training epochs
        
    Returns:
        tuple: (model, train_losses, val_losses)
    """
    # Initialize model, loss function, and optimizer
    model = TemperatureModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    
    # Track losses
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        
        # Calculate average training loss for the epoch
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                epoch_val_loss += loss.item()
        
        # Calculate average validation loss for the epoch
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {avg_train_loss:.4f}, '
                  f'Val Loss: {avg_val_loss:.4f}')
    
    # Plot training curve
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Save the trained model
    torch.save(model.state_dict(), 'temperature_model.pth')
    print("Model saved to 'temperature_model.pth'")
    
    return model, train_losses, val_losses

def evaluate_model(model):
    """
    Evaluate the trained model on new temperature values.
    
    Args:
        model (nn.Module): Trained temperature conversion model
    """
    # Generate 5 new test temperatures
    test_celsius = torch.FloatTensor([-20, -10, 0, 15, 30]).unsqueeze(1)
    test_fahrenheit = (test_celsius * 9/5) + 32
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        pred_fahrenheit = model(test_celsius)
    
    # Calculate absolute errors
    absolute_errors = torch.abs(pred_fahrenheit - test_fahrenheit)
    mean_absolute_error = torch.mean(absolute_errors)
    
    # Print results
    print("\nModel Evaluation on New Data:")
    print(f"{'Celsius':>8} {'Actual F':>8} {'Predicted F':>12} {'Error':>8}")
    for i in range(len(test_celsius)):
        print(f"{test_celsius[i].item():8.1f} "
              f"{test_fahrenheit[i].item():8.1f} "
              f"{pred_fahrenheit[i].item():12.1f} "
              f"{absolute_errors[i].item():8.1f}")
    
    print(f"\nMean Absolute Error: {mean_absolute_error.item():.2f}°F")

def main():
    # Part 1: Tensor Operations
    print("\n=== Part 1: Tensor Operations ===")
    tensor_results = tensor_operations()
    print("Random tensor:\n", tensor_results['random_tensor'])
    print("\nRandom tensor + Ones tensor:\n", tensor_results['add_result'])
    print("\nMean along each row of random tensor:", tensor_results['mean_rows'])
    
    # Part 2: Dataset and DataLoader creation
    print("\n=== Part 2: Dataset Creation ===")
    train_loader, val_loader = create_dataloaders()
    print(f"Created dataloaders with {len(train_loader.dataset)} training samples "
          f"and {len(val_loader.dataset)} validation samples")
    
    # Part 3: Neural Network training and evaluation
    print("\n=== Part 3: Neural Network Training ===")
    model, train_losses, val_losses = train_model(train_loader, val_loader)
    
    # Evaluate the model
    evaluate_model(model)

if __name__ == "__main__":
    main()