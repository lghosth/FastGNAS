# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import json
from typing import List, Dict, Any, Tuple

# ==============================================================================
# 1. Config Module: Defines properties for GNN arch, graph, and hardware
# ==============================================================================
class Config:
    """
    Manages all configurable options and parameters.
    """
    # GNN Architecture Configs
    L_MAX = 5  # Max number of GNN layers

    # Options for categorical features
    ACTIVATION_TYPES = ["relu", "sigmoid", "tanh", "none"]
    ATTENTION_TYPES = ["gat", "gcn", "none"] # 'gcn' is a special case with no attention
    AGGREGATION_TYPES = ["sum", "mean", "max", "mlp"]

    # Names of numerical features (used by preprocessor)
    ARCH_NUMERICAL_FEATURES = ["attention_heads", "hidden_units"]
    GRAPH_NUMERICAL_FEATURES = ["vertices", "edges", "avg_degree", "diameter"]
    DEVICE_NUMERICAL_FEATURES = ["gpu_cores", "core_frequency", "fp_capability"]

    # Creates maps from category names to indices for one-hot encoding
    @staticmethod
    def get_category_maps() -> Dict[str, Dict[str, int]]:
        return {
            "activation": {name: i for i, name in enumerate(Config.ACTIVATION_TYPES)},
            "attention": {name: i for i, name in enumerate(Config.ATTENTION_TYPES)},
            "aggregation": {name: i for i, name in enumerate(Config.AGGREGATION_TYPES)},
        }

# ==============================================================================
# 2. Data Preprocessing Module: Converts raw data to model input tensors
# ==============================================================================
class DataPreprocessor:
    """
    Handles the conversion of raw data samples into input tensors for the neural network.
    Manages one-hot encoding, normalization, and saving/loading of normalization parameters.
    """
    def __init__(self, config: Config):
        self.config = config
        self.category_maps = config.get_category_maps()
        self.normalization_params = {} # to store min and max for numerical features

    def _one_hot_encode(self, value: str, category_type: str) -> List[float]:
        num_classes = len(self.category_maps[category_type])
        encoding = [0.0] * num_classes
        if value in self.category_maps[category_type]:
            encoding[self.category_maps[category_type][value]] = 1.0
        return encoding

    def _normalize(self, value: float, feature_name: str) -> float:
        if feature_name not in self.normalization_params:
            return value
        min_val, max_val = self.normalization_params[feature_name]
        if max_val == min_val:
            return 0.5
        return (value - min_val) / (max_val - min_val)

    def fit(self, samples: List[Dict[str, Any]]):
        print("Fitting preprocessor on training data...")
        numerical_data = {key: [] for key in 
                          self.config.ARCH_NUMERICAL_FEATURES + 
                          self.config.GRAPH_NUMERICAL_FEATURES + 
                          self.config.DEVICE_NUMERICAL_FEATURES}
        for sample in samples:
            for layer in sample['architecture']:
                for key in self.config.ARCH_NUMERICAL_FEATURES:
                    numerical_data[key].append(layer.get(key, 0))
            for key in self.config.GRAPH_NUMERICAL_FEATURES:
                numerical_data[key].append(sample['graph'].get(key, 0))
            for key in self.config.DEVICE_NUMERICAL_FEATURES:
                numerical_data[key].append(sample['device'].get(key, 0))
        
        for key, values in numerical_data.items():
            if values:
                self.normalization_params[key] = (min(values), max(values))
        print("Preprocessor fitted.")

    def transform_single(self, sample: Dict[str, Any]) -> torch.Tensor:
        x_arc_list = []
        num_layers = len(sample['architecture'])
        for i in range(self.config.L_MAX):
            if i < num_layers:
                layer = sample['architecture'][i]
                act_enc = self._one_hot_encode(layer['activation'], 'activation')
                att_enc = self._one_hot_encode(layer['attention_type'], 'attention')
                agg_enc = self._one_hot_encode(layer['aggregation_type'], 'aggregation')
                heads_norm = self._normalize(layer['attention_heads'], 'attention_heads')
                hidden_norm = self._normalize(layer['hidden_units'], 'hidden_units')
                x_arc_list.extend(act_enc + att_enc + agg_enc + [heads_norm, hidden_norm])
            else:
                layer_feature_dim = (len(self.config.ACTIVATION_TYPES) +
                                     len(self.config.ATTENTION_TYPES) +
                                     len(self.config.AGGREGATION_TYPES) +
                                     len(self.config.ARCH_NUMERICAL_FEATURES))
                x_arc_list.extend([0.0] * layer_feature_dim)
        
        x_arc = torch.tensor(x_arc_list, dtype=torch.float32)
        x_gra_list = [self._normalize(sample['graph'][key], key) for key in self.config.GRAPH_NUMERICAL_FEATURES]
        x_gra = torch.tensor(x_gra_list, dtype=torch.float32)
        x_dev_list = [self._normalize(sample['device'][key], key) for key in self.config.DEVICE_NUMERICAL_FEATURES]
        x_dev = torch.tensor(x_dev_list, dtype=torch.float32)
        return torch.cat([x_arc, x_gra, x_dev])

    def fit_transform(self, samples: List[Dict[str, Any]], labels: List[float]) -> Tuple[torch.Tensor, torch.Tensor]:
        self.fit(samples)
        transformed_samples = [self.transform_single(s) for s in samples]
        X = torch.stack(transformed_samples)
        y = torch.tensor(labels, dtype=torch.float32).view(-1, 1)
        return X, y
    
    def get_state(self) -> Dict[str, Any]:
        return {'normalization_params': self.normalization_params}

    def set_state(self, state: Dict[str, Any]):
        self.normalization_params = state['normalization_params']

# ==============================================================================
# 3. Neural Network Model Module
# ==============================================================================
class MloadModel(nn.Module):
    """
    A simple neural network model for predicting GNN workload.
    """
    def __init__(self, input_dim: int):
        super(MloadModel, self).__init__()
        self.input_dim = input_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

# ==============================================================================
# 4. Public Predictor Interface Class
# ==============================================================================
class MloadPredictor:
    """
    An interface class providing GNN load prediction services. 
    It encapsulates model loading and preprocessing logic, offering a simple prediction method.
    """
    def __init__(self, model: MloadModel, preprocessor: DataPreprocessor):
        """
        Constructor is internally called by the 'load_from_file' class method.
        """
        self._model = model
        self._preprocessor = preprocessor
        self._model.eval() # Switch to evaluation mode

    @classmethod
    def load_from_file(cls, filepath: str, config: Config = Config()):
        """
        Loads the model and preprocessor from a file and creates a predictor instance.
        This is the recommended way to create an MloadPredictor instance.

        :param filepath: Path to the model checkpoint file (.pth).
        :param config: A Config instance.
        :return: An MloadPredictor instance, or None if the file does not exist.
        """
        try:
            checkpoint = torch.load(filepath)
            
            # 1. Restore model
            input_dim = checkpoint['input_dim']
            model = MloadModel(input_dim)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # 2. Restore preprocessor
            preprocessor = DataPreprocessor(config)
            preprocessor.set_state(checkpoint['preprocessor_state'])
            
            print(f"Predictor loaded successfully from {filepath}")
            return cls(model, preprocessor)
        except FileNotFoundError:
            print(f"Error: Checkpoint file not found at {filepath}")
            return None
            
    def predict(self, sample: Dict[str, Any]) -> float:
        """
        Predicts the load for a single GNN configuration sample.

        :param sample: A dictionary containing 'architecture', 'graph', and 'device' keys.
        :return: The predicted runtime (float).
        """
        # 1. Transform input data using the internal preprocessor
        input_tensor = self._preprocessor.transform_single(sample)
        
        # 2. The model expects a batch dimension, so we add one
        input_tensor = input_tensor.unsqueeze(0) # [features] -> [1, features]

        # 3. Perform prediction in a no-gradient context
        with torch.no_grad():
            predicted_value = self._model(input_tensor)
            
        # 4. Extract the float value from the tensor and return it
        return predicted_value.item()

# ==============================================================================
# 5. Training and Utility Functions
# ==============================================================================
def train_model(model: MloadModel, X_train: torch.Tensor, y_train: torch.Tensor, epochs: int = 200, learning_rate: float = 0.001):
    print(f"\nStarting training for {epochs} epochs...")
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        model.train()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')
    print("Training finished.")

def save_checkpoint(model: MloadModel, preprocessor: DataPreprocessor, filepath: str):
    checkpoint = {
        'input_dim': model.input_dim,
        'model_state_dict': model.state_dict(),
        'preprocessor_state': preprocessor.get_state(),
    }
    torch.save(checkpoint, filepath)
    print(f"Model and preprocessor saved to {filepath}")



if __name__ == "__main__":
    # Mock Data
    p100_device_info = {
        "gpu_cores": 3584, 
        "core_frequency": 1.328, 
        "fp_capability": 6.0
    }
    a6000_device_info = {
        "gpu_cores": 10752, 
        "core_frequency": 2.1, 
        "fp_capability": 8.6
    }
    sample_data = [
        {
            "architecture": [
                {"activation": "relu", "attention_type": "gat", "aggregation_type": "sum", "attention_heads": 8, "hidden_units": 128},
                {"activation": "relu", "attention_type": "gat", "aggregation_type": "sum", "attention_heads": 8, "hidden_units": 64}
            ],
            "graph": {"vertices": 1000, "edges": 5000, "avg_degree": 5.0, "diameter": 10},
            "device": p100_device_info
        },
        {
            "architecture": [
                {"activation": "tanh", "attention_type": "gcn", "aggregation_type": "mean", "attention_heads": 1, "hidden_units": 256}
            ],
            "graph": {"vertices": 50000, "edges": 200000, "avg_degree": 4.0, "diameter": 25},
            "device": p100_device_info
        },
        {
            "architecture": [
                {"activation": "relu", "attention_type": "none", "aggregation_type": "max", "attention_heads": 1, "hidden_units": 64},
                {"activation": "relu", "attention_type": "none", "aggregation_type": "max", "attention_heads": 1, "hidden_units": 64},
                {"activation": "relu", "attention_type": "none", "aggregation_type": "max", "attention_heads": 1, "hidden_units": 32}
            ],
            "graph": {"vertices": 2000, "edges": 8000, "avg_degree": 4.0, "diameter": 12},
            "device": p100_device_info
        },
        {
            "architecture": [
                {"activation": "sigmoid", "attention_type": "gat", "aggregation_type": "mlp", "attention_heads": 16, "hidden_units": 512}
            ],
            "graph": {"vertices": 100000, "edges": 1000000, "avg_degree": 10.0, "diameter": 15},
            "device": p100_device_info
        },
        {
            "architecture": [
                {"activation": "relu", "attention_type": "gcn", "aggregation_type": "mean", "attention_heads": 1, "hidden_units": 128},
                {"activation": "relu", "attention_type": "gcn", "aggregation_type": "mean", "attention_heads": 1, "hidden_units": 128},
                {"activation": "relu", "attention_type": "gcn", "aggregation_type": "mean", "attention_heads": 1, "hidden_units": 128},
                {"activation": "relu", "attention_type": "gcn", "aggregation_type": "mean", "attention_heads": 1, "hidden_units": 128}
            ],
            "graph": {"vertices": 80000, "edges": 400000, "avg_degree": 5.0, "diameter": 30},
            "device": p100_device_info
        },
    ]
    labels = [15.5, 80.2, 25.0, 250.8, 120.3]
    
    # Initialization and Training
    config_instance = Config()
    preprocessor = DataPreprocessor(config_instance)
    X_train_tensor, y_train_tensor = preprocessor.fit_transform(sample_data, labels)
    input_dimension = X_train_tensor.shape[1]
    model = MloadModel(input_dim=input_dimension)
    train_model(model, X_train_tensor, y_train_tensor, epochs=500)
    
    # Save
    MODEL_FILE_PATH = "mload_checkpoint.pth"
    save_checkpoint(model, preprocessor, MODEL_FILE_PATH)
    
    print("\n" + "="*50)
    print("DEMONSTRATING PREDICTION INTERFACE")
    print("="*50 + "\n")
    
    # --- Load model and create predictor using MloadPredictor class ---
    # This is what a user would do to use the service.
    predictor = MloadPredictor.load_from_file(MODEL_FILE_PATH)
    
    # --- Use the predictor for prediction ---
    if predictor:
        # Create a new sample data
        new_sample = {
            "architecture": [
                {"activation": "relu", "attention_type": "gat", "aggregation_type": "mean", "attention_heads": 4, "hidden_units": 200},
                {"activation": "tanh", "attention_type": "none", "aggregation_type": "sum", "attention_heads": 1, "hidden_units": 50}
            ],
            "graph": {"vertices": 30000, "edges": 150000, "avg_degree": 5.0, "diameter": 20},
            "device": {"gpu_cores": 5120, "core_frequency": 1.8, "fp_capability": 7.5}
        }
        
        # Call the predict method with the dictionary
        predicted_runtime = predictor.predict(new_sample)
        
        print(f"New Sample Configuration:\n{json.dumps(new_sample, indent=2)}")
        print(f"Predicted Runtime: {predicted_runtime:.2f} seconds")

        # Predict on another sample
        another_sample = sample_data[0] # Use a sample from the training set for validation
        predicted_runtime_2 = predictor.predict(another_sample)
        print(f"\nPredicting on a known sample (first training sample):")
        print(f"Predicted Runtime: {predicted_runtime_2:.2f} seconds (Actual was: {labels[0]})")
