import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedShuffleSplit
import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from sklearn.preprocessing import StandardScaler
import matplotlib
from scipy.signal import savgol_filter
from scipy import linalg

# 修复matplotlib后端问题
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import traceback
import os

# 导入深度学习相关库
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

warnings.filterwarnings('ignore')


class SpectrumPreprocessor:
    """Spectrum data preprocessing class"""

    @staticmethod
    def sg_smoothing(X, window_length=11, polyorder=2):
        """Savitzky-Golay smoothing"""
        return savgol_filter(X, window_length, polyorder, axis=1)

    @staticmethod
    def first_derivative(X):
        """First derivative"""
        return np.gradient(X, axis=1)

    @staticmethod
    def snv(X):
        """Standard Normal Variate transformation"""
        X_snv = np.zeros_like(X)
        for i in range(X.shape[0]):
            spectrum = X[i, :]
            mean_val = np.mean(spectrum)
            std_val = np.std(spectrum)
            if std_val > 0:
                X_snv[i, :] = (spectrum - mean_val) / std_val
            else:
                X_snv[i, :] = spectrum - mean_val
        return X_snv

    @staticmethod
    def msc(X):
        """Multiplicative Scatter Correction"""
        mean_spectrum = np.mean(X, axis=0)
        X_msc = np.zeros_like(X)
        for i in range(X.shape[0]):
            A = np.vstack([mean_spectrum, np.ones(len(mean_spectrum))]).T
            try:
                result = linalg.lstsq(A, X[i, :])
                slope, intercept = result[0]
            except TypeError:
                result = linalg.lstsq(A, X[i, :])
                slope, intercept = result[0]
            if abs(slope) > 1e-10:
                X_msc[i, :] = (X[i, :] - intercept) / slope
            else:
                X_msc[i, :] = X[i, :] - intercept
        return X_msc

    @staticmethod
    def snv_d1st(X):
        """SNV + First derivative"""
        X_snv = SpectrumPreprocessor.snv(X)
        return SpectrumPreprocessor.first_derivative(X_snv)

    @staticmethod
    def apply_preprocessing(X, method='snv_d1st'):
        """Apply specified preprocessing method"""
        preprocess_methods = {
            'snv_d1st': SpectrumPreprocessor.snv_d1st
        }

        if method not in preprocess_methods:
            raise ValueError(f"Unsupported preprocessing method: {method}")

        print(f"Applying preprocessing method: {method}")
        return preprocess_methods[method](X)


class SpectrumDataLoader:
    """Data Loader (renamed to avoid conflict with PyTorch's DataLoader)"""

    @staticmethod
    def load_spectrum_data(file_path):
        """Load spectrum data
        New Excel format: First two columns are Variety and Maturity, third column is GETI, followed by spectrum data
        """
        print(f"Loading spectrum data: {file_path}")
        spectrum_data = pd.read_excel(file_path, header=0)
        print(f"Spectrum data shape: {spectrum_data.shape}")

        # Extract variety, maturity, and sample ID
        variety_data = spectrum_data.iloc[:, 0].values
        maturity_data = spectrum_data.iloc[:, 1].values
        sample_ids = spectrum_data.iloc[:, 2].values
        X = spectrum_data.iloc[:, 3:].values
        wavelength_names = list(spectrum_data.columns[3:])

        print(f"Number of samples: {len(sample_ids)}")
        print(f"Number of wavelengths: {len(wavelength_names)}")
        print(f"Number of variety types: {len(np.unique(variety_data))}")
        print(f"Number of maturity types: {len(np.unique(maturity_data))}")

        return X, sample_ids, wavelength_names, variety_data, maturity_data

    @staticmethod
    def load_catechin_data(file_path):
        """Load catechin data
        New Excel format: First two columns are Variety and Maturity, third column is GETI, followed by catechin data
        """
        print(f"Loading catechin data: {file_path}")
        catechin_data = pd.read_excel(file_path, header=0)
        print(f"Catechin data shape: {catechin_data.shape}")

        # Extract variety, maturity, and sample ID
        variety_data = catechin_data.iloc[:, 0].values
        maturity_data = catechin_data.iloc[:, 1].values
        sample_ids = catechin_data.iloc[:, 2].values
        catechin_names = list(catechin_data.columns[3:])
        y = catechin_data.iloc[:, 3:].values

        print(f"Number of samples: {len(sample_ids)}")
        print(f"Number of catechin types: {len(catechin_names)}")
        print(f"Catechin names: {catechin_names}")
        print(f"Number of variety types: {len(np.unique(variety_data))}")
        print(f"Number of maturity types: {len(np.unique(maturity_data))}")

        return y, sample_ids, catechin_names, variety_data, maturity_data

    @staticmethod
    def align_data_by_catechin(spectrum_ids, spectrum_X, spectrum_variety, spectrum_maturity,
                               catechin_ids, catechin_y, catechin_variety, catechin_maturity):
        """Align data based on catechin data
        Only keep spectrum data corresponding to GETI present in catechin data
        """
        print("\nAligning data based on catechin data...")
        print(f"Number of catechin samples: {len(catechin_ids)}")
        print(f"Number of spectrum samples: {len(spectrum_ids)}")

        # Create mapping dictionary for spectrum data for quick lookup
        spectrum_dict = {}
        for i, geti in enumerate(spectrum_ids):
            if geti not in spectrum_dict:
                spectrum_dict[geti] = {
                    'index': i,
                    'X': spectrum_X[i],
                    'variety': spectrum_variety[i],
                    'maturity': spectrum_maturity[i]
                }
            else:
                print(f"Warning: GETI {geti} appears multiple times in spectrum data")

        aligned_X = []
        aligned_y = []
        aligned_ids = []
        aligned_variety = []
        aligned_maturity = []
        missing_count = 0

        # Use catechin data as baseline
        for i, geti in enumerate(catechin_ids):
            if geti in spectrum_dict:
                spectrum_info = spectrum_dict[geti]

                # Verify if variety and maturity match
                if catechin_variety[i] != spectrum_info['variety']:
                    print(
                        f"Warning: Sample {geti} variety mismatch (catechin: {catechin_variety[i]}, spectrum: {spectrum_info['variety']})")
                    continue

                if catechin_maturity[i] != spectrum_info['maturity']:
                    print(
                        f"Warning: Sample {geti} maturity mismatch (catechin: {catechin_maturity[i]}, spectrum: {spectrum_info['maturity']})")
                    continue

                aligned_X.append(spectrum_info['X'])
                aligned_y.append(catechin_y[i])
                aligned_ids.append(geti)
                aligned_variety.append(catechin_variety[i])
                aligned_maturity.append(catechin_maturity[i])
            else:
                missing_count += 1
                print(f"Warning: Catechin sample {geti} not found in spectrum data")

        aligned_X = np.array(aligned_X)
        aligned_y = np.array(aligned_y)
        aligned_ids = np.array(aligned_ids)
        aligned_variety = np.array(aligned_variety)
        aligned_maturity = np.array(aligned_maturity)

        print(f"Successfully aligned samples: {len(aligned_ids)}")
        print(f"Catechin samples without corresponding spectrum data: {missing_count}")

        if len(aligned_ids) == 0:
            raise ValueError("No common sample IDs found, please check data files!")

        # 检查并处理NaN值
        nan_mask_X = np.any(np.isnan(aligned_X), axis=1)
        nan_mask_y = np.any(np.isnan(aligned_y), axis=1)
        nan_mask = np.logical_or(nan_mask_X, nan_mask_y)

        if np.any(nan_mask):
            print(f"Warning: Found {np.sum(nan_mask)} samples with NaN values. Removing these samples.")
            aligned_X = aligned_X[~nan_mask]
            aligned_y = aligned_y[~nan_mask]
            aligned_ids = aligned_ids[~nan_mask]
            aligned_variety = aligned_variety[~nan_mask]
            aligned_maturity = aligned_maturity[~nan_mask]

        print(f"Aligned data shape (after removing NaN): X={aligned_X.shape}, y={aligned_y.shape}")
        print(f"Aligned variety types: {np.unique(aligned_variety)}")
        print(f"Aligned maturity types: {np.unique(aligned_maturity)}")

        return aligned_X, aligned_y, aligned_ids, aligned_variety, aligned_maturity


class NeuralNetworkModel(nn.Module):
    """Deep learning neural network model"""

    def __init__(self, input_size, hidden_sizes=[256, 128, 64], dropout_rate=0.2):
        super(NeuralNetworkModel, self).__init__()

        layers = []
        prev_size = input_size

        # Build hidden layers
        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, 1))

        self.model = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        """Weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.model(x)


class DLRegression:
    """Deep learning regression model"""

    class Scaler:
        """Data standardization class"""

        def __init__(self):
            self._mean = 0
            self._std = 0

        def fit_transform(self, traindata):
            self._mean = traindata.mean(axis=0)
            self._std = traindata.std(axis=0)
            return (traindata - self._mean) / (self._std + 1e-8)

        def transform(self, testdata):
            return (testdata - self._mean) / (self._std + 1e-8)

        def inverse_transform(self, scaled_data):
            """Transform scaled data back to original scale"""
            return scaled_data * (self._std + 1e-8) + self._mean

    def __init__(self, hidden_sizes=[256, 128, 64], dropout_rate=0.2,
                 learning_rate=0.001, batch_size=32, epochs=500,
                 patience=30, weight_decay=1e-4, device=None):

        self.hidden_sizes = hidden_sizes
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.weight_decay = weight_decay

        # Device selection
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.model = None
        self.X_scaler = self.Scaler()
        self.y_scaler = self.Scaler()
        self.loss_history = {'train': [], 'val': []}

    def fit(self, X, y, verbose=True):
        """Train deep learning model"""
        # 检查输入数据是否有NaN
        if np.any(np.isnan(X)) or np.any(np.isnan(y)):
            print("Warning: Input data contains NaN values. Replacing NaN with column means.")
            X = np.where(np.isnan(X), np.nanmean(X, axis=0), X)
            y = np.where(np.isnan(y), np.nanmean(y, axis=0), y)

        # Data standardization
        X_scaled = self.X_scaler.fit_transform(X)
        y_scaled = self.y_scaler.fit_transform(y)

        # 再次检查标准化后的数据
        if np.any(np.isnan(X_scaled)) or np.any(np.isnan(y_scaled)):
            print("Warning: Scaled data contains NaN. Replacing with 0.")
            X_scaled = np.where(np.isnan(X_scaled), 0, X_scaled)
            y_scaled = np.where(np.isnan(y_scaled), 0, y_scaled)

        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.FloatTensor(y_scaled).to(self.device)

        # Initialize model
        input_size = X.shape[1]
        self.model = NeuralNetworkModel(
            input_size=input_size,
            hidden_sizes=self.hidden_sizes,
            dropout_rate=self.dropout_rate
        ).to(self.device)

        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )

        # Early stopping setup
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        # Split training set into training and validation
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        split_idx = int(0.8 * len(X))

        train_idx = indices[:split_idx]
        val_idx = indices[split_idx:]

        X_train = X_tensor[train_idx]
        y_train = y_tensor[train_idx]
        X_val = X_tensor[val_idx]
        y_val = y_tensor[val_idx]

        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)

        train_loader = TorchDataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = TorchDataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        if verbose:
            print(f"Training device: {self.device}")
            print(f"Training set size: {len(train_idx)}, Validation set size: {len(val_idx)}")
            print(f"Network structure: Input({input_size}) -> ", end="")
            for i, size in enumerate(self.hidden_sizes):
                print(f"Hidden{i + 1}({size}) -> ", end="")
            print("Output(1)")
            print("Starting training...")

        # Training loop
        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_batches = 0

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_batches += 1

            avg_train_loss = train_loss / train_batches
            self.loss_history['train'].append(avg_train_loss)

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_batches = 0

            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    val_batches += 1

            avg_val_loss = val_loss / val_batches
            self.loss_history['val'].append(avg_val_loss)

            # Learning rate adjustment
            scheduler.step(avg_val_loss)

            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            # Print training progress
            if verbose and (epoch + 1) % 50 == 0:
                print(f"Epoch [{epoch + 1:4d}/{self.epochs}] | "
                      f"Train Loss: {avg_train_loss:.6f} | "
                      f"Val Loss: {avg_val_loss:.6f} | "
                      f"LR: {optimizer.param_groups[0]['lr']:.6f}")

            # Check early stopping
            if patience_counter >= self.patience:
                if verbose:
                    print(f"Early stopping triggered at Epoch {epoch + 1}")
                break

        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        if verbose:
            print(f"Training completed, best validation loss: {best_val_loss:.6f}")

        return self

    def predict(self, X_test):
        """Prediction"""
        if self.model is None:
            raise ValueError("Model not trained, please call fit method first")

        # 检查输入数据是否有NaN
        if np.any(np.isnan(X_test)):
            print("Warning: Test data contains NaN values. Replacing NaN with 0.")
            X_test = np.where(np.isnan(X_test), 0, X_test)

        # Data standardization
        X_test_scaled = self.X_scaler.transform(X_test)

        # 再次检查标准化后的数据
        if np.any(np.isnan(X_test_scaled)):
            print("Warning: Scaled test data contains NaN. Replacing with 0.")
            X_test_scaled = np.where(np.isnan(X_test_scaled), 0, X_test_scaled)

        # Convert to PyTorch tensor
        X_tensor = torch.FloatTensor(X_test_scaled).to(self.device)

        # Prediction
        self.model.eval()
        with torch.no_grad():
            predictions_scaled = self.model(X_tensor).cpu().numpy()

        # Inverse standardization
        predictions = self.y_scaler.inverse_transform(predictions_scaled)

        return predictions

    def plot_loss_history(self, save_path=None):
        """Plot loss history"""
        plt.figure(figsize=(10, 6))

        epochs = range(1, len(self.loss_history['train']) + 1)

        plt.plot(epochs, self.loss_history['train'], 'b-', label='Training Loss', alpha=0.8)
        plt.plot(epochs, self.loss_history['val'], 'r-', label='Validation Loss', alpha=0.8)

        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.title('Training and Validation Loss History')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def calculate_rpd(y_true, y_pred):
    """Calculate RPD (Relative Percent Difference) metric

    RPD (Relative Percent Difference) = Standard Deviation / RMSE
    Evaluation criteria:
    RPD > 2.5: Excellent prediction ability
    2.0 < RPD <= 2.5: Good
    1.8 < RPD <= 2.0: Acceptable
    RPD <= 1.8: Model needs improvement
    """
    # 处理NaN值
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    # 移除NaN值
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return 0

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    std_dev = np.std(y_true)

    if rmse == 0:
        return float('inf')

    rpd = std_dev / rmse
    return rpd


def evaluate_single_catechin(y_true, y_pred, catechin_name, dataset_name="Test Set"):
    """Evaluate regression model performance for a single catechin, including RPD metric"""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    # 处理NaN值
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]

    if len(y_true_clean) == 0:
        print(f"Warning: All data contains NaN for {catechin_name} - {dataset_name}")
        return {
            'mse': np.nan,
            'rmse': np.nan,
            'mae': np.nan,
            'r2': np.nan,
            'evs': np.nan,
            'rpd': 0
        }

    metrics = {}
    metrics['mse'] = mean_squared_error(y_true_clean, y_pred_clean)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    metrics['mae'] = mean_absolute_error(y_true_clean, y_pred_clean)
    metrics['r2'] = r2_score(y_true_clean, y_pred_clean)
    metrics['evs'] = explained_variance_score(y_true_clean, y_pred_clean)
    metrics['rpd'] = calculate_rpd(y_true_clean, y_pred_clean)

    # Evaluate RPD level
    if metrics['rpd'] > 2.5:
        rpd_level = "Excellent"
    elif metrics['rpd'] > 2.0:
        rpd_level = "Good"
    elif metrics['rpd'] > 1.8:
        rpd_level = "Acceptable"
    else:
        rpd_level = "Needs Improvement"

    print(f"\n{catechin_name} - {dataset_name} Evaluation Results:")
    print(f"  MSE: {metrics['mse']:.6f}")
    print(f"  RMSE: {metrics['rmse']:.6f}")
    print(f"  MAE: {metrics['mae']:.6f}")
    print(f"  R²: {metrics['r2']:.6f}")
    print(f"  Explained Variance: {metrics['evs']:.6f}")
    print(f"  RPD: {metrics['rpd']:.6f} ({rpd_level})")

    return metrics


def plot_single_catechin_results(y_true, y_pred, catechin_name, dataset_name="Test Set"):
    """Plot regression results for a single catechin"""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    # 处理NaN值
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]

    if len(y_true_clean) == 0:
        print(f"Warning: Cannot create plot for {catechin_name} - {dataset_name} due to NaN values")
        return None

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'{catechin_name} - {dataset_name} Regression Analysis', fontsize=16, fontweight='bold')

    # 1. Actual vs Predicted scatter plot
    ax = axes[0, 0]
    ax.scatter(y_true_clean, y_pred_clean, alpha=0.6, color='blue')

    # Add regression line
    coeffs = np.polyfit(y_true_clean, y_pred_clean, 1)
    poly = np.poly1d(coeffs)
    y_fit = poly(y_true_clean)

    min_val = min(y_true_clean.min(), y_pred_clean.min())
    max_val = max(y_true_clean.max(), y_pred_clean.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Ideal Line')
    ax.plot(y_true_clean, y_fit, 'g-', alpha=0.8, label='Regression Line')

    ax.set_xlabel('Actual Value (mg/g)')
    ax.set_ylabel('Predicted Value (mg/g)')
    ax.set_title('Actual vs Predicted')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add R² and RMSE text
    r2 = r2_score(y_true_clean, y_pred_clean)
    rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
    ax.text(0.05, 0.95, f'R² = {r2:.3f}\nRMSE = {rmse:.3f}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 2. Residual plot
    ax = axes[0, 1]
    residuals = y_pred_clean - y_true_clean
    ax.scatter(y_pred_clean, residuals, alpha=0.6, color='green')
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.8)
    ax.set_xlabel('Predicted Value (mg/g)')
    ax.set_ylabel('Residual (mg/g)')
    ax.set_title('Residual Plot')
    ax.grid(True, alpha=0.3)

    # 3. Error distribution histogram
    ax = axes[1, 0]
    ax.hist(residuals, bins=20, edgecolor='black', alpha=0.7, color='orange')
    ax.axvline(x=0, color='r', linestyle='--', alpha=0.8)
    ax.set_xlabel('Residual (mg/g)')
    ax.set_ylabel('Frequency')
    ax.set_title('Residual Distribution')
    ax.grid(True, alpha=0.3)

    # Add statistical information
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    ax.text(0.05, 0.95, f'Mean: {mean_residual:.4f}\nStd: {std_residual:.4f}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 4. Metrics summary
    ax = axes[1, 1]
    ax.axis('off')

    mse = mean_squared_error(y_true_clean, y_pred_clean)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    r2 = r2_score(y_true_clean, y_pred_clean)
    rpd = calculate_rpd(y_true_clean, y_pred_clean)

    metrics_text = f"Evaluation Metrics Summary:\n\n"
    metrics_text += f"MSE: {mse:.6f}\n"
    metrics_text += f"RMSE: {rmse:.6f}\n"
    metrics_text += f"MAE: {mae:.6f}\n"
    metrics_text += f"R²: {r2:.6f}\n"
    metrics_text += f"RPD: {rpd:.6f}\n\n"

    # RPD level description
    if rpd > 2.5:
        rpd_level = "Excellent (RPD > 2.5)"
    elif rpd > 2.0:
        rpd_level = "Good (2.0 < RPD ≤ 2.5)"
    elif rpd > 1.8:
        rpd_level = "Acceptable (1.8 < RPD ≤ 2.0)"
    else:
        rpd_level = "Needs Improvement (RPD ≤ 1.8)"

    metrics_text += f"RPD Level: {rpd_level}"

    ax.text(0.1, 0.5, metrics_text, fontsize=12, verticalalignment='center')

    plt.tight_layout()
    filename = f'{catechin_name}_{dataset_name}_results.png'.replace(' ', '_').replace('/', '_')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Chart saved as: {filename}")

    return fig


def create_geti_based_split(aligned_ids, test_geti_list):
    """Create train-test split based on GETI values
    训练集：不在test_geti_list中的样本
    测试集：在test_geti_list中的样本
    """
    print(f"\nCreating train-test split based on GETI values...")
    print(f"Test set GETI list: {test_geti_list}")
    print(f"Number of test samples: {len(test_geti_list)}")

    # 创建布尔掩码，标记哪些样本在测试集中
    test_mask = np.isin(aligned_ids, test_geti_list)
    train_mask = ~test_mask

    # 获取训练集和测试集的索引
    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]

    print(f"Training set samples: {len(train_idx)}")
    print(f"Test set samples: {len(test_idx)}")

    # 检查是否有测试集样本不在对齐的数据中
    missing_test_samples = [geti for geti in test_geti_list if geti not in aligned_ids]
    if missing_test_samples:
        print(f"Warning: The following test samples are not in aligned data: {missing_test_samples}")

    return train_idx, test_idx


def train_single_catechin_model(X, y, catechin_index, catechin_name, variety, maturity,
                                train_idx, test_idx, test_size=0.25, random_state=42):
    """Train deep learning model for a single catechin using predefined GETI-based split"""
    print(f"\n{'=' * 60}")
    print(f"Training deep learning model for {catechin_name}")
    print(f"{'=' * 60}")

    # Extract data for current catechin
    y_single = y[:, catechin_index].reshape(-1, 1)

    print(f"Target variable statistics:")
    print(f"  Mean: {np.mean(y_single):.4f}")
    print(f"  Standard Deviation: {np.std(y_single):.4f}")
    print(f"  Minimum: {np.min(y_single):.4f}")
    print(f"  Maximum: {np.max(y_single):.4f}")

    # Apply SNV+D1st preprocessing
    X_processed = SpectrumPreprocessor.apply_preprocessing(X, 'snv_d1st')

    # 检查预处理后的数据是否有NaN
    if np.any(np.isnan(X_processed)):
        print(f"Warning: Preprocessed data contains NaN. Replacing with 0.")
        X_processed = np.where(np.isnan(X_processed), 0, X_processed)

    # Use predefined GETI-based split
    X_train, X_test = X_processed[train_idx], X_processed[test_idx]
    y_train, y_test = y_single[train_idx], y_single[test_idx]

    # 再次检查训练和测试数据
    if np.any(np.isnan(X_train)) or np.any(np.isnan(y_train)):
        print("Warning: Training data contains NaN. Replacing with appropriate values.")
        X_train = np.where(np.isnan(X_train), np.nanmean(X_train, axis=0), X_train)
        y_train = np.where(np.isnan(y_train), np.nanmean(y_train, axis=0), y_train)

    if np.any(np.isnan(X_test)) or np.any(np.isnan(y_test)):
        print("Warning: Test data contains NaN. Replacing with appropriate values.")
        X_test = np.where(np.isnan(X_test), np.nanmean(X_train, axis=0), X_test)
        y_test = np.where(np.isnan(y_test), np.nanmean(y_train, axis=0), y_test)

    print(f"\nTraining set shape: {X_train.shape}, Test set shape: {X_test.shape}")

    # Set random seed for reproducibility
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    # Create and train deep learning regression model
    # Adjust network size based on data volume
    input_size = X_train.shape[1]

    # Dynamically adjust hidden layer size based on input feature count
    if input_size < 100:
        hidden_sizes = [512, 256, 128, 64, 32, 16]
    elif input_size < 300:
        hidden_sizes = [256, 128, 64, 32]
    else:
        hidden_sizes = [512, 256, 128, 64, 32]

    # Adjust batch size based on data volume
    batch_size = min(16, len(X_train))

    dl_model = DLRegression(
        hidden_sizes=hidden_sizes,
        dropout_rate=0.25,
        learning_rate=0.0005,
        batch_size=batch_size,
        epochs=2000,  # Increase training epochs
        patience=100,  # Increase early stopping patience
        weight_decay=1e-6
    )

    # Train model
    start_time = datetime.datetime.now()
    dl_model.fit(X_train, y_train, verbose=True)
    end_time = datetime.datetime.now()
    training_time = (end_time - start_time).total_seconds()
    print(f"Model training time: {training_time:.2f} seconds")

    # Plot loss history
    loss_plot_path = f'{catechin_name}_loss_history.png'.replace(' ', '_').replace('/', '_')
    dl_model.plot_loss_history(save_path=loss_plot_path)
    print(f"Loss history chart saved as: {loss_plot_path}")

    # Training set prediction and evaluation
    y_train_pred = dl_model.predict(X_train)
    train_metrics = evaluate_single_catechin(y_train, y_train_pred, catechin_name, "Training Set")

    # Test set prediction and evaluation
    y_test_pred = dl_model.predict(X_test)
    test_metrics = evaluate_single_catechin(y_test, y_test_pred, catechin_name, "Test Set")

    # Plot results
    plot_single_catechin_results(y_train, y_train_pred, catechin_name, "Training Set")
    plot_single_catechin_results(y_test, y_test_pred, catechin_name, "Test Set")

    return {
        'catechin_name': catechin_name,
        'catechin_index': catechin_index,
        'model': dl_model,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'training_time': training_time,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred,
        'train_idx': train_idx,
        'test_idx': test_idx
    }


def main():
    """Main function with GETI-based train-test split"""
    # 请替换为您的实际文件路径
    spectrum_path = r"C:\Users\Mayn\Desktop\实验MAX\鲜叶-儿茶素对应\儿茶素数据\儿茶素对应光谱 - 副本 - 副本.xlsx"
    catechin_path = r"C:\Users\Mayn\Desktop\实验MAX\鲜叶-儿茶素对应\儿茶素数据\数据-儿茶素-毫克每克 - 副本 - 副本.xlsx"

    # 从图片中提取的测试集GETI列表
    test_geti_list = [
        "11-C-8", "11-C-6", "1-A-2", "1-A-8", "5-A-11", "5-A-1", "5-C-1", "5-C-12", "6-A-10", "6-A-12", "6-B-7", "6-B-4",
        "9-A-9", "9-A-6", "9-B-3", "9-B-5", "9-C-12", "9-C-11", "9-C-2", "9-C-1",
        "1-A-3", "2-A-12","6-A-7","6-C-1"
    ]

    print("=" * 60)
    print("Tea Catechin Content Prediction Model")
    print("Training Strategy: Using all samples EXCEPT the 11 specified GETI as training set")
    print(f"Test Set: The 11 specified GETI samples from the image")
    print(f"Test GETI: {test_geti_list}")
    print("=" * 60)

    try:
        # 1. 加载数据
        print("\n1. Loading data...")
        X, spectrum_ids, wavelength_names, spectrum_variety, spectrum_maturity = SpectrumDataLoader.load_spectrum_data(
            spectrum_path)
        y, catechin_ids, catechin_names, catechin_variety, catechin_maturity = SpectrumDataLoader.load_catechin_data(
            catechin_path)

        # 2. 基于儿茶素数据对齐数据
        print("\n2. Aligning data based on catechin data...")
        X_aligned, y_aligned, aligned_ids, aligned_variety, aligned_maturity = SpectrumDataLoader.align_data_by_catechin(
            spectrum_ids, X, spectrum_variety, spectrum_maturity,
            catechin_ids, y, catechin_variety, catechin_maturity)

        print(f"\nAligned data statistics:")
        print(f"Number of samples: {X_aligned.shape[0]}")
        print(f"Number of spectral features: {X_aligned.shape[1]}")
        print(f"Number of catechin compounds: {y_aligned.shape[1]}")
        print(f"Catechin compounds: {catechin_names}")

        # 3. 基于GETI创建训练集和测试集划分
        print(f"\n3. Creating train-test split based on GETI values...")
        train_idx, test_idx = create_geti_based_split(aligned_ids, test_geti_list)

        # 打印划分统计信息
        print(f"\nData split statistics:")
        print(f"Training set sample count: {len(train_idx)}")
        print(f"Test set sample count: {len(test_idx)}")
        print(f"Training set proportion: {len(train_idx) / len(aligned_ids):.2%}")
        print(f"Test set proportion: {len(test_idx) / len(aligned_ids):.2%}")

        # 打印测试集详细信息
        print(f"\nTest set details:")
        for i, idx in enumerate(test_idx):
            geti = aligned_ids[idx]
            variety = aligned_variety[idx]
            maturity = aligned_maturity[idx]
            print(f"  {i + 1:2d}. GETI: {geti}, Variety: {variety}, Maturity: {maturity}")

        # 4. 为每个儿茶素化合物训练单独的模型
        print(f"\n4. Starting to train deep learning models for each catechin compound...")
        print(f"Preprocessing method: SNV + First Derivative")
        print(f"Train-Test Split: Based on specified GETI values")
        print(f"Deep learning framework: PyTorch")
        print(f"Model type: Multilayer Perceptron (MLP)")

        all_results = []
        models_dict = {}  # 用于存储训练好的模型

        for i, catechin_name in enumerate(catechin_names):
            print(f"\n{'=' * 60}")
            print(f"Processing catechin {i + 1}/{len(catechin_names)}: {catechin_name}")
            print(f"{'=' * 60}")

            # 为单个儿茶素训练深度学习模型
            result = train_single_catechin_model(
                X_aligned, y_aligned, i, catechin_name,
                aligned_variety, aligned_maturity,
                train_idx, test_idx,
                test_size=0.2, random_state=42
            )

            all_results.append(result)
            models_dict[catechin_name] = result  # 存储模型

        # 5. 总结所有儿茶素化合物的结果
        print(f"\n{'=' * 60}")
        print("Deep Learning Model Performance Summary for All Catechin Compounds")
        print(f"{'=' * 60}")

        summary_data = []
        for result in all_results:
            catechin_name = result['catechin_name']
            train_metrics = result['train_metrics']
            test_metrics = result['test_metrics']

            summary_data.append({
                'Catechin Compound': catechin_name,
                'Training Set R²': train_metrics['r2'],
                'Test Set R²': test_metrics['r2'],
                'Training Set RMSE': train_metrics['rmse'],
                'Test Set RMSE': test_metrics['rmse'],
                'Training Set MAE': train_metrics['mae'],
                'Test Set MAE': test_metrics['mae'],
                'Training Set RPD': train_metrics['rpd'],
                'Test Set RPD': test_metrics['rpd'],
                'Training Time (seconds)': result['training_time']
            })

        # 创建总结DataFrame
        summary_df = pd.DataFrame(summary_data)

        # 按测试集R²降序排序
        summary_df = summary_df.sort_values('Test Set R²', ascending=False)

        print("\nDeep Learning Model Performance Summary (Sorted by Test Set R² descending):")
        print("=" * 120)
        print(summary_df.to_string(index=False))
        print("=" * 120)

        # 6. 保存结果到CSV文件
        output_dir = "catechin_dl_results_geti_based_split"
        os.makedirs(output_dir, exist_ok=True)

        # 保存总结结果
        summary_path = os.path.join(output_dir, "dl_summary_results.csv")
        summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
        print(f"\nSummary results saved to: {summary_path}")

        # 7. 生成性能比较图表
        print("\nGenerating deep learning model performance comparison charts...")
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # R²比较
        ax = axes[0, 0]
        x_pos = np.arange(len(summary_df))
        width = 0.35
        ax.bar(x_pos - width / 2, summary_df['Training Set R²'], width, label='Training Set', color='skyblue')
        ax.bar(x_pos + width / 2, summary_df['Test Set R²'], width, label='Test Set', color='lightcoral')
        ax.set_xlabel('Catechin Compound')
        ax.set_ylabel('R²')
        ax.set_title('Deep Learning Models - R² Comparison for Each Catechin')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(summary_df['Catechin Compound'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # RPD比较
        ax = axes[0, 1]
        bars = ax.bar(x_pos, summary_df['Test Set RPD'], color='lightgreen')
        ax.set_xlabel('Catechin Compound')
        ax.set_ylabel('RPD')
        ax.set_title('Deep Learning Models - Test Set RPD Comparison for Each Catechin')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(summary_df['Catechin Compound'], rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')

        # Add RPD horizontal reference lines
        ax.axhline(y=2.5, color='red', linestyle='--', alpha=0.5, label='Excellent (RPD>2.5)')
        ax.axhline(y=2.0, color='orange', linestyle='--', alpha=0.5, label='Good (RPD>2.0)')
        ax.axhline(y=1.8, color='yellow', linestyle='--', alpha=0.5, label='Acceptable (RPD>1.8)')
        ax.legend()

        # Add RPD level labels on bars
        for i, (bar, rpd) in enumerate(zip(bars, summary_df['Test Set RPD'])):
            height = bar.get_height()
            if rpd > 2.5:
                label = 'Excellent'
                color = 'green'
            elif rpd > 2.0:
                label = 'Good'
                color = 'orange'
            elif rpd > 1.8:
                label = 'Acceptable'
                color = 'yellow'
            else:
                label = 'Needs Improvement'
                color = 'red'

            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.05,
                    label, ha='center', va='bottom', color=color, fontsize=9)

        # RMSE比较
        ax = axes[1, 0]
        ax.bar(x_pos - width / 2, summary_df['Training Set RMSE'], width, label='Training Set', color='skyblue')
        ax.bar(x_pos + width / 2, summary_df['Test Set RMSE'], width, label='Test Set', color='lightcoral')
        ax.set_xlabel('Catechin Compound')
        ax.set_ylabel('RMSE')
        ax.set_title('Deep Learning Models - RMSE Comparison for Each Catechin')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(summary_df['Catechin Compound'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # Training time comparison
        ax = axes[1, 1]
        ax.bar(x_pos, summary_df['Training Time (seconds)'], color='purple')
        ax.set_xlabel('Catechin Compound')
        ax.set_ylabel('Training Time (seconds)')
        ax.set_title('Deep Learning Models - Training Time Comparison for Each Catechin')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(summary_df['Catechin Compound'], rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        comparison_path = os.path.join(output_dir, "dl_performance_comparison.png")
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Performance comparison chart saved to: {comparison_path}")

        # 8. 保存详细的预测结果
        print("\nSaving detailed prediction results...")
        all_predictions = []

        for result in all_results:
            catechin_name = result['catechin_name']
            train_idx = result['train_idx']
            test_idx = result['test_idx']

            # Training set results
            train_df = pd.DataFrame({
                'Dataset': ['Training Set'] * len(result['y_train']),
                'Catechin Compound': [catechin_name] * len(result['y_train']),
                'Sample Index': list(range(len(result['y_train']))),
                'Original Index': train_idx,
                'Variety': aligned_variety[train_idx],
                'Maturity': aligned_maturity[train_idx],
                'GETI': aligned_ids[train_idx],
                'Actual Value': result['y_train'].flatten(),
                'Predicted Value': result['y_train_pred'].flatten(),
                'Residual': (result['y_train_pred'] - result['y_train']).flatten()
            })

            # Test set results
            test_df = pd.DataFrame({
                'Dataset': ['Test Set'] * len(result['y_test']),
                'Catechin Compound': [catechin_name] * len(result['y_test']),
                'Sample Index': list(range(len(result['y_test']))),
                'Original Index': test_idx,
                'Variety': aligned_variety[test_idx],
                'Maturity': aligned_maturity[test_idx],
                'GETI': aligned_ids[test_idx],
                'Actual Value': result['y_test'].flatten(),
                'Predicted Value': result['y_test_pred'].flatten(),
                'Residual': (result['y_test_pred'] - result['y_test']).flatten()
            })

            all_predictions.append(train_df)
            all_predictions.append(test_df)

        # Combine all results
        predictions_df = pd.concat(all_predictions, ignore_index=True)
        predictions_path = os.path.join(output_dir, "dl_detailed_predictions.csv")
        predictions_df.to_csv(predictions_path, index=False, encoding='utf-8-sig')
        print(f"Detailed prediction results saved to: {predictions_path}")

        # 9. 专门保存测试集预测结果（图片中的11个个体）
        print("\nSaving test set (11 specified GETI) prediction results...")
        test_results_list = []

        for result in all_results:
            catechin_name = result['catechin_name']
            test_idx = result['test_idx']
            y_test = result['y_test']
            y_test_pred = result['y_test_pred']

            # 为每个测试样本创建结果
            for i, idx in enumerate(test_idx):
                geti = aligned_ids[idx]
                variety = aligned_variety[idx]
                maturity = aligned_maturity[idx]
                actual_value = y_test[i][0] if len(y_test) > i else np.nan
                predicted_value = y_test_pred[i][0] if len(y_test_pred) > i else np.nan

                test_results_list.append({
                    'GETI': geti,
                    'Variety': variety,
                    'Maturity': maturity,
                    'Catechin Compound': catechin_name,
                    'Actual Value (mg/g)': actual_value,
                    'Predicted Value (mg/g)': predicted_value,
                    'Residual (mg/g)': predicted_value - actual_value
                })

        # 创建测试集结果DataFrame
        test_results_df = pd.DataFrame(test_results_list)

        # 重新排列列以便更好地查看
        test_results_df = test_results_df[['GETI', 'Variety', 'Maturity', 'Catechin Compound',
                                           'Actual Value (mg/g)', 'Predicted Value (mg/g)', 'Residual (mg/g)']]

        # 保存测试集结果
        test_results_path = os.path.join(output_dir, "test_set_predictions_11_geti.csv")
        test_results_df.to_csv(test_results_path, index=False, encoding='utf-8-sig')
        print(f"Test set (11 GETI) prediction results saved to: {test_results_path}")

        # 创建测试集结果透视表，方便查看每个GETI的所有儿茶素预测结果
        test_pivot_df = test_results_df.pivot_table(
            index=['GETI', 'Variety', 'Maturity'],
            columns='Catechin Compound',
            values='Predicted Value (mg/g)',
            aggfunc='first'
        ).reset_index()

        # 计算总儿茶素含量
        catechin_columns = [col for col in test_pivot_df.columns if col not in ['GETI', 'Variety', 'Maturity']]
        if catechin_columns:
            test_pivot_df['Total Catechin (mg/g)'] = test_pivot_df[catechin_columns].sum(axis=1)

        test_pivot_path = os.path.join(output_dir, "test_set_predictions_pivot_11_geti.csv")
        test_pivot_df.to_csv(test_pivot_path, index=False, encoding='utf-8-sig')
        print(f"Test set predictions pivot table saved to: {test_pivot_path}")

        # 打印测试集预测结果摘要
        print(f"\n{'=' * 60}")
        print("TEST SET PREDICTIONS SUMMARY (11 SPECIFIED GETI)")
        print(f"{'=' * 60}")

        for geti in test_geti_list:
            if geti in aligned_ids[test_idx]:
                geti_mask = test_results_df['GETI'] == geti
                geti_results = test_results_df[geti_mask]

                if not geti_results.empty:
                    print(f"\nGETI: {geti}")
                    print(f"Variety: {geti_results['Variety'].iloc[0]}, Maturity: {geti_results['Maturity'].iloc[0]}")
                    print("-" * 40)

                    for _, row in geti_results.iterrows():
                        catechin = row['Catechin Compound']
                        actual = row['Actual Value (mg/g)']
                        predicted = row['Predicted Value (mg/g)']
                        residual = row['Residual (mg/g)']

                        print(f"  {catechin}:")
                        print(
                            f"    Actual: {actual:.4f} mg/g, Predicted: {predicted:.4f} mg/g, Residual: {residual:.4f} mg/g")

        # 10. 生成模型报告
        report_path = os.path.join(output_dir, "dl_model_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("Tea Catechin Content Prediction Deep Learning Model Report\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Generation Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Number of Samples: {X_aligned.shape[0]}\n")
            f.write(f"Number of Features: {X_aligned.shape[1]}\n")
            f.write(f"Number of Catechin Compounds: {y_aligned.shape[1]}\n")
            f.write(f"Preprocessing Method: SNV + First Derivative\n")
            f.write(f"Model Type: Multilayer Perceptron (MLP)\n")
            f.write(f"Framework: PyTorch\n")
            f.write(f"Train-Test Split: Based on specified GETI values\n")
            f.write(f"Training Set Count: {len(train_idx)} ({len(train_idx) / len(aligned_ids):.1%})\n")
            f.write(f"Test Set Count: {len(test_idx)} ({len(test_idx) / len(aligned_ids):.1%})\n")
            f.write(f"Test Set GETI: {test_geti_list}\n")
            f.write(f"Random Seed: 42\n\n")

            f.write("Test Set Details:\n")
            for i, idx in enumerate(test_idx):
                geti = aligned_ids[idx]
                variety = aligned_variety[idx]
                maturity = aligned_maturity[idx]
                f.write(f"  {i + 1:2d}. GETI: {geti}, Variety: {variety}, Maturity: {maturity}\n")

            f.write("\nPerformance Summary for Each Catechin Compound:\n")
            f.write("-" * 60 + "\n")
            for i, row in summary_df.iterrows():
                f.write(f"\n{row['Catechin Compound']}:\n")
                f.write(f"  Training Set R²: {row['Training Set R²']:.4f}\n")
                f.write(f"  Test Set R²: {row['Test Set R²']:.4f}\n")
                f.write(f"  Training Set RMSE: {row['Training Set RMSE']:.4f}\n")
                f.write(f"  Test Set RMSE: {row['Test Set RMSE']:.4f}\n")
                f.write(f"  Training Set MAE: {row['Training Set MAE']:.4f}\n")
                f.write(f"  Test Set MAE: {row['Test Set MAE']:.4f}\n")
                f.write(f"  Training Set RPD: {row['Training Set RPD']:.4f}\n")
                f.write(f"  Test Set RPD: {row['Test Set RPD']:.4f}\n")
                f.write(f"  Training Time: {row['Training Time (seconds)']:.2f} seconds\n")

            f.write(f"\n\nOverall Statistics:\n")
            f.write("-" * 60 + "\n")
            f.write(f"Average Training Set R²: {summary_df['Training Set R²'].mean():.4f}\n")
            f.write(f"Average Test Set R²: {summary_df['Test Set R²'].mean():.4f}\n")
            f.write(f"Average Test Set RPD: {summary_df['Test Set RPD'].mean():.4f}\n")
            f.write(f"Total Training Time: {summary_df['Training Time (seconds)'].sum():.2f} seconds\n")

        print(f"\nModel report saved to: {report_path}")

        # 11. 输出最终总结
        print(f"\n{'=' * 60}")
        print("MODEL TRAINING AND PREDICTION COMPLETED!")
        print(f"{'=' * 60}")
        print(f"Training Strategy: All samples EXCEPT the 11 specified GETI")
        print(f"Test Set: 11 specified GETI from the image")
        print(f"Number of Catechin Compounds Processed: {len(catechin_names)}")
        print(f"Average Test Set R²: {summary_df['Test Set R²'].mean():.4f}")
        print(f"Average Test Set RPD: {summary_df['Test Set RPD'].mean():.4f}")

        # 最佳和最差模型
        best_model = summary_df.loc[summary_df['Test Set R²'].idxmax()]
        worst_model = summary_df.loc[summary_df['Test Set R²'].idxmin()]

        print(f"\nBest Model:")
        print(f"  Catechin Compound: {best_model['Catechin Compound']}")
        print(f"  Test Set R²: {best_model['Test Set R²']:.4f}")
        print(f"  Test Set RPD: {best_model['Test Set RPD']:.4f}")

        print(f"\nWorst Model:")
        print(f"  Catechin Compound: {worst_model['Catechin Compound']}")
        print(f"  Test Set R²: {worst_model['Test Set R²']:.4f}")
        print(f"  Test Set RPD: {worst_model['Test Set RPD']:.4f}")

        print(f"\nGenerated Files:")
        print(f"  1. {output_dir}/dl_summary_results.csv - 总结结果")
        print(f"  2. {output_dir}/dl_performance_comparison.png - 性能比较图表")
        print(f"  3. {output_dir}/dl_detailed_predictions.csv - 详细预测结果")
        print(f"  4. {output_dir}/test_set_predictions_11_geti.csv - 测试集预测结果")
        print(f"  5. {output_dir}/test_set_predictions_pivot_11_geti.csv - 测试集预测结果透视表")
        print(f"  6. {output_dir}/dl_model_report.txt - 模型报告")
        print(f"  7. 每个儿茶素的回归分析图表")
        print(f"  8. 每个儿茶素的损失历史图表")
        print(f"{'=' * 60}")

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        print("\nPlease check:")
        print("1. File paths are correct")
        print("2. File format is correct (should be .xlsx)")
        print("3. File content meets requirements (first three columns should be Variety, Maturity, GETI)")
        print(
            "4. Required Python libraries are installed: pandas, numpy, scipy, scikit-learn, matplotlib, seaborn, torch")
        print("5. Ensure PyTorch is installed: pip install torch")


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"CUDA is available, using GPU acceleration")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available, using CPU")

    # Run main function
    main()