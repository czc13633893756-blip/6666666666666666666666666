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
import itertools

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
    """光谱数据预处理类"""

    @staticmethod
    def sg_smoothing(X, window_length=11, polyorder=2):
        """Savitzky-Golay平滑"""
        return savgol_filter(X, window_length, polyorder, axis=1)

    @staticmethod
    def first_derivative(X):
        """一阶导数"""
        return np.gradient(X, axis=1)

    @staticmethod
    def snv(X):
        """标准正态变量变换"""
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
        """多元散射校正"""
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
        """SNV + 一阶导数"""
        X_snv = SpectrumPreprocessor.snv(X)
        return SpectrumPreprocessor.first_derivative(X_snv)

    @staticmethod
    def sg_snv_d1st(X, window_length=11, polyorder=2):
        """SG平滑 + SNV + 一阶导数"""
        # 第一步：SG平滑
        X_sg = SpectrumPreprocessor.sg_smoothing(X, window_length, polyorder)
        # 第二步：SNV
        X_snv = SpectrumPreprocessor.snv(X_sg)
        # 第三步：一阶导数
        X_sg_snv_d1st = SpectrumPreprocessor.first_derivative(X_snv)
        return X_sg_snv_d1st

    @staticmethod
    def apply_preprocessing(X, method='sg_snv_d1st'):
        """应用指定的预处理方法"""
        preprocess_methods = {
            'snv_d1st': SpectrumPreprocessor.snv_d1st,
            'sg_snv_d1st': SpectrumPreprocessor.sg_snv_d1st
        }

        if method not in preprocess_methods:
            raise ValueError(f"不支持的预处理方法: {method}")

        print(f"应用预处理方法: {method}")
        return preprocess_methods[method](X)


class SpectrumDataLoader:
    """数据加载器"""

    @staticmethod
    def load_spectrum_data(file_path):
        print(f"加载光谱数据: {file_path}")
        spectrum_data = pd.read_excel(file_path, header=0)
        variety_data = spectrum_data.iloc[:, 0].values
        maturity_data = spectrum_data.iloc[:, 1].values
        sample_ids = spectrum_data.iloc[:, 2].values
        X = spectrum_data.iloc[:, 3:].values
        wavelength_names = list(spectrum_data.columns[3:])
        return X, sample_ids, wavelength_names, variety_data, maturity_data

    @staticmethod
    def load_catechin_data(file_path):
        print(f"加载儿茶素数据: {file_path}")
        catechin_data = pd.read_excel(file_path, header=0)
        variety_data = catechin_data.iloc[:, 0].values
        maturity_data = catechin_data.iloc[:, 1].values
        sample_ids = catechin_data.iloc[:, 2].values
        catechin_names = list(catechin_data.columns[3:])
        y = catechin_data.iloc[:, 3:].values
        return y, sample_ids, catechin_names, variety_data, maturity_data

    @staticmethod
    def align_data_by_catechin(spectrum_ids, spectrum_X, spectrum_variety, spectrum_maturity,
                               catechin_ids, catechin_y, catechin_variety, catechin_maturity):
        print("\n以儿茶素数据为基准进行数据对齐...")
        spectrum_dict = {}
        for i, geti in enumerate(spectrum_ids):
            if geti not in spectrum_dict:
                spectrum_dict[geti] = {
                    'index': i, 'X': spectrum_X[i],
                    'variety': spectrum_variety[i], 'maturity': spectrum_maturity[i]
                }

        aligned_X, aligned_y, aligned_ids = [], [], []
        aligned_variety, aligned_maturity = [], []

        for i, geti in enumerate(catechin_ids):
            if geti in spectrum_dict:
                spectrum_info = spectrum_dict[geti]
                if catechin_variety[i] != spectrum_info['variety']:
                    continue
                aligned_X.append(spectrum_info['X'])
                aligned_y.append(catechin_y[i])
                aligned_ids.append(geti)
                aligned_variety.append(catechin_variety[i])
                aligned_maturity.append(spectrum_info['maturity'])

        aligned_X = np.array(aligned_X)
        aligned_y = np.array(aligned_y)
        aligned_ids = np.array(aligned_ids)
        aligned_variety = np.array(aligned_variety)
        aligned_maturity = np.array(aligned_maturity)

        if len(aligned_ids) == 0:
            raise ValueError("没有找到共同的样本ID，请检查数据文件！")

        return aligned_X, aligned_y, aligned_ids, aligned_variety, aligned_maturity


# ==========================================
# 深度学习模型部分 (ResNet + Attention)
# ==========================================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""

    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class ResBlock1D(nn.Module):
    """带SE注意力机制的一维残差块"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, downsample=None):
        super(ResBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                               padding=kernel_size // 2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2,
                               bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.se = SEBlock(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class SpectralResNetAttention(nn.Module):
    """高级光谱回归模型: 1D ResNet + SE Attention"""

    def __init__(self, input_length, layers=[2, 2, 2], base_filters=16, dropout_rate=0.3):
        super(SpectralResNetAttention, self).__init__()
        self.in_channels = base_filters
        self.conv1 = nn.Conv1d(1, base_filters, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(base_filters)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(base_filters, layers[0], stride=1)
        self.layer2 = self._make_layer(base_filters * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(base_filters * 4, layers[2], stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(base_filters * 4, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1)
        )
        self._initialize_weights()

    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        layers = []
        layers.append(ResBlock1D(self.in_channels, out_channels, stride=stride, downsample=downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResBlock1D(out_channels, out_channels))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class DLRegression:
    """深度学习回归模型包装器"""

    class Scaler:
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
            return scaled_data * (self._std + 1e-8) + self._mean

    def __init__(self, hidden_sizes=None, dropout_rate=0.3,
                 learning_rate=0.001, batch_size=32, epochs=500,
                 patience=50, weight_decay=1e-4, device=None,
                 layers=[2, 2, 2]):

        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.weight_decay = weight_decay
        self.layers = layers

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.model = None
        self.X_scaler = self.Scaler()
        self.y_scaler = self.Scaler()
        self.loss_history = {'train': [], 'val': []}

    def fit(self, X, y, verbose=True):
        """训练深度学习模型"""
        X_scaled = self.X_scaler.fit_transform(X)
        y_scaled = self.y_scaler.fit_transform(y)

        X_tensor = torch.FloatTensor(X_scaled).unsqueeze(1).to(self.device)
        y_tensor = torch.FloatTensor(y_scaled).to(self.device)

        dataset = TensorDataset(X_tensor, y_tensor)

        input_length = X.shape[1]
        self.model = SpectralResNetAttention(
            input_length=input_length,
            layers=self.layers,
            base_filters=16,
            dropout_rate=self.dropout_rate
        ).to(self.device)

        criterion = nn.MSELoss()
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=15
        )

        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        indices = np.arange(len(X))
        np.random.shuffle(indices)
        split_idx = int(0.85 * len(X))

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
            print(f"训练设备: {self.device}")
            print(f"模型架构: ResNet-18 (1D Modified) with SE-Attention")
            print(f"超参数: LR={self.learning_rate}, Drop={self.dropout_rate}, BS={self.batch_size}, Layers={self.layers}")
            print(f"训练集大小: {len(train_idx)}, 验证集大小: {len(val_idx)}")
            print("开始训练...")

        for epoch in range(self.epochs):
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

            scheduler.step(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            if verbose and (epoch + 1) % 50 == 0:
                print(f"Epoch [{epoch + 1:4d}/{self.epochs}] | "
                      f"Train Loss: {avg_train_loss:.6f} | "
                      f"Val Loss: {avg_val_loss:.6f} | "
                      f"LR: {optimizer.param_groups[0]['lr']:.6f}")

            if patience_counter >= self.patience:
                if verbose:
                    print(f"早停触发，在Epoch {epoch + 1}停止训练")
                break

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        if verbose:
            print(f"训练完成，最佳验证损失: {best_val_loss:.6f}")

        return self

    def predict(self, X_test):
        if self.model is None:
            raise ValueError("模型未训练，请先调用fit方法")

        X_test_scaled = self.X_scaler.transform(X_test)
        X_tensor = torch.FloatTensor(X_test_scaled).unsqueeze(1).to(self.device)

        self.model.eval()
        with torch.no_grad():
            predictions_scaled = self.model(X_tensor).cpu().numpy()

        predictions = self.y_scaler.inverse_transform(predictions_scaled)
        return predictions

    def plot_loss_history(self, save_path=None):
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(self.loss_history['train']) + 1)
        plt.plot(epochs, self.loss_history['train'], 'b-', label='训练损失', alpha=0.8)
        plt.plot(epochs, self.loss_history['val'], 'r-', label='验证损失', alpha=0.8)
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.title('训练和验证损失历史')
        plt.legend()
        plt.grid(True, alpha=0.3)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def calculate_rpd(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    std_dev = np.std(y_true)
    if rmse == 0: return float('inf')
    return std_dev / rmse


def evaluate_single_catechin(y_true, y_pred, catechin_name, dataset_name="测试集"):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    metrics = {}
    metrics['mse'] = mean_squared_error(y_true, y_pred)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    metrics['r2'] = r2_score(y_true, y_pred)
    metrics['evs'] = explained_variance_score(y_true, y_pred)
    metrics['rpd'] = calculate_rpd(y_true, y_pred)

    if metrics['rpd'] > 2.5:
        rpd_level = "优秀"
    elif metrics['rpd'] > 2.0:
        rpd_level = "良好"
    elif metrics['rpd'] > 1.8:
        rpd_level = "可接受"
    else:
        rpd_level = "需要改进"

    print(f"\n{catechin_name} - {dataset_name}评估结果:")
    print(f"  MSE: {metrics['mse']:.6f}")
    print(f"  RMSE: {metrics['rmse']:.6f}")
    print(f"  MAE: {metrics['mae']:.6f}")
    print(f"  R²: {metrics['r2']:.6f}")
    print(f"  RPD: {metrics['rpd']:.6f} ({rpd_level})")
    return metrics


def plot_single_catechin_results(y_true, y_pred, catechin_name, dataset_name="测试集"):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'{catechin_name} - {dataset_name}回归分析', fontsize=16, fontweight='bold')

    ax = axes[0, 0]
    ax.scatter(y_true, y_pred, alpha=0.6, color='blue')
    coeffs = np.polyfit(y_true, y_pred, 1)
    poly = np.poly1d(coeffs)
    y_fit = poly(y_true)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='理想线')
    ax.plot(y_true, y_fit, 'g-', alpha=0.8, label='回归线')
    ax.set_xlabel('真实值 (mg/g)')
    ax.set_ylabel('预测值 (mg/g)')
    ax.set_title('真实值 vs 预测值')
    ax.legend()
    ax.grid(True, alpha=0.3)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    ax.text(0.05, 0.95, f'R² = {r2:.3f}\nRMSE = {rmse:.3f}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax = axes[0, 1]
    residuals = y_pred - y_true
    ax.scatter(y_pred, residuals, alpha=0.6, color='green')
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.8)
    ax.set_xlabel('预测值 (mg/g)')
    ax.set_ylabel('残差 (mg/g)')
    ax.set_title('残差图')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.hist(residuals, bins=20, edgecolor='black', alpha=0.7, color='orange')
    ax.axvline(x=0, color='r', linestyle='--', alpha=0.8)
    ax.set_xlabel('残差 (mg/g)')
    ax.set_ylabel('频数')
    ax.set_title('残差分布')
    ax.grid(True, alpha=0.3)
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    ax.text(0.05, 0.95, f'均值: {mean_residual:.4f}\n标准差: {std_residual:.4f}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax = axes[1, 1]
    ax.axis('off')
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rpd = calculate_rpd(y_true, y_pred)
    metrics_text = f"评估指标汇总:\n\n"
    metrics_text += f"MSE: {mse:.6f}\n"
    metrics_text += f"RMSE: {rmse:.6f}\n"
    metrics_text += f"MAE: {mae:.6f}\n"
    metrics_text += f"R²: {r2:.6f}\n"
    metrics_text += f"RPD: {rpd:.6f}\n\n"
    if rpd > 2.5:
        rpd_level = "优秀 (RPD > 2.5)"
    elif rpd > 2.0:
        rpd_level = "良好"
    elif rpd > 1.8:
        rpd_level = "可接受"
    else:
        rpd_level = "需要改进"
    metrics_text += f"RPD等级: {rpd_level}"
    ax.text(0.1, 0.5, metrics_text, fontsize=12, verticalalignment='center')

    plt.tight_layout()
    filename = f'{catechin_name}_{dataset_name}_results.png'.replace(' ', '_').replace('/', '_')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"图表已保存为: {filename}")
    return fig


def create_stratified_split(X, variety, maturity, test_size=0.2, random_state=42):
    stratify_labels = [f"{v}_{m}" for v, m in zip(variety, maturity)]
    unique_labels = np.unique(stratify_labels)
    print(f"分层标签数量: {len(unique_labels)}")
    for label in unique_labels:
        count = stratify_labels.count(label)
        print(f"  标签 '{label}': {count} 个样本")
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    for train_index, test_index in sss.split(X, stratify_labels):
        return train_index, test_index


# ==========================================
# 5折交叉验证网格搜索功能
# ==========================================
def grid_search_cv(X, y, param_grid, k_folds=5, device=None):
    """
    使用K折交叉验证进行超参数网格搜索
    """
    print(f"\n开始 {k_folds} 折交叉验证寻找最佳超参数...")
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    # 生成所有参数组合
    keys = param_grid.keys()
    combinations = list(itertools.product(*param_grid.values()))

    best_score = float('inf')
    best_params = None

    total_combinations = len(combinations)
    print(f"总共有 {total_combinations} 种参数组合待测试")

    for idx, combo in enumerate(combinations):
        current_params = dict(zip(keys, combo))
        print(f"  测试组合 {idx + 1}/{total_combinations}: {current_params} ... ", end="")

        val_losses = []

        # 对每组参数进行交叉验证
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train_fold, y_train_fold = X[train_idx], y[train_idx]
            X_val_fold, y_val_fold = X[val_idx], y[val_idx]

            # 使用较少的 epoch 和 patience 进行快速搜索
            model = DLRegression(
                dropout_rate=current_params.get('dropout_rate', 0.3),
                learning_rate=current_params.get('learning_rate', 0.001),
                batch_size=current_params.get('batch_size', 32),
                layers=current_params.get('layers', [2, 2, 2]),
                epochs=100,  # 搜索时减少轮数以节省时间
                patience=15,
                device=device
            )

            # 训练模型 (关闭详细输出)
            model.fit(X_train_fold, y_train_fold, verbose=False)

            # 验证集预测
            y_pred = model.predict(X_val_fold)
            mse = mean_squared_error(y_val_fold, y_pred)
            val_losses.append(mse)

        avg_mse = np.mean(val_losses)
        print(f"-> 平均验证 MSE: {avg_mse:.6f}")

        if avg_mse < best_score:
            best_score = avg_mse
            best_params = current_params

    print(f"最佳参数找到: {best_params}, 最佳 CV MSE: {best_score:.6f}")
    return best_params


def train_single_catechin_model(X, y, catechin_index, catechin_name, variety, maturity,
                                train_idx, test_idx, test_size=0.25, random_state=42):
    """训练单个儿茶素物质的深度学习模型，包含超参数搜索"""
    print(f"\n{'=' * 60}")
    print(f"训练 {catechin_name} 的深度学习模型")
    print(f"{'=' * 60}")

    y_single = y[:, catechin_index].reshape(-1, 1)

    # SG+SNV+D1st 预处理
    X_processed = SpectrumPreprocessor.apply_preprocessing(X, 'sg_snv_d1st')

    X_train, X_test = X_processed[train_idx], X_processed[test_idx]
    y_train, y_test = y_single[train_idx], y_single[test_idx]

    print(f"训练集形状: {X_train.shape}, 测试集形状: {X_test.shape}")

    torch.manual_seed(random_state)
    np.random.seed(random_state)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 步骤 1: 定义超参数搜索空间
    # 可以根据计算资源调整网格大小
    param_grid = {
        'learning_rate': [0.001, 0.0005],
        'dropout_rate': [0.3, 0.5],
        'batch_size': [32],
        'layers': [
            [2, 2, 2],  # 标准深度
            [3, 3, 3]  # 更深一点的网络
        ]
    }

    # 步骤 2: 5折交叉验证寻找最佳参数 (仅在训练集上进行)
    best_params = grid_search_cv(X_train, y_train, param_grid, k_folds=5, device=device)

    # 步骤 3: 使用最佳参数训练最终模型
    print(f"\n使用最佳参数开始最终训练...")

    dl_model = DLRegression(
        dropout_rate=best_params['dropout_rate'],
        learning_rate=best_params['learning_rate'],
        batch_size=best_params['batch_size'],
        layers=best_params['layers'],
        epochs=1500,  # 最终训练使用完整轮数
        patience=80,
        weight_decay=1e-3,
        device=device
    )

    start_time = datetime.datetime.now()
    dl_model.fit(X_train, y_train, verbose=True)
    end_time = datetime.datetime.now()
    training_time = (end_time - start_time).total_seconds()
    print(f"模型训练时间: {training_time:.2f}秒")

    loss_plot_path = f'{catechin_name}_loss_history.png'.replace(' ', '_').replace('/', '_')
    dl_model.plot_loss_history(save_path=loss_plot_path)
    print(f"损失历史图已保存为: {loss_plot_path}")

    y_train_pred = dl_model.predict(X_train)
    train_metrics = evaluate_single_catechin(y_train, y_train_pred, catechin_name, "训练集")

    y_test_pred = dl_model.predict(X_test)
    test_metrics = evaluate_single_catechin(y_test, y_test_pred, catechin_name, "测试集")

    plot_single_catechin_results(y_train, y_train_pred, catechin_name, "训练集")
    plot_single_catechin_results(y_test, y_test_pred, catechin_name, "测试集")

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
        'test_idx': test_idx,
        'best_params': best_params
    }


def main():
    """主函数"""
    spectrum_path = r"C:\Users\Mayn\Desktop\实验MAX\鲜叶-儿茶素对应\儿茶素数据\儿茶素对应光谱 - 副本 - 副本.xlsx"
    catechin_path = r"C:\Users\Mayn\Desktop\实验MAX\鲜叶-儿茶素对应\儿茶素数据\数据-儿茶素-毫克每克 - 副本 - 副本.xlsx"

    print("=" * 60)
    print("茶叶儿茶素含量预测模型 (深度学习: 1D-ResNet + Attention)")
    print("增加: 5折交叉验证寻找最佳超参数")
    print("预处理: SG平滑 + SNV + 一阶导数")
    print("=" * 60)

    try:
        print("\n1. 加载数据...")
        X, spectrum_ids, wavelength_names, spectrum_variety, spectrum_maturity = SpectrumDataLoader.load_spectrum_data(
            spectrum_path)
        y, catechin_ids, catechin_names, catechin_variety, catechin_maturity = SpectrumDataLoader.load_catechin_data(
            catechin_path)

        print("\n2. 以儿茶素数据为基准对齐数据...")
        X_aligned, y_aligned, aligned_ids, aligned_variety, aligned_maturity = SpectrumDataLoader.align_data_by_catechin(
            spectrum_ids, X, spectrum_variety, spectrum_maturity,
            catechin_ids, y, catechin_variety, catechin_maturity)

        print(f"\n3. 创建基于品种和成熟度的分层抽样划分...")
        train_idx, test_idx = create_stratified_split(
            X_aligned, aligned_variety, aligned_maturity,
            test_size=0.2, random_state=42
        )

        print(f"\n4. 开始训练流程...")
        all_results = []

        for i, catechin_name in enumerate(catechin_names):
            print(f"\n{'=' * 60}")
            print(f"处理第 {i + 1}/{len(catechin_names)} 个儿茶素物质: {catechin_name}")
            print(f"{'=' * 60}")

            result = train_single_catechin_model(
                X_aligned, y_aligned, i, catechin_name,
                aligned_variety, aligned_maturity,
                train_idx, test_idx,
                test_size=0.2, random_state=42
            )

            all_results.append(result)

        print(f"\n{'=' * 60}")
        print("所有儿茶素物质的深度学习模型性能汇总")
        print(f"{'=' * 60}")

        summary_data = []
        for result in all_results:
            catechin_name = result['catechin_name']
            train_metrics = result['train_metrics']
            test_metrics = result['test_metrics']
            best_params = result['best_params']

            summary_data.append({
                '儿茶素物质': catechin_name,
                '最佳参数': str(best_params),
                '训练集R²': train_metrics['r2'],
                '测试集R²': test_metrics['r2'],
                '训练集RMSE': train_metrics['rmse'],
                '测试集RMSE': test_metrics['rmse'],
                '训练集RPD': train_metrics['rpd'],
                '测试集RPD': test_metrics['rpd'],
                '训练时间(秒)': result['training_time']
            })

        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('测试集R²', ascending=False)

        print("\n深度学习模型性能汇总表 (按测试集R²降序排序):")
        print("=" * 120)
        print(summary_df.to_string(index=False))
        print("=" * 120)

        output_dir = "catechin_dl_results_resnet_attn_cv"
        os.makedirs(output_dir, exist_ok=True)

        summary_path = os.path.join(output_dir, "dl_summary_results.csv")
        summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
        print(f"\n汇总结果已保存到: {summary_path}")

    except Exception as e:
        print(f"错误: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        print(f"CUDA可用，使用GPU加速: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA不可用，使用CPU")
    main()