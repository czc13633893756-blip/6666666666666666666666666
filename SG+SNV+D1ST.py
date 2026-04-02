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
import itertools  # 新增：用于参数组合

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
    def apply_preprocessing(X, method='sg_snv_d1st'):  # 修改默认预处理方法为sg_snv_d1st
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
    """数据加载器 (重命名以避免与PyTorch的DataLoader冲突)"""

    @staticmethod
    def load_spectrum_data(file_path):
        """加载光谱数据
        新的Excel格式：前两列是Variety和Maturity，第三列是GETI，后面是光谱数据
        """
        print(f"加载光谱数据: {file_path}")
        spectrum_data = pd.read_excel(file_path, header=0)
        print(f"光谱数据形状: {spectrum_data.shape}")

        # 提取品种、成熟度和样本ID
        variety_data = spectrum_data.iloc[:, 0].values
        maturity_data = spectrum_data.iloc[:, 1].values
        sample_ids = spectrum_data.iloc[:, 2].values
        X = spectrum_data.iloc[:, 3:].values
        wavelength_names = list(spectrum_data.columns[3:])

        print(f"样本数量: {len(sample_ids)}")
        print(f"波长数量: {len(wavelength_names)}")
        print(f"品种类型数量: {len(np.unique(variety_data))}")
        print(f"成熟度类型数量: {len(np.unique(maturity_data))}")

        return X, sample_ids, wavelength_names, variety_data, maturity_data

    @staticmethod
    def load_catechin_data(file_path):
        """加载儿茶素数据
        新的Excel格式：前两列是Variety和Maturity，第三列是GETI，后面是儿茶素数据
        """
        print(f"加载儿茶素数据: {file_path}")
        catechin_data = pd.read_excel(file_path, header=0)
        print(f"儿茶素数据形状: {catechin_data.shape}")

        # 提取品种、成熟度和样本ID
        variety_data = catechin_data.iloc[:, 0].values
        maturity_data = catechin_data.iloc[:, 1].values
        sample_ids = catechin_data.iloc[:, 2].values
        catechin_names = list(catechin_data.columns[3:])
        y = catechin_data.iloc[:, 3:].values

        print(f"样本数量: {len(sample_ids)}")
        print(f"儿茶素种类数量: {len(catechin_names)}")
        print(f"儿茶素名称: {catechin_names}")
        print(f"品种类型数量: {len(np.unique(variety_data))}")
        print(f"成熟度类型数量: {len(np.unique(maturity_data))}")

        return y, sample_ids, catechin_names, variety_data, maturity_data

    @staticmethod
    def align_data_by_catechin(spectrum_ids, spectrum_X, spectrum_variety, spectrum_maturity,
                               catechin_ids, catechin_y, catechin_variety, catechin_maturity):
        """以儿茶素数据为基准对齐数据
        只保留儿茶素数据中存在的GETI对应的光谱数据
        """
        print("\n以儿茶素数据为基准进行数据对齐...")
        print(f"儿茶素数据样本数: {len(catechin_ids)}")
        print(f"光谱数据样本数: {len(spectrum_ids)}")

        # 创建光谱数据的映射字典，便于快速查找
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
                print(f"警告: GETI {geti} 在光谱数据中重复出现")

        aligned_X = []
        aligned_y = []
        aligned_ids = []
        aligned_variety = []
        aligned_maturity = []
        missing_count = 0

        # 以儿茶素数据为基准
        for i, geti in enumerate(catechin_ids):
            if geti in spectrum_dict:
                spectrum_info = spectrum_dict[geti]

                # 验证品种和成熟度是否匹配
                if catechin_variety[i] != spectrum_info['variety']:
                    print(f"警告: 样本 {geti} 的品种不匹配 (儿茶素: {catechin_variety[i]}, 光谱: {spectrum_info['variety']})")
                    continue

                if catechin_maturity[i] != spectrum_info['maturity']:
                    print(f"警告: 样本 {geti} 的成熟度不匹配 (儿茶素: {catechin_maturity[i]}, 光谱: {spectrum_info['maturity']})")
                    continue

                aligned_X.append(spectrum_info['X'])
                aligned_y.append(catechin_y[i])
                aligned_ids.append(geti)
                aligned_variety.append(catechin_variety[i])
                aligned_maturity.append(spectrum_info['maturity'])
            else:
                missing_count += 1
                print(f"警告: 儿茶素样本 {geti} 在光谱数据中未找到对应数据")

        aligned_X = np.array(aligned_X)
        aligned_y = np.array(aligned_y)
        aligned_ids = np.array(aligned_ids)
        aligned_variety = np.array(aligned_variety)
        aligned_maturity = np.array(aligned_maturity)

        print(f"成功对齐的样本数: {len(aligned_ids)}")
        print(f"未找到对应光谱数据的儿茶素样本数: {missing_count}")

        if len(aligned_ids) == 0:
            raise ValueError("没有找到共同的样本ID，请检查数据文件！")

        print(f"对齐后数据形状: X={aligned_X.shape}, y={aligned_y.shape}")
        print(f"对齐后品种类型: {np.unique(aligned_variety)}")
        print(f"对齐后成熟度类型: {np.unique(aligned_maturity)}")

        return aligned_X, aligned_y, aligned_ids, aligned_variety, aligned_maturity


class NeuralNetworkModel(nn.Module):
    """深度学习神经网络模型"""

    def __init__(self, input_size, hidden_sizes=[256, 128, 64], dropout_rate=0.2):
        super(NeuralNetworkModel, self).__init__()

        layers = []
        prev_size = input_size

        # 构建隐藏层
        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size

        # 输出层
        layers.append(nn.Linear(prev_size, 1))

        self.model = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        """权重初始化"""
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
    """深度学习回归模型"""

    class Scaler:
        """数据标准化类"""

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
            """将标准化后的数据还原"""
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

        # 设备选择
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
        # 数据标准化
        X_scaled = self.X_scaler.fit_transform(X)
        y_scaled = self.y_scaler.fit_transform(y)

        # 转换为PyTorch张量
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.FloatTensor(y_scaled).to(self.device)

        # 创建数据加载器
        dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = TorchDataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # 初始化模型
        input_size = X.shape[1]
        self.model = NeuralNetworkModel(
            input_size=input_size,
            hidden_sizes=self.hidden_sizes,
            dropout_rate=self.dropout_rate
        ).to(self.device)

        # 定义损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )

        # 早停设置
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        # 从训练集中划分验证集
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
            print(f"训练设备: {self.device}")
            print(f"训练集大小: {len(train_idx)}, 验证集大小: {len(val_idx)}")
            print(f"网络结构: 输入层({input_size}) -> ", end="")
            for i, size in enumerate(self.hidden_sizes):
                print(f"隐藏层{i + 1}({size}) -> ", end="")
            print("输出层(1)")
            print("开始训练...")

        # 训练循环
        for epoch in range(self.epochs):
            # 训练阶段
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

            # 验证阶段
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

            # 学习率调整
            scheduler.step(avg_val_loss)

            # 早停检查
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            # 打印训练进度
            if verbose and (epoch + 1) % 50 == 0:
                print(f"Epoch [{epoch + 1:4d}/{self.epochs}] | "
                      f"Train Loss: {avg_train_loss:.6f} | "
                      f"Val Loss: {avg_val_loss:.6f} | "
                      f"LR: {optimizer.param_groups[0]['lr']:.6f}")

            # 检查早停
            if patience_counter >= self.patience:
                if verbose:
                    print(f"早停触发，在Epoch {epoch + 1}停止训练")
                break

        # 加载最佳模型
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        if verbose:
            print(f"训练完成，最佳验证损失: {best_val_loss:.6f}")

        return self

    def predict(self, X_test):
        """预测"""
        if self.model is None:
            raise ValueError("模型未训练，请先调用fit方法")

        # 数据标准化
        X_test_scaled = self.X_scaler.transform(X_test)

        # 转换为PyTorch张量
        X_tensor = torch.FloatTensor(X_test_scaled).to(self.device)

        # 预测
        self.model.eval()
        with torch.no_grad():
            predictions_scaled = self.model(X_tensor).cpu().numpy()

        # 反标准化
        predictions = self.y_scaler.inverse_transform(predictions_scaled)

        return predictions

    def plot_loss_history(self, save_path=None):
        """绘制损失历史"""
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
    """计算RPD（相对分析误差）指标"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    std_dev = np.std(y_true)

    if rmse == 0:
        return float('inf')

    rpd = std_dev / rmse
    return rpd


def evaluate_single_catechin(y_true, y_pred, catechin_name, dataset_name="测试集"):
    """评估单个儿茶素物质的回归模型性能，包含RPD指标"""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    metrics = {}
    metrics['mse'] = mean_squared_error(y_true, y_pred)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    metrics['r2'] = r2_score(y_true, y_pred)
    metrics['evs'] = explained_variance_score(y_true, y_pred)
    metrics['rpd'] = calculate_rpd(y_true, y_pred)

    # 评估RPD等级
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
    print(f"  可解释方差: {metrics['evs']:.6f}")
    print(f"  RPD: {metrics['rpd']:.6f} ({rpd_level})")

    return metrics


def plot_single_catechin_results(y_true, y_pred, catechin_name, dataset_name="测试集"):
    """绘制单个儿茶素物质的回归结果图"""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'{catechin_name} - {dataset_name}回归分析', fontsize=16, fontweight='bold')

    # 1. 真实值 vs 预测值散点图
    ax = axes[0, 0]
    ax.scatter(y_true, y_pred, alpha=0.6, color='blue')

    # 添加回归线
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

    # 添加R²和RMSE文本
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    ax.text(0.05, 0.95, f'R² = {r2:.3f}\nRMSE = {rmse:.3f}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 2. 残差图
    ax = axes[0, 1]
    residuals = y_pred - y_true
    ax.scatter(y_pred, residuals, alpha=0.6, color='green')
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.8)
    ax.set_xlabel('预测值 (mg/g)')
    ax.set_ylabel('残差 (mg/g)')
    ax.set_title('残差图')
    ax.grid(True, alpha=0.3)

    # 3. 误差分布直方图
    ax = axes[1, 0]
    ax.hist(residuals, bins=20, edgecolor='black', alpha=0.7, color='orange')
    ax.axvline(x=0, color='r', linestyle='--', alpha=0.8)
    ax.set_xlabel('残差 (mg/g)')
    ax.set_ylabel('频数')
    ax.set_title('残差分布')
    ax.grid(True, alpha=0.3)

    # 添加统计信息
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    ax.text(0.05, 0.95, f'均值: {mean_residual:.4f}\n标准差: {std_residual:.4f}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 4. 指标汇总
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

    # RPD等级说明
    if rpd > 2.5:
        rpd_level = "优秀 (RPD > 2.5)"
    elif rpd > 2.0:
        rpd_level = "良好 (2.0 < RPD ≤ 2.5)"
    elif rpd > 1.8:
        rpd_level = "可接受 (1.8 < RPD ≤ 2.0)"
    else:
        rpd_level = "需要改进 (RPD ≤ 1.8)"

    metrics_text += f"RPD等级: {rpd_level}"

    ax.text(0.1, 0.5, metrics_text, fontsize=12, verticalalignment='center')

    plt.tight_layout()
    filename = f'{catechin_name}_{dataset_name}_results.png'.replace(' ', '_').replace('/', '_')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"图表已保存为: {filename}")

    return fig


def create_stratified_split(X, variety, maturity, test_size=0.2, random_state=42):
    """创建基于品种和成熟度的分层抽样划分"""
    # 创建分层标签：品种_成熟度组合
    stratify_labels = [f"{v}_{m}" for v, m in zip(variety, maturity)]

    # 获取唯一的分层标签
    unique_labels = np.unique(stratify_labels)
    print(f"分层标签数量: {len(unique_labels)}")
    for label in unique_labels:
        count = stratify_labels.count(label)
        print(f"  标签 '{label}': {count} 个样本")

    # 使用StratifiedShuffleSplit进行分层抽样
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)

    for train_index, test_index in sss.split(X, stratify_labels):
        return train_index, test_index


# ==========================================
# 新增: 5折交叉验证寻找最佳超参数
# ==========================================
def hyperparameter_search(X, y, param_grid, device, k_folds=5):
    """
    使用K折交叉验证进行超参数网格搜索
    """
    print(f"\n开始 {k_folds} 折交叉验证寻找最佳参数...")

    # 生成所有参数组合
    keys = param_grid.keys()
    combinations = list(itertools.product(*param_grid.values()))

    best_score = float('inf')
    best_params = None

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    total_combinations = len(combinations)
    print(f"总共有 {total_combinations} 种参数组合待测试")

    for idx, combo in enumerate(combinations):
        current_params = dict(zip(keys, combo))
        val_losses = []

        # 打印当前进度
        if (idx + 1) % 1 == 0:
            print(f"测试组合 {idx + 1}/{total_combinations}: {current_params} ... ", end="")

        for train_idx, val_idx in kf.split(X):
            X_fold_train, X_fold_val = X[train_idx], X[val_idx]
            y_fold_train, y_fold_val = y[train_idx], y[val_idx]

            # 使用较少的epoch和patience进行快速搜索
            temp_model = DLRegression(
                hidden_sizes=current_params['hidden_sizes'],
                dropout_rate=current_params['dropout_rate'],
                learning_rate=current_params['learning_rate'],
                batch_size=32,
                epochs=150,  # 搜索时使用较少轮数
                patience=15,  # 搜索时耐心值减小
                device=device
            )

            temp_model.fit(X_fold_train, y_fold_train, verbose=False)
            y_pred = temp_model.predict(X_fold_val)
            fold_mse = mean_squared_error(y_fold_val, y_pred)
            val_losses.append(fold_mse)

        avg_loss = np.mean(val_losses)
        print(f"平均MSE: {avg_loss:.6f}")

        if avg_loss < best_score:
            best_score = avg_loss
            best_params = current_params

    print(f"\n最佳参数找到: {best_params}, 最佳交叉验证MSE: {best_score:.6f}")
    return best_params


def train_single_catechin_model(X, y, catechin_index, catechin_name, variety, maturity,
                                train_idx, test_idx, test_size=0.25, random_state=42):
    """训练单个儿茶素物质的深度学习模型，使用预定义的分层划分"""
    print(f"\n{'=' * 60}")
    print(f"训练 {catechin_name} 的深度学习模型")
    print(f"{'=' * 60}")

    # 提取当前儿茶素物质的数据
    y_single = y[:, catechin_index].reshape(-1, 1)

    print(f"目标变量统计:")
    print(f"  均值: {np.mean(y_single):.4f}")
    print(f"  标准差: {np.std(y_single):.4f}")
    print(f"  最小值: {np.min(y_single):.4f}")
    print(f"  最大值: {np.max(y_single):.4f}")

    # 按品种和成熟度统计
    unique_strata = np.unique([f"{v}_{m}" for v, m in zip(variety, maturity)], return_counts=True)
    print(f"  品种×成熟度组合: {unique_strata[0]}")
    print(f"  各组合样本数: {unique_strata[1]}")

    # 应用SG+SNV+D1st预处理
    X_processed = SpectrumPreprocessor.apply_preprocessing(X, 'sg_snv_d1st')  # 修改为SG+SNV+D1st预处理

    # 使用预定义的分层划分
    X_train, X_test = X_processed[train_idx], X_processed[test_idx]
    y_train, y_test = y_single[train_idx], y_single[test_idx]

    # 检查训练集和测试集中品种和成熟度的分布
    train_variety = variety[train_idx]
    train_maturity = maturity[train_idx]
    test_variety = variety[test_idx]
    test_maturity = maturity[test_idx]

    print(f"\n训练集品种分布:")
    unique_train_variety, counts_train_variety = np.unique(train_variety, return_counts=True)
    for var, count in zip(unique_train_variety, counts_train_variety):
        print(f"  {var}: {count} 个样本")

    print(f"\n测试集品种分布:")
    unique_test_variety, counts_test_variety = np.unique(test_variety, return_counts=True)  # 修复这里的错误
    for var, count in zip(unique_test_variety, counts_test_variety):
        print(f"  {var}: {count} 个样本")

    print(f"\n训练集成熟度分布:")
    unique_train_maturity, counts_train_maturity = np.unique(train_maturity, return_counts=True)
    for mat, count in zip(unique_train_maturity, counts_train_maturity):
        print(f"  {mat}: {count} 个样本")

    print(f"\n测试集成熟度分布:")
    unique_test_maturity, counts_test_maturity = np.unique(test_maturity, return_counts=True)  # 修复这里的错误
    for mat, count in zip(unique_test_maturity, counts_test_maturity):
        print(f"  {mat}: {count} 个样本")

    print(f"\n训练集形状: {X_train.shape}, 测试集形状: {X_test.shape}")

    # 设置随机种子以保证可重复性
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    # 检测设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ==========================================================
    # 步骤 1: 定义超参数搜索空间
    # ==========================================================
    input_size = X_train.shape[1]

    # 简单的网格搜索空间
    param_grid = {
        'learning_rate': [0.001, 0.0005],
        'dropout_rate': [0.2, 0.4],
        'hidden_sizes': [
            [256, 128, 64],
            [512, 256, 128, 64]
        ]
    }

    # ==========================================================
    # 步骤 2: 5折交叉验证寻找最佳参数
    # ==========================================================
    best_params = hyperparameter_search(X_train, y_train, param_grid, device, k_folds=5)

    # ==========================================================
    # 步骤 3: 使用最佳参数训练最终模型
    # ==========================================================
    print(f"\n使用最佳参数开始最终训练: {best_params}")

    # 根据数据量调整批量大小
    batch_size = min(32, len(X_train))

    dl_model = DLRegression(
        hidden_sizes=best_params['hidden_sizes'],
        dropout_rate=best_params['dropout_rate'],
        learning_rate=best_params['learning_rate'],
        batch_size=batch_size,
        epochs=2000,  # 最终训练增加轮数
        patience=100,  # 最终训练增加耐心
        weight_decay=1e-6,
        device=device
    )

    # 训练模型
    start_time = datetime.datetime.now()
    dl_model.fit(X_train, y_train, verbose=True)
    end_time = datetime.datetime.now()
    training_time = (end_time - start_time).total_seconds()
    print(f"模型训练时间: {training_time:.2f}秒")

    # 绘制损失历史
    loss_plot_path = f'{catechin_name}_loss_history.png'.replace(' ', '_').replace('/', '_')
    dl_model.plot_loss_history(save_path=loss_plot_path)
    print(f"损失历史图已保存为: {loss_plot_path}")

    # 训练集预测和评估
    y_train_pred = dl_model.predict(X_train)
    train_metrics = evaluate_single_catechin(y_train, y_train_pred, catechin_name, "训练集")

    # 测试集预测和评估
    y_test_pred = dl_model.predict(X_test)
    test_metrics = evaluate_single_catechin(y_test, y_test_pred, catechin_name, "测试集")

    # 绘制结果图
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
    # 注意：请将以下路径替换为您的实际文件路径
    spectrum_path = r"C:\Users\Mayn\Desktop\实验MAX\鲜叶-儿茶素对应\儿茶素数据\儿茶素对应光谱 - 副本 - 副本.xlsx"
    catechin_path = r"C:\Users\Mayn\Desktop\实验MAX\鲜叶-儿茶素对应\儿茶素数据\数据-儿茶素-毫克每克 - 副本 - 副本.xlsx"

    print("=" * 60)
    print("茶叶儿茶素含量预测模型 (深度学习 + SG+SNV+D1st预处理 + 5折CV)")
    print("使用基于品种和成熟度的分层抽样")
    print("以儿茶素数据为基准进行数据对齐")
    print("=" * 60)

    try:
        # 1. 加载数据
        print("\n1. 加载数据...")
        X, spectrum_ids, wavelength_names, spectrum_variety, spectrum_maturity = SpectrumDataLoader.load_spectrum_data(
            spectrum_path)
        y, catechin_ids, catechin_names, catechin_variety, catechin_maturity = SpectrumDataLoader.load_catechin_data(
            catechin_path)

        # 2. 以儿茶素数据为基准对齐数据
        print("\n2. 以儿茶素数据为基准对齐数据...")
        X_aligned, y_aligned, aligned_ids, aligned_variety, aligned_maturity = SpectrumDataLoader.align_data_by_catechin(
            spectrum_ids, X, spectrum_variety, spectrum_maturity,
            catechin_ids, y, catechin_variety, catechin_maturity)

        print(f"\n对齐后数据统计:")
        print(f"样本数量: {X_aligned.shape[0]}")
        print(f"光谱特征数量: {X_aligned.shape[1]}")
        print(f"儿茶素物质数量: {y_aligned.shape[1]}")
        print(f"儿茶素物质: {catechin_names}")
        print(f"品种类型: {np.unique(aligned_variety)}")
        print(f"成熟度类型: {np.unique(aligned_maturity)}")

        # 3. 创建基于品种和成熟度的分层抽样划分
        print(f"\n3. 创建基于品种和成熟度的分层抽样划分...")
        train_idx, test_idx = create_stratified_split(
            X_aligned, aligned_variety, aligned_maturity,
            test_size=0.2, random_state=42
        )

        # 打印划分统计
        print(f"训练集样本数: {len(train_idx)}")
        print(f"测试集样本数: {len(test_idx)}")
        print(f"训练集比例: {len(train_idx) / len(X_aligned):.2%}")
        print(f"测试集比例: {len(test_idx) / len(X_aligned):.2%}")

        # 4. 为每个儿茶素物质单独训练模型
        print(f"\n4. 开始为每个儿茶素物质单独训练深度学习模型 (使用SG+SNV+D1st预处理)...")
        print(f"预处理方法: SG平滑 + SNV + 一阶导数")
        print(f"分层抽样: 基于品种和成熟度组合")
        print(f"数据对齐: 以儿茶素数据为基准")
        print(f"深度学习框架: PyTorch")
        print(f"模型类型: 多层感知机(MLP)")
        print(f"优化策略: 5折交叉验证寻找最佳超参数")

        all_results = []

        for i, catechin_name in enumerate(catechin_names):
            print(f"\n{'=' * 60}")
            print(f"处理第 {i + 1}/{len(catechin_names)} 个儿茶素物质: {catechin_name}")
            print(f"{'=' * 60}")

            # 训练单个儿茶素物质的深度学习模型
            result = train_single_catechin_model(
                X_aligned, y_aligned, i, catechin_name,
                aligned_variety, aligned_maturity,
                train_idx, test_idx,
                test_size=0.2, random_state=42
            )

            all_results.append(result)

        # 5. 汇总所有儿茶素物质的结果
        print(f"\n{'=' * 60}")
        print("所有儿茶素物质的深度学习模型性能汇总")
        print(f"{'=' * 60}")

        summary_data = []
        for result in all_results:
            catechin_name = result['catechin_name']
            train_metrics = result['train_metrics']
            test_metrics = result['test_metrics']

            summary_data.append({
                '儿茶素物质': catechin_name,
                '训练集R²': train_metrics['r2'],
                '测试集R²': test_metrics['r2'],
                '训练集RMSE': train_metrics['rmse'],
                '测试集RMSE': test_metrics['rmse'],
                '训练集MAE': train_metrics['mae'],
                '测试集MAE': test_metrics['mae'],
                '训练集RPD': train_metrics['rpd'],
                '测试集RPD': test_metrics['rpd'],
                '训练时间(秒)': result['training_time'],
                '最佳参数': str(result['best_params'])
            })

        # 创建汇总DataFrame
        summary_df = pd.DataFrame(summary_data)

        # 按测试集R²降序排序
        summary_df = summary_df.sort_values('测试集R²', ascending=False)

        print("\n深度学习模型性能汇总表 (按测试集R²降序排序):")
        print("=" * 120)
        print(summary_df.to_string(index=False))
        print("=" * 120)

        # 6. 保存结果到CSV文件
        output_dir = "catechin_dl_results_sg_snv_d1st_cv"
        os.makedirs(output_dir, exist_ok=True)

        # 保存汇总结果
        summary_path = os.path.join(output_dir, "dl_summary_results.csv")
        summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
        print(f"\n汇总结果已保存到: {summary_path}")

        # 7. 绘制性能比较图
        print("\n生成深度学习模型性能比较图...")
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # R²比较
        ax = axes[0, 0]
        x_pos = np.arange(len(summary_df))
        width = 0.35
        ax.bar(x_pos - width / 2, summary_df['训练集R²'], width, label='训练集', color='skyblue')
        ax.bar(x_pos + width / 2, summary_df['测试集R²'], width, label='测试集', color='lightcoral')
        ax.set_xlabel('儿茶素物质')
        ax.set_ylabel('R²')
        ax.set_title('深度学习模型 - 各儿茶素物质的R²比较 (SG+SNV+D1st预处理)')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(summary_df['儿茶素物质'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # RPD比较
        ax = axes[0, 1]
        bars = ax.bar(x_pos, summary_df['测试集RPD'], color='lightgreen')
        ax.set_xlabel('儿茶素物质')
        ax.set_ylabel('RPD')
        ax.set_title('深度学习模型 - 各儿茶素物质的测试集RPD比较 (SG+SNV+D1st预处理)')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(summary_df['儿茶素物质'], rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')

        # 添加RPD水平参考线
        ax.axhline(y=2.5, color='red', linestyle='--', alpha=0.5, label='优秀 (RPD>2.5)')
        ax.axhline(y=2.0, color='orange', linestyle='--', alpha=0.5, label='良好 (RPD>2.0)')
        ax.axhline(y=1.8, color='yellow', linestyle='--', alpha=0.5, label='可接受 (RPD>1.8)')
        ax.legend()

        # 在柱状图上添加RPD等级标签
        for i, (bar, rpd) in enumerate(zip(bars, summary_df['测试集RPD'])):
            height = bar.get_height()
            if rpd > 2.5:
                label = '优秀'
                color = 'green'
            elif rpd > 2.0:
                label = '良好'
                color = 'orange'
            elif rpd > 1.8:
                label = '可接受'
                color = 'yellow'
            else:
                label = '需改进'
                color = 'red'

            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.05,
                    label, ha='center', va='bottom', color=color, fontsize=9)

        # RMSE比较
        ax = axes[1, 0]
        ax.bar(x_pos - width / 2, summary_df['训练集RMSE'], width, label='训练集', color='skyblue')
        ax.bar(x_pos + width / 2, summary_df['测试集RMSE'], width, label='测试集', color='lightcoral')
        ax.set_xlabel('儿茶素物质')
        ax.set_ylabel('RMSE')
        ax.set_title('深度学习模型 - 各儿茶素物质的RMSE比较 (SG+SNV+D1st预处理)')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(summary_df['儿茶素物质'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # 训练时间
        ax = axes[1, 1]
        ax.bar(x_pos, summary_df['训练时间(秒)'], color='purple')
        ax.set_xlabel('儿茶素物质')
        ax.set_ylabel('训练时间 (秒)')
        ax.set_title('深度学习模型 - 各儿茶素物质的训练时间比较')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(summary_df['儿茶素物质'], rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        comparison_path = os.path.join(output_dir, "dl_performance_comparison.png")
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"性能比较图已保存到: {comparison_path}")

        # 8. 保存详细的预测结果
        print("\n保存详细的预测结果...")
        all_predictions = []

        for result in all_results:
            catechin_name = result['catechin_name']
            train_idx = result['train_idx']
            test_idx = result['test_idx']

            # 训练集结果
            train_df = pd.DataFrame({
                '数据集': ['训练集'] * len(result['y_train']),
                '儿茶素物质': [catechin_name] * len(result['y_train']),
                '样本索引': list(range(len(result['y_train']))),
                '原始索引': train_idx,
                '品种': aligned_variety[train_idx],
                '成熟度': aligned_maturity[train_idx],
                'GETI': aligned_ids[train_idx],
                '真实值': result['y_train'].flatten(),
                '预测值': result['y_train_pred'].flatten(),
                '残差': (result['y_train_pred'] - result['y_train']).flatten()
            })

            # 测试集结果
            test_df = pd.DataFrame({
                '数据集': ['测试集'] * len(result['y_test']),
                '儿茶素物质': [catechin_name] * len(result['y_test']),
                '样本索引': list(range(len(result['y_test']))),
                '原始索引': test_idx,
                '品种': aligned_variety[test_idx],
                '成熟度': aligned_maturity[test_idx],
                'GETI': aligned_ids[test_idx],
                '真实值': result['y_test'].flatten(),
                '预测值': result['y_test_pred'].flatten(),
                '残差': (result['y_test_pred'] - result['y_test']).flatten()
            })

            all_predictions.append(train_df)
            all_predictions.append(test_df)

        # 合并所有结果
        predictions_df = pd.concat(all_predictions, ignore_index=True)
        predictions_path = os.path.join(output_dir, "dl_detailed_predictions.csv")
        predictions_df.to_csv(predictions_path, index=False, encoding='utf-8-sig')
        print(f"详细预测结果已保存到: {predictions_path}")

        # 9. 保存分层抽样信息
        stratify_info = pd.DataFrame({
            '样本索引': np.arange(len(aligned_ids)),
            'GETI': aligned_ids,
            '品种': aligned_variety,
            '成熟度': aligned_maturity,
            '数据集划分': ['训练集' if i in train_idx else '测试集' for i in range(len(aligned_ids))]
        })

        stratify_path = os.path.join(output_dir, "stratify_info.csv")
        stratify_info.to_csv(stratify_path, index=False, encoding='utf-8-sig')
        print(f"分层抽样信息已保存到: {stratify_path}")

        # 10. 生成模型报告
        report_path = os.path.join(output_dir, "dl_model_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("茶叶儿茶素含量预测深度学习模型报告 (SG+SNV+D1st预处理 + 分层抽样 + 5折CV)\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"样本数量: {X_aligned.shape[0]}\n")
            f.write(f"特征数量: {X_aligned.shape[1]}\n")
            f.write(f"儿茶素物质数量: {y_aligned.shape[1]}\n")
            f.write(f"预处理方法: SG平滑 + SNV + 一阶导数\n")
            f.write(f"模型类型: 多层感知机(MLP)\n")
            f.write(f"框架: PyTorch\n")
            f.write(f"抽样方法: 基于品种和成熟度的分层抽样\n")
            f.write(f"优化策略: 5折交叉验证寻找最佳参数\n")
            f.write(f"数据对齐: 以儿茶素数据为基准\n")
            f.write(f"训练集数量: {len(train_idx)} ({len(train_idx) / len(aligned_ids):.1%})\n")
            f.write(f"测试集数量: {len(test_idx)} ({len(test_idx) / len(aligned_ids):.1%})\n")
            f.write(f"随机种子: 42\n\n")

            f.write("品种分布:\n")
            unique_varieties, variety_counts = np.unique(aligned_variety, return_counts=True)
            for var, count in zip(unique_varieties, variety_counts):
                f.write(f"  {var}: {count} 个样本 ({count / len(aligned_ids):.1%})\n")

            f.write("\n成熟度分布:\n")
            unique_maturities, maturity_counts = np.unique(aligned_maturity, return_counts=True)
            for mat, count in zip(unique_maturities, maturity_counts):
                f.write(f"  {mat}: {count} 个样本 ({count / len(aligned_ids):.1%})\n")

            f.write("\n各儿茶素物质模型性能汇总:\n")
            f.write("-" * 60 + "\n")
            for i, row in summary_df.iterrows():
                f.write(f"\n{row['儿茶素物质']}:\n")
                f.write(f"  最佳参数: {row['最佳参数']}\n")
                f.write(f"  训练集R²: {row['训练集R²']:.4f}\n")
                f.write(f"  测试集R²: {row['测试集R²']:.4f}\n")
                f.write(f"  训练集RMSE: {row['训练集RMSE']:.4f}\n")
                f.write(f"  测试集RMSE: {row['测试集RMSE']:.4f}\n")
                f.write(f"  训练集MAE: {row['训练集MAE']:.4f}\n")
                f.write(f"  测试集MAE: {row['测试集MAE']:.4f}\n")
                f.write(f"  训练集RPD: {row['训练集RPD']:.4f}\n")
                f.write(f"  测试集RPD: {row['测试集RPD']:.4f}\n")
                f.write(f"  训练时间: {row['训练时间(秒)']:.2f}秒\n")

            f.write(f"\n\n总体统计:\n")
            f.write("-" * 60 + "\n")
            f.write(f"平均训练集R²: {summary_df['训练集R²'].mean():.4f}\n")
            f.write(f"平均测试集R²: {summary_df['测试集R²'].mean():.4f}\n")
            f.write(f"平均测试集RPD: {summary_df['测试集RPD'].mean():.4f}\n")
            f.write(f"总训练时间: {summary_df['训练时间(秒)'].sum():.2f}秒\n")

            # RPD等级统计
            rpd_levels = []
            for rpd in summary_df['测试集RPD']:
                if rpd > 2.5:
                    rpd_levels.append('优秀')
                elif rpd > 2.0:
                    rpd_levels.append('良好')
                elif rpd > 1.8:
                    rpd_levels.append('可接受')
                else:
                    rpd_levels.append('需改进')

            f.write(f"\nRPD等级统计:\n")
            f.write(f"  优秀 (RPD > 2.5): {rpd_levels.count('优秀')} 个\n")
            f.write(f"  良好 (2.0 < RPD ≤ 2.5): {rpd_levels.count('良好')} 个\n")
            f.write(f"  可接受 (1.8 < RPD ≤ 2.0): {rpd_levels.count('可接受')} 个\n")
            f.write(f"  需改进 (RPD ≤ 1.8): {rpd_levels.count('需改进')} 个\n")

        print(f"模型报告已保存到: {report_path}")

        # 11. 输出最终总结
        print(f"\n{'=' * 60}")
        print("深度学习模型训练完成!")
        print(f"{'=' * 60}")
        print(f"预处理方法: SG平滑 + SNV + 一阶导数")
        print(f"模型类型: 多层感知机(MLP)")
        print(f"抽样方法: 基于品种和成熟度的分层抽样")
        print(f"数据对齐: 以儿茶素数据为基准")
        print(f"处理的儿茶素物质数量: {len(catechin_names)}")
        print(f"平均测试集R²: {summary_df['测试集R²'].mean():.4f}")
        print(f"平均测试集RPD: {summary_df['测试集RPD'].mean():.4f}")

        # 最佳和最差模型
        best_model = summary_df.loc[summary_df['测试集R²'].idxmax()]
        worst_model = summary_df.loc[summary_df['测试集R²'].idxmin()]

        print(f"\n最佳模型:")
        print(f"  儿茶素物质: {best_model['儿茶素物质']}")
        print(f"  测试集R²: {best_model['测试集R²']:.4f}")
        print(f"  测试集RPD: {best_model['测试集RPD']:.4f}")

        print(f"\n最差模型:")
        print(f"  儿茶素物质: {worst_model['儿茶素物质']}")
        print(f"  测试集R²: {worst_model['测试集R²']:.4f}")
        print(f"  测试集RPD: {worst_model['测试集RPD']:.4f}")

        print(f"\n生成的文件:")
        print(f"  1. {output_dir}/dl_summary_results.csv - 汇总结果")
        print(f"  2. {output_dir}/dl_performance_comparison.png - 性能比较图")
        print(f"  3. {output_dir}/dl_detailed_predictions.csv - 详细预测结果")
        print(f"  4. {output_dir}/stratify_info.csv - 分层抽样信息")
        print(f"  5. {output_dir}/dl_model_report.txt - 模型报告")
        print(f"  6. 各儿茶素物质的回归分析图")
        print(f"  7. 各儿茶素物质的损失历史图")
        print(f"{'=' * 60}")

    except Exception as e:
        print(f"错误: {e}")
        traceback.print_exc()
        print("\n请检查:")
        print("1. 文件路径是否正确")
        print("2. 文件格式是否正确（应为.xlsx格式）")
        print("3. 文件内容是否符合要求（前三列应为Variety, Maturity, GETI）")
        print("4. 必要的Python库是否已安装: pandas, numpy, scipy, scikit-learn, matplotlib, seaborn, torch")
        print("5. 确保已安装PyTorch: pip install torch")


if __name__ == "__main__":
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)

    # 检查CUDA是否可用
    if torch.cuda.is_available():
        print(f"CUDA可用，使用GPU加速")
        print(f"GPU设备: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA不可用，使用CPU")

    # 运行主函数
    main()