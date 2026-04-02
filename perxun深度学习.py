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
import seaborn as sns
import time
import json

# 修复matplotlib后端问题
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
import traceback
import os

# 导入深度学习相关库
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, CyclicLR
import torch.nn.functional as F

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
    def second_derivative(X):
        """二阶导数"""
        return np.gradient(np.gradient(X, axis=1), axis=1)

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
    def detrend(X):
        """去趋势处理"""
        from scipy import signal
        X_detrend = np.zeros_like(X)
        for i in range(X.shape[0]):
            X_detrend[i, :] = signal.detrend(X[i, :])
        return X_detrend

    @staticmethod
    def snv_d1st(X):
        """SNV + 一阶导数"""
        X_snv = SpectrumPreprocessor.snv(X)
        return SpectrumPreprocessor.first_derivative(X_snv)

    @staticmethod
    def sg_snv_d1st(X):
        """SG平滑 + SNV + 一阶导数"""
        X_sg = SpectrumPreprocessor.sg_smoothing(X)
        X_snv = SpectrumPreprocessor.snv(X_sg)
        return SpectrumPreprocessor.first_derivative(X_snv)

    @staticmethod
    def msc_sg_snv_d1st(X):
        """MSC + SG平滑 + SNV + 一阶导数"""
        X_msc = SpectrumPreprocessor.msc(X)
        X_sg = SpectrumPreprocessor.sg_smoothing(X_msc)
        X_snv = SpectrumPreprocessor.snv(X_sg)
        return SpectrumPreprocessor.first_derivative(X_snv)

    @staticmethod
    def detrend_snv_d1st(X):
        """去趋势 + SNV + 一阶导数"""
        X_detrend = SpectrumPreprocessor.detrend(X)
        X_snv = SpectrumPreprocessor.snv(X_detrend)
        return SpectrumPreprocessor.first_derivative(X_snv)

    @staticmethod
    def apply_preprocessing(X, method='snv_d1st'):
        """应用指定的预处理方法"""
        preprocess_methods = {
            'snv_d1st': SpectrumPreprocessor.snv_d1st,
            'sg_snv_d1st': SpectrumPreprocessor.sg_snv_d1st,
            'msc_sg_snv_d1st': SpectrumPreprocessor.msc_sg_snv_d1st,
            'detrend_snv_d1st': SpectrumPreprocessor.detrend_snv_d1st,
            'snv': SpectrumPreprocessor.snv,
            'msc': SpectrumPreprocessor.msc,
            'sg_smoothing': SpectrumPreprocessor.sg_smoothing
        }

        if method not in preprocess_methods:
            raise ValueError(f"不支持的预处理方法: {method}")

        print(f"应用预处理方法: {method}")
        return preprocess_methods[method](X)

    @staticmethod
    def evaluate_preprocessing_methods(X, y, train_idx, test_idx, catechin_names):
        """评估不同预处理方法的效果"""
        methods = ['snv_d1st', 'sg_snv_d1st', 'msc_sg_snv_d1st', 'detrend_snv_d1st', 'snv', 'msc', 'sg_smoothing']
        results = {}

        for method in methods:
            print(f"\n评估预处理方法: {method}")
            try:
                X_processed = SpectrumPreprocessor.apply_preprocessing(X, method)
                X_train, X_test = X_processed[train_idx], X_processed[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # 使用简单的PLSR模型快速评估
                from sklearn.cross_decomposition import PLSRegression
                from sklearn.model_selection import cross_val_score

                r2_scores = []
                for i in range(y.shape[1]):
                    pls = PLSRegression(n_components=min(10, X_train.shape[1]))
                    scores = cross_val_score(pls, X_train, y_train[:, i], cv=5, scoring='r2')
                    r2_scores.append(np.mean(scores))

                avg_r2 = np.mean(r2_scores)
                results[method] = avg_r2
                print(f"  平均交叉验证R²: {avg_r2:.4f}")

            except Exception as e:
                print(f"  方法 {method} 出错: {e}")
                results[method] = -1

        # 找出最佳预处理方法
        if results:
            best_method = max(results.items(), key=lambda x: x[1])
            print(f"\n最佳预处理方法: {best_method[0]} (R²: {best_method[1]:.4f})")
            return results, best_method[0]
        else:
            print("\n没有有效的预处理方法结果")
            return results, 'snv_d1st'  # 默认返回


class SpectrumDataLoader:
    """数据加载器"""

    @staticmethod
    def load_spectrum_data(file_path):
        """加载光谱数据"""
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
        """加载儿茶素数据"""
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
        """以儿茶素数据为基准对齐数据"""
        print("\n以儿茶素数据为基准进行数据对齐...")
        print(f"儿茶素数据样本数: {len(catechin_ids)}")
        print(f"光谱数据样本数: {len(spectrum_ids)}")

        # 创建光谱数据的映射字典
        spectrum_dict = {}
        for i, geti in enumerate(spectrum_ids):
            if geti not in spectrum_dict:
                spectrum_dict[geti] = {
                    'index': i,
                    'X': spectrum_X[i],
                    'variety': spectrum_variety[i],
                    'maturity': spectrum_maturity[i]
                }

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
                    continue
                if catechin_maturity[i] != spectrum_info['maturity']:
                    continue

                aligned_X.append(spectrum_info['X'])
                aligned_y.append(catechin_y[i])
                aligned_ids.append(geti)
                aligned_variety.append(catechin_variety[i])
                aligned_maturity.append(catechin_maturity[i])
            else:
                missing_count += 1

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


class CorrelationAnalyzer:
    """相关性分析类"""

    @staticmethod
    def calculate_correlation_matrix(y, catechin_names):
        """计算儿茶素物质之间的皮尔逊相关系数矩阵"""
        print("\n计算儿茶素物质之间的皮尔逊相关系数矩阵...")

        # 计算相关系数矩阵
        correlation_matrix = np.corrcoef(y.T)

        # 创建热力图
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix,
                    xticklabels=catechin_names,
                    yticklabels=catechin_names,
                    annot=True, cmap='coolwarm',
                    center=0, fmt='.2f',
                    mask=mask, square=True)
        plt.title('儿茶素物质皮尔逊相关系数矩阵', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        # 保存热力图
        os.makedirs("correlation_analysis", exist_ok=True)
        plt.savefig('correlation_analysis/catechin_correlation_matrix.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        print("相关系数矩阵热力图已保存为: correlation_analysis/catechin_correlation_matrix.png")

        return correlation_matrix

    @staticmethod
    def identify_strong_correlations(correlation_matrix, catechin_names, threshold=0.7):
        """识别强相关性组"""
        print(f"\n识别强相关性组 (阈值: {threshold})...")

        strong_correlations = []
        correlation_groups = []
        visited = set()

        n = len(catechin_names)

        # 找出所有强相关性对
        for i in range(n):
            for j in range(i + 1, n):
                if abs(correlation_matrix[i, j]) >= threshold:
                    strong_correlations.append((i, j, correlation_matrix[i, j]))

        # 按相关性强度排序
        strong_correlations.sort(key=lambda x: abs(x[2]), reverse=True)

        # 构建相关性组
        for i, j, corr in strong_correlations:
            group_found = False
            for group in correlation_groups:
                if i in group or j in group:
                    if i not in group:
                        group.add(i)
                    if j not in group:
                        group.add(j)
                    group_found = True
                    break

            if not group_found:
                new_group = {i, j}
                correlation_groups.append(new_group)

        # 处理未分组的儿茶素
        all_indices = set(range(n))
        grouped_indices = set()
        for group in correlation_groups:
            grouped_indices.update(group)

        ungrouped_indices = all_indices - grouped_indices
        for idx in ungrouped_indices:
            correlation_groups.append({idx})

        # 打印相关性组信息
        print("强相关性分组结果:")
        for i, group in enumerate(correlation_groups):
            group_names = [catechin_names[idx] for idx in group]
            if len(group) > 1:
                # 计算组内平均相关性
                group_correlations = []
                group_indices = list(group)
                for a in range(len(group_indices)):
                    for b in range(a + 1, len(group_indices)):
                        corr_val = correlation_matrix[group_indices[a], group_indices[b]]
                        group_correlations.append(abs(corr_val))
                avg_corr = np.mean(group_correlations) if group_correlations else 0
                print(f"  组 {i + 1}: {group_names} (平均相关性: {avg_corr:.3f})")
            else:
                print(f"  组 {i + 1}: {group_names} (独立)")

        return correlation_groups


class AdvancedMultiTaskNeuralNetwork(nn.Module):
    """改进的多任务深度学习神经网络模型"""

    def __init__(self, input_size, output_size, hidden_sizes=[512, 256, 128, 64],
                 dropout_rate=0.3, use_batch_norm=True, use_residual=False):
        super(AdvancedMultiTaskNeuralNetwork, self).__init__()

        self.output_size = output_size
        self.use_residual = use_residual

        layers = []
        prev_size = input_size

        # 构建共享隐藏层
        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, hidden_size))

            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))

            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))

            prev_size = hidden_size

        # 共享特征提取器
        self.shared_encoder = nn.Sequential(*layers)

        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(prev_size, prev_size // 2),
            nn.ReLU(),
            nn.Linear(prev_size // 2, prev_size),
            nn.Sigmoid()
        )

        # 多任务输出层，每个任务有独立的小网络
        self.task_heads = nn.ModuleList()
        for _ in range(output_size):
            task_head = nn.Sequential(
                nn.Linear(prev_size, prev_size // 2),
                nn.BatchNorm1d(prev_size // 2) if use_batch_norm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(dropout_rate / 2),
                nn.Linear(prev_size // 2, prev_size // 4),
                nn.BatchNorm1d(prev_size // 4) if use_batch_norm else nn.Identity(),
                nn.ReLU(),
                nn.Linear(prev_size // 4, 1)
            )
            self.task_heads.append(task_head)

        self._initialize_weights()

    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)  # 小偏置
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        shared_features = self.shared_encoder(x)

        # 应用注意力机制
        attention_weights = self.attention(shared_features)
        attended_features = shared_features * attention_weights

        outputs = []
        for head in self.task_heads:
            output = head(attended_features)
            outputs.append(output)

        return torch.cat(outputs, dim=1)


class AdvancedMultiTaskDLRegression:
    """改进的多任务深度学习回归模型"""

    class AdvancedScaler:
        """改进的数据标准化类，处理零方差特征"""

        def __init__(self, epsilon=1e-8):
            self._mean = 0
            self._std = 0
            self.epsilon = epsilon
            self.zero_variance_mask = None

        def fit_transform(self, traindata):
            self._mean = traindata.mean(axis=0)
            self._std = traindata.std(axis=0)

            # 标记零方差特征
            self.zero_variance_mask = self._std < self.epsilon
            if np.any(self.zero_variance_mask):
                print(f"警告: 发现 {np.sum(self.zero_variance_mask)} 个零方差特征")
                # 将零方差特征的标准差设为1，避免除零
                self._std[self.zero_variance_mask] = 1.0

            return (traindata - self._mean) / (self._std + self.epsilon)

        def transform(self, testdata):
            return (testdata - self._mean) / (self._std + self.epsilon)

        def inverse_transform(self, scaled_data):
            return scaled_data * (self._std + self.epsilon) + self._mean

    def __init__(self, output_size, hidden_sizes=[512, 256, 128, 64], dropout_rate=0.3,
                 learning_rate=0.0001, batch_size=16, epochs=3000,
                 patience=100, weight_decay=1e-5, use_batch_norm=True,
                 use_residual=False, scheduler_type='plateau', device=None):

        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.weight_decay = weight_decay
        self.use_batch_norm = use_batch_norm
        self.use_residual = use_residual
        self.scheduler_type = scheduler_type

        # 设备选择
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.model = None
        self.X_scaler = self.AdvancedScaler()
        self.y_scaler = self.AdvancedScaler()
        self.loss_history = {'train': [], 'val': []}
        self.r2_history = {'train': [], 'val': []}
        self.best_epoch = 0

    def _create_optimizer(self, model):
        """创建优化器"""
        # 使用AdamW优化器，对偏置使用不同的学习率
        param_groups = []
        for name, param in model.named_parameters():
            if 'bias' in name:
                param_groups.append({'params': param, 'weight_decay': 0.0})
            else:
                param_groups.append({'params': param})

        optimizer = optim.AdamW(param_groups, lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer

    def _create_scheduler(self, optimizer):
        """创建学习率调度器"""
        if self.scheduler_type == 'plateau':
            return ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)
        elif self.scheduler_type == 'cosine':
            return CosineAnnealingLR(optimizer, T_max=self.epochs // 10, eta_min=self.learning_rate / 100)
        elif self.scheduler_type == 'cyclic':
            return CyclicLR(optimizer, base_lr=self.learning_rate / 10, max_lr=self.learning_rate,
                            step_size_up=self.epochs // 20, mode='triangular2')
        else:
            return None

    def fit(self, X, y, verbose=True, validation_split=0.1):
        """训练改进的多任务深度学习模型"""
        # 数据标准化
        X_scaled = self.X_scaler.fit_transform(X)
        y_scaled = self.y_scaler.fit_transform(y)

        # 转换为PyTorch张量
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.FloatTensor(y_scaled).to(self.device)

        # 初始化模型
        input_size = X.shape[1]
        self.model = AdvancedMultiTaskNeuralNetwork(
            input_size=input_size,
            output_size=self.output_size,
            hidden_sizes=self.hidden_sizes,
            dropout_rate=self.dropout_rate,
            use_batch_norm=self.use_batch_norm,
            use_residual=self.use_residual
        ).to(self.device)

        # 定义损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = self._create_optimizer(self.model)
        scheduler = self._create_scheduler(optimizer)

        # 从训练集中划分验证集
        dataset_size = len(X)
        val_size = int(dataset_size * validation_split)
        train_size = dataset_size - val_size

        # 确保训练集至少有一个batch
        if train_size < self.batch_size:
            self.batch_size = max(4, train_size // 2)
            print(f"调整batch_size为: {self.batch_size}")

        indices = np.arange(dataset_size)
        np.random.shuffle(indices)

        train_idx = indices[:train_size]
        val_idx = indices[train_size:]

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
            print(f"batch_size: {self.batch_size}")
            print(f"网络结构: 输入层({input_size}) -> ", end="")
            for i, size in enumerate(self.hidden_sizes):
                print(f"隐藏层{i + 1}({size}) -> ", end="")
            print(f"多任务输出层({self.output_size})")
            print(f"使用注意力机制: 是")
            print("开始训练...")

        # 早停设置
        best_val_loss = float('inf')
        best_val_r2 = -float('inf')
        patience_counter = 0
        best_model_state = None

        # 训练循环
        for epoch in range(self.epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            train_batches = 0
            train_predictions = []
            train_targets = []

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)

                # 添加L1正则化
                l1_lambda = 0.0001
                l1_norm = sum(p.abs().sum() for p in self.model.parameters())
                loss = loss + l1_lambda * l1_norm

                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()

                train_loss += loss.item()
                train_batches += 1

                train_predictions.append(outputs.detach().cpu().numpy())
                train_targets.append(batch_y.detach().cpu().numpy())

            avg_train_loss = train_loss / train_batches
            self.loss_history['train'].append(avg_train_loss)

            # 计算训练集R2
            train_predictions = np.vstack(train_predictions)
            train_targets = np.vstack(train_targets)
            train_r2 = r2_score(train_targets, train_predictions, multioutput='uniform_average')
            self.r2_history['train'].append(train_r2)

            # 验证阶段
            self.model.eval()
            val_loss = 0.0
            val_batches = 0
            val_predictions = []
            val_targets = []

            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    val_batches += 1

                    val_predictions.append(outputs.cpu().numpy())
                    val_targets.append(batch_y.cpu().numpy())

            avg_val_loss = val_loss / val_batches
            self.loss_history['val'].append(avg_val_loss)

            # 计算验证集R2
            val_predictions = np.vstack(val_predictions)
            val_targets = np.vstack(val_targets)
            val_r2 = r2_score(val_targets, val_predictions, multioutput='uniform_average')
            self.r2_history['val'].append(val_r2)

            # 学习率调整
            if scheduler is not None:
                if self.scheduler_type == 'plateau':
                    scheduler.step(avg_val_loss)
                else:
                    scheduler.step()

            # 早停检查（基于R2）
            if val_r2 > best_val_r2:
                best_val_r2 = val_r2
                best_val_loss = avg_val_loss
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                self.best_epoch = epoch
            else:
                patience_counter += 1

            # 打印训练进度
            if verbose and (epoch + 1) % 50 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch [{epoch + 1:4d}/{self.epochs}] | "
                      f"Train Loss: {avg_train_loss:.6f}, R²: {train_r2:.4f} | "
                      f"Val Loss: {avg_val_loss:.6f}, R²: {val_r2:.4f} | "
                      f"LR: {current_lr:.6f} | "
                      f"Best R²: {best_val_r2:.4f}")

            # 检查早停
            if patience_counter >= self.patience:
                if verbose:
                    print(f"早停触发，在Epoch {epoch + 1}停止训练，最佳R²: {best_val_r2:.4f}")
                break

        # 加载最佳模型
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        if verbose:
            print(f"训练完成，最佳验证R²: {best_val_r2:.4f}，最佳验证损失: {best_val_loss:.6f}")
            print(f"最佳模型在Epoch {self.best_epoch + 1}")

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

    def plot_training_history(self, save_path=None):
        """绘制训练历史"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        epochs = range(1, len(self.loss_history['train']) + 1)

        # 损失曲线
        axes[0, 0].plot(epochs, self.loss_history['train'], 'b-', label='训练损失', alpha=0.8)
        axes[0, 0].plot(epochs, self.loss_history['val'], 'r-', label='验证损失', alpha=0.8)
        axes[0, 0].axvline(x=self.best_epoch + 1, color='g', linestyle='--', alpha=0.5, label='最佳模型')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss (MSE)')
        axes[0, 0].set_title('训练和验证损失历史')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # R2曲线
        axes[0, 1].plot(epochs, self.r2_history['train'], 'b-', label='训练R²', alpha=0.8)
        axes[0, 1].plot(epochs, self.r2_history['val'], 'r-', label='验证R²', alpha=0.8)
        axes[0, 1].axvline(x=self.best_epoch + 1, color='g', linestyle='--', alpha=0.5, label='最佳模型')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('R² Score')
        axes[0, 1].set_title('训练和验证R²历史')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 损失对数曲线
        axes[1, 0].semilogy(epochs, self.loss_history['train'], 'b-', label='训练损失', alpha=0.8)
        axes[1, 0].semilogy(epochs, self.loss_history['val'], 'r-', label='验证损失', alpha=0.8)
        axes[1, 0].axvline(x=self.best_epoch + 1, color='g', linestyle='--', alpha=0.5, label='最佳模型')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Log Loss (MSE)')
        axes[1, 0].set_title('训练和验证损失（对数坐标）')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def calculate_rpd(y_true, y_pred):
    """计算RPD指标"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    std_dev = np.std(y_true)

    if rmse == 0:
        return float('inf')

    rpd = std_dev / rmse
    return rpd


def evaluate_multi_task_predictions(y_true, y_pred, catechin_names, dataset_name="测试集"):
    """评估多任务预测结果"""
    print(f"\n{dataset_name}多任务预测评估结果:")
    print("=" * 120)

    all_metrics = {}

    for i, catechin_name in enumerate(catechin_names):
        y_true_single = y_true[:, i]
        y_pred_single = y_pred[:, i]

        # 移除NaN值
        mask = ~np.isnan(y_true_single) & ~np.isnan(y_pred_single)
        y_true_single = y_true_single[mask]
        y_pred_single = y_pred_single[mask]

        if len(y_true_single) < 2:
            print(f"{catechin_name:>15}: 样本数不足，无法计算指标")
            continue

        metrics = {}
        metrics['mse'] = mean_squared_error(y_true_single, y_pred_single)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true_single, y_pred_single)
        metrics['r2'] = r2_score(y_true_single, y_pred_single)
        metrics['rpd'] = calculate_rpd(y_true_single, y_pred_single)
        metrics['bias'] = np.mean(y_pred_single - y_true_single)

        # 计算NSE (Nash-Sutcliffe效率系数)
        mean_obs = np.mean(y_true_single)
        sst = np.sum((y_true_single - mean_obs) ** 2)
        sse = np.sum((y_true_single - y_pred_single) ** 2)
        metrics['nse'] = 1 - sse / sst if sst > 0 else -float('inf')

        # 评估RPD等级
        if metrics['rpd'] > 2.5:
            rpd_level = "优秀"
        elif metrics['rpd'] > 2.0:
            rpd_level = "良好"
        elif metrics['rpd'] > 1.8:
            rpd_level = "可接受"
        else:
            rpd_level = "需要改进"

        all_metrics[catechin_name] = metrics

        print(f"{catechin_name:>20}: "
              f"R²={metrics['r2']:.4f}, "
              f"RMSE={metrics['rmse']:.4f}, "
              f"MAE={metrics['mae']:.4f}, "
              f"RPD={metrics['rpd']:.4f} ({rpd_level}), "
              f"Bias={metrics['bias']:.4f}, "
              f"NSE={metrics['nse']:.4f}")

    # 计算平均指标
    if all_metrics:
        avg_r2 = np.mean([metrics['r2'] for metrics in all_metrics.values()])
        avg_rmse = np.mean([metrics['rmse'] for metrics in all_metrics.values()])
        avg_rpd = np.mean([metrics['rpd'] for metrics in all_metrics.values()])
        avg_mae = np.mean([metrics['mae'] for metrics in all_metrics.values()])
        avg_nse = np.mean([metrics['nse'] for metrics in all_metrics.values()])

        print("-" * 120)
        print(f"平均指标: R²={avg_r2:.4f}, RMSE={avg_rmse:.4f}, MAE={avg_mae:.4f}, RPD={avg_rpd:.4f}, NSE={avg_nse:.4f}")
        print("=" * 120)
    else:
        print("没有有效的评估结果")

    return all_metrics


def create_stratified_split(X, variety, maturity, test_size=0.2, random_state=42):
    """创建基于品种和成熟度的分层抽样划分"""
    stratify_labels = [f"{v}_{m}" for v, m in zip(variety, maturity)]

    unique_labels = np.unique(stratify_labels)
    print(f"分层标签数量: {len(unique_labels)}")

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)

    for train_index, test_index in sss.split(X, stratify_labels):
        return train_index, test_index


def plot_multi_task_results(y_true, y_pred, catechin_names, dataset_name="测试集"):
    """绘制多任务预测结果图"""
    n_catechins = len(catechin_names)
    n_cols = min(4, n_catechins)
    n_rows = (n_catechins + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5))
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()

    for i, (ax, catechin_name) in enumerate(zip(axes, catechin_names)):
        if i >= n_catechins:
            ax.axis('off')
            continue

        y_true_single = y_true[:, i]
        y_pred_single = y_pred[:, i]

        # 移除NaN值
        mask = ~np.isnan(y_true_single) & ~np.isnan(y_pred_single)
        y_true_single = y_true_single[mask]
        y_pred_single = y_pred_single[mask]

        if len(y_true_single) < 2:
            ax.text(0.5, 0.5, f'{catechin_name}\n数据不足',
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=12)
            ax.axis('off')
            continue

        # 创建散点图
        scatter = ax.scatter(y_true_single, y_pred_single, alpha=0.6, color='blue', s=50)

        # 添加回归线和理想线
        min_val = min(y_true_single.min(), y_pred_single.min())
        max_val = max(y_true_single.max(), y_pred_single.max())
        margin = (max_val - min_val) * 0.05
        min_val -= margin
        max_val += margin

        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='1:1线', linewidth=2)

        # 计算回归线
        if len(y_true_single) > 1:
            coeffs = np.polyfit(y_true_single, y_pred_single, 1)
            poly = np.poly1d(coeffs)
            x_fit = np.linspace(min_val, max_val, 100)
            y_fit = poly(x_fit)
            ax.plot(x_fit, y_fit, 'g-', alpha=0.8, label='回归线', linewidth=2)

        ax.set_xlabel('真实值 (mg/g)', fontsize=12)
        ax.set_ylabel('预测值 (mg/g)', fontsize=12)
        ax.set_title(f'{catechin_name}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # 计算并添加指标文本
        r2 = r2_score(y_true_single, y_pred_single)
        rmse = np.sqrt(mean_squared_error(y_true_single, y_pred_single))
        mae = mean_absolute_error(y_true_single, y_pred_single)
        rpd = calculate_rpd(y_true_single, y_pred_single)

        text_str = f'R² = {r2:.3f}\nRMSE = {rmse:.3f}\nMAE = {mae:.3f}\nRPD = {rpd:.3f}'
        ax.text(0.05, 0.95, text_str, transform=ax.transAxes,
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

        # 设置坐标轴范围
        ax.set_xlim([min_val, max_val])
        ax.set_ylim([min_val, max_val])
        ax.set_aspect('equal', adjustable='box')

    plt.suptitle(f'{dataset_name} - 多任务预测结果', fontsize=16, fontweight='bold')
    plt.tight_layout()

    filename = f'improved_multi_task_{dataset_name}_results.png'.replace(' ', '_')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"多任务预测结果图已保存为: {filename}")


def main():
    """主函数"""
    # 文件路径
    spectrum_path = r"C:\Users\Mayn\Desktop\实验MAX\鲜叶-儿茶素对应\儿茶素数据\儿茶素对应光谱 - 副本 - 副本.xlsx"
    catechin_path = r"C:\Users\Mayn\Desktop\实验MAX\鲜叶-儿茶素对应\儿茶素数据\数据-儿茶素-毫克每克 - 副本 - 副本.xlsx"

    print("=" * 80)
    print("茶叶儿茶素含量改进多任务预测模型")
    print("=" * 80)

    try:
        # 1. 加载数据
        print("\n1. 加载数据...")
        X, spectrum_ids, wavelength_names, spectrum_variety, spectrum_maturity = SpectrumDataLoader.load_spectrum_data(
            spectrum_path)
        y, catechin_ids, catechin_names, catechin_variety, catechin_maturity = SpectrumDataLoader.load_catechin_data(
            catechin_path)

        # 2. 数据对齐
        print("\n2. 数据对齐...")
        X_aligned, y_aligned, aligned_ids, aligned_variety, aligned_maturity = SpectrumDataLoader.align_data_by_catechin(
            spectrum_ids, X, spectrum_variety, spectrum_maturity,
            catechin_ids, y, catechin_variety, catechin_maturity
        )

        print(f"\n对齐后数据统计:")
        print(f"样本数量: {X_aligned.shape[0]}")
        print(f"光谱特征数量: {X_aligned.shape[1]}")
        print(f"儿茶素物质数量: {y_aligned.shape[1]}")
        print(f"儿茶素物质: {catechin_names}")

        # 检查数据质量
        print("\n数据质量检查:")
        for i, name in enumerate(catechin_names):
            non_zero = np.sum(y_aligned[:, i] > 0)
            mean_val = np.mean(y_aligned[:, i])
            std_val = np.std(y_aligned[:, i])
            print(f"  {name}: 非零样本={non_zero}, 均值={mean_val:.4f}, 标准差={std_val:.4f}")

        # 3. 相关性分析
        print("\n3. 儿茶素物质相关性分析...")
        correlation_matrix = CorrelationAnalyzer.calculate_correlation_matrix(y_aligned, catechin_names)
        correlation_groups = CorrelationAnalyzer.identify_strong_correlations(correlation_matrix, catechin_names,
                                                                              threshold=0.3)

        # 4. 创建分层抽样划分
        print("\n4. 创建分层抽样划分...")
        train_idx, test_idx = create_stratified_split(
            X_aligned, aligned_variety, aligned_maturity, test_size=0.15, random_state=42
        )

        print(f"训练集样本数: {len(train_idx)}")
        print(f"测试集样本数: {len(test_idx)}")

        # 5. 评估不同预处理方法
        print("\n5. 评估不同预处理方法...")
        preprocessing_results, best_preprocessing_method = SpectrumPreprocessor.evaluate_preprocessing_methods(
            X_aligned, y_aligned, train_idx, test_idx, catechin_names
        )

        # 6. 应用最佳预处理方法
        print(f"\n6. 应用最佳预处理方法: {best_preprocessing_method}")
        X_processed = SpectrumPreprocessor.apply_preprocessing(X_aligned, best_preprocessing_method)

        # 划分训练集和测试集
        X_train, X_test = X_processed[train_idx], X_processed[test_idx]
        y_train, y_test = y_aligned[train_idx], y_aligned[test_idx]

        print(f"训练集形状: X={X_train.shape}, y={y_train.shape}")
        print(f"测试集形状: X={X_test.shape}, y={y_test.shape}")

        # 7. 多任务模型训练
        print("\n7. 训练改进的多任务深度学习模型...")

        # 设置随机种子
        torch.manual_seed(42)
        np.random.seed(42)

        # 根据数据量调整参数
        input_size = X_train.shape[1]
        output_size = y_train.shape[1]

        # 自动调整网络结构和训练参数
        if X_train.shape[0] < 100:  # 小样本
            hidden_sizes = [256, 128, 64, 32]
            batch_size = 8
            learning_rate = 0.0005
            epochs = 2000
        elif X_train.shape[0] < 500:  # 中等样本
            hidden_sizes = [512, 256, 128, 64]
            batch_size = 32
            learning_rate = 0.0002
            epochs = 3000


        # 创建改进的多任务模型
        improved_model = AdvancedMultiTaskDLRegression(
            output_size=output_size,
            hidden_sizes=hidden_sizes,
            dropout_rate=0.3,  # 较高的dropout防止过拟合
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            patience=200,  # 更多的耐心
            weight_decay=1e-6,
            use_batch_norm=True,
            use_residual=True,
            scheduler_type='cosine'  # 使用余弦退火
        )

        # 训练模型
        start_time = time.time()
        improved_model.fit(X_train, y_train, verbose=True, validation_split=0.2)
        end_time = time.time()
        training_time = end_time - start_time

        print(f"改进模型训练时间: {training_time:.2f}秒")

        # 绘制训练历史
        improved_model.plot_training_history('improved_multi_task_training_history.png')
        print("改进模型训练历史图已保存为: improved_multi_task_training_history.png")

        # 8. 模型预测和评估
        print("\n8. 模型预测和评估...")

        # 训练集预测
        y_train_pred = improved_model.predict(X_train)
        train_metrics = evaluate_multi_task_predictions(y_train, y_train_pred, catechin_names, "训练集")

        # 测试集预测
        y_test_pred = improved_model.predict(X_test)
        test_metrics = evaluate_multi_task_predictions(y_test, y_test_pred, catechin_names, "测试集")

        # 绘制预测结果图
        plot_multi_task_results(y_train, y_train_pred, catechin_names, "训练集")
        plot_multi_task_results(y_test, y_test_pred, catechin_names, "测试集")

        # 9. 保存结果
        print("\n9. 保存结果...")
        output_dir = "improved_multi_task_results"
        os.makedirs(output_dir, exist_ok=True)

        # 保存预测结果
        predictions_df = pd.DataFrame({
            'GETI': np.concatenate([aligned_ids[train_idx], aligned_ids[test_idx]]),
            '品种': np.concatenate([aligned_variety[train_idx], aligned_variety[test_idx]]),
            '成熟度': np.concatenate([aligned_maturity[train_idx], aligned_maturity[test_idx]]),
            '数据集': ['训练集'] * len(train_idx) + ['测试集'] * len(test_idx)
        })

        for i, catechin_name in enumerate(catechin_names):
            predictions_df[f'{catechin_name}_真实值'] = np.concatenate([y_train[:, i], y_test[:, i]])
            predictions_df[f'{catechin_name}_预测值'] = np.concatenate([y_train_pred[:, i], y_test_pred[:, i]])
            predictions_df[f'{catechin_name}_残差'] = predictions_df[f'{catechin_name}_预测值'] - predictions_df[
                f'{catechin_name}_真实值']
            predictions_df[f'{catechin_name}_相对误差'] = 100 * predictions_df[f'{catechin_name}_残差'] / (
                        predictions_df[f'{catechin_name}_真实值'] + 1e-8)

        predictions_path = os.path.join(output_dir, "improved_multi_task_predictions.csv")
        predictions_df.to_csv(predictions_path, index=False, encoding='utf-8-sig')
        print(f"详细预测结果已保存到: {predictions_path}")

        # 保存评估指标
        metrics_data = []
        for catechin_name in catechin_names:
            if catechin_name in train_metrics and catechin_name in test_metrics:
                metrics_data.append({
                    '儿茶素物质': catechin_name,
                    '训练集R²': train_metrics[catechin_name]['r2'],
                    '测试集R²': test_metrics[catechin_name]['r2'],
                    '训练集RMSE': train_metrics[catechin_name]['rmse'],
                    '测试集RMSE': test_metrics[catechin_name]['rmse'],
                    '训练集MAE': train_metrics[catechin_name]['mae'],
                    '测试集MAE': test_metrics[catechin_name]['mae'],
                    '训练集RPD': train_metrics[catechin_name]['rpd'],
                    '测试集RPD': test_metrics[catechin_name]['rpd'],
                    '训练集NSE': train_metrics[catechin_name]['nse'],
                    '测试集NSE': test_metrics[catechin_name]['nse'],
                    '训练集Bias': train_metrics[catechin_name]['bias'],
                    '测试集Bias': test_metrics[catechin_name]['bias']
                })

        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data)
            metrics_path = os.path.join(output_dir, "improved_multi_task_metrics.csv")
            metrics_df.to_csv(metrics_path, index=False, encoding='utf-8-sig')
            print(f"评估指标已保存到: {metrics_path}")

        # 10. 生成总结报告
        print("\n10. 生成总结报告...")

        if test_metrics:
            avg_train_r2 = np.mean([m['r2'] for m in train_metrics.values()])
            avg_test_r2 = np.mean([m['r2'] for m in test_metrics.values()])
            avg_test_rpd = np.mean([m['rpd'] for m in test_metrics.values()])
            avg_test_nse = np.mean([m['nse'] for m in test_metrics.values()])

            # RPD等级统计
            rpd_levels = []
            for metrics in test_metrics.values():
                rpd = metrics['rpd']
                if rpd > 2.5:
                    rpd_levels.append('优秀')
                elif rpd > 2.0:
                    rpd_levels.append('良好')
                elif rpd > 1.8:
                    rpd_levels.append('可接受')
                else:
                    rpd_levels.append('需改进')

            report_path = os.path.join(output_dir, "improved_multi_task_report.txt")
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("茶叶儿茶素含量改进多任务预测模型报告\n")
                f.write("=" * 70 + "\n\n")
                f.write(f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"样本数量: {X_aligned.shape[0]}\n")
                f.write(f"训练集样本数: {len(train_idx)}\n")
                f.write(f"测试集样本数: {len(test_idx)}\n")
                f.write(f"儿茶素物质数量: {len(catechin_names)}\n")
                f.write(f"最佳预处理方法: {best_preprocessing_method}\n")
                f.write(f"模型类型: 改进多任务深度学习（带注意力机制）\n")
                f.write(f"训练时间: {training_time:.2f}秒\n")
                f.write(f"最佳epoch: {improved_model.best_epoch + 1}\n\n")

                f.write("模型参数:\n")
                f.write(f"  隐藏层结构: {hidden_sizes}\n")
                f.write(f"  Dropout率: 0.4\n")
                f.write(f"  学习率: {learning_rate}\n")
                f.write(f"  Batch大小: {batch_size}\n")
                f.write(f"  权重衰减: 1e-5\n")
                f.write(f"  总Epochs: {epochs}\n")
                f.write(f"  早停Patience: 150\n")
                f.write(f"  调度器: cosine\n\n")

                f.write("性能总结:\n")
                f.write(f"  平均训练集R²: {avg_train_r2:.4f}\n")
                f.write(f"  平均测试集R²: {avg_test_r2:.4f}\n")
                f.write(f"  平均测试集RPD: {avg_test_rpd:.4f}\n")
                f.write(f"  平均测试集NSE: {avg_test_nse:.4f}\n\n")

                f.write("RPD等级分布:\n")
                f.write(f"  优秀 (RPD > 2.5): {rpd_levels.count('优秀')} 个\n")
                f.write(f"  良好 (2.0 < RPD ≤ 2.5): {rpd_levels.count('良好')} 个\n")
                f.write(f"  可接受 (1.8 < RPD ≤ 2.0): {rpd_levels.count('可接受')} 个\n")
                f.write(f"  需改进 (RPD ≤ 1.8): {rpd_levels.count('需改进')} 个\n")

            print(f"总结报告已保存到: {report_path}")

        # 11. 最终输出
        print(f"\n{'=' * 80}")
        print("改进多任务模型训练完成!")
        print(f"{'=' * 80}")
        if test_metrics:
            print(f"处理的儿茶素物质数量: {len(catechin_names)}")
            print(f"最佳预处理方法: {best_preprocessing_method}")
            print(f"平均测试集R²: {avg_test_r2:.4f}")
            print(f"平均测试集RPD: {avg_test_rpd:.4f}")
            print(f"平均测试集NSE: {avg_test_nse:.4f}")
            print(f"训练时间: {training_time:.2f}秒")
        print(f"\n生成的文件:")
        print(f"  1. {output_dir}/improved_multi_task_predictions.csv - 详细预测结果")
        print(f"  2. {output_dir}/improved_multi_task_metrics.csv - 评估指标")
        print(f"  3. {output_dir}/improved_multi_task_report.txt - 总结报告")
        print(f"  4. improved_multi_task_training_history.png - 训练历史图")
        print(f"  5. improved_multi_task_训练集_results.png - 训练集预测图")
        print(f"  6. improved_multi_task_测试集_results.png - 测试集预测图")
        print(f"  7. correlation_analysis/ - 相关性分析结果")
        print(f"{'=' * 80}")

    except Exception as e:
        print(f"错误: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    # 检查CUDA是否可用
    if torch.cuda.is_available():
        print(f"CUDA可用，使用GPU加速")
        print(f"GPU设备: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA不可用，使用CPU")

    # 运行主函数
    main()
