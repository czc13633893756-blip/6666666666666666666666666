import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score
import os
import warnings

# 忽略不必要的 sklearn 警告
warnings.filterwarnings("ignore", category=UserWarning)


# ==========================================
# 1. 宽度学习系统 (Broad Learning System) 实现
# ==========================================
class BroadLearningSystem:
    def __init__(self, n_feature_nodes=10, n_feature_mapped=10, n_enhancement_nodes=50, s=0.8, c=2 ** -30):
        self.n_feature_nodes = n_feature_nodes
        self.n_feature_mapped = n_feature_mapped
        self.n_enhancement_nodes = n_enhancement_nodes
        self.s = s
        self.c = c

        self.weights_feature = []
        self.bias_feature = []
        self.weights_enhance = None
        self.bias_enhance = None
        self.output_weights = None
        self.scaler_input = MinMaxScaler()
        self.onehot = OneHotEncoder(sparse_output=False)

    def _tansig(self, x):
        return np.tanh(x)

    def _sparse_autoencoder_weights(self, X):
        in_dim = X.shape[1]
        out_dim = self.n_feature_mapped
        W = 2 * np.random.random((in_dim, out_dim)) - 1
        b = 2 * np.random.random((1, out_dim)) - 1
        return W, b

    def fit(self, X, y):
        # 强制转换为 numpy array
        X = np.array(X)

        # 数据标准化
        X = self.scaler_input.fit_transform(X)

        # 处理 Y (One-hot 编码)
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        y = self.onehot.fit_transform(y)

        N = X.shape[0]

        # 1. 生成特征节点 (Feature Nodes)
        self.Z = []
        for i in range(self.n_feature_nodes):
            W, b = self._sparse_autoencoder_weights(X)
            self.weights_feature.append(W)
            self.bias_feature.append(b)
            Zi = self._tansig(np.dot(X, W) + b)
            self.Z.append(Zi)

        self.Z_concat = np.hstack(self.Z)

        # 2. 生成增强节点 (Enhancement Nodes)
        in_dim_enhance = self.Z_concat.shape[1]
        self.weights_enhance = 2 * np.random.random((in_dim_enhance, self.n_enhancement_nodes)) - 1
        self.bias_enhance = 2 * np.random.random((1, self.n_enhancement_nodes)) - 1

        H = self._tansig(np.dot(self.Z_concat, self.weights_enhance) + self.bias_enhance)

        # 3. 合并特征层和增强层
        A = np.hstack([self.Z_concat, H])

        # 4. 计算输出权重 (伪逆求解)
        try:
            lambda_I = np.eye(A.shape[1]) * self.c
            pseudo_inverse = np.linalg.inv(np.dot(A.T, A) + lambda_I).dot(A.T)
        except np.linalg.LinAlgError:
            pseudo_inverse = np.linalg.pinv(A)

        self.output_weights = np.dot(pseudo_inverse, y)
        print("BLS 模型训练完成。")

    def predict(self, X):
        X = np.array(X)
        X = self.scaler_input.transform(X)

        # 生成特征节点
        Z_list = []
        for i in range(self.n_feature_nodes):
            Zi = self._tansig(np.dot(X, self.weights_feature[i]) + self.bias_feature[i])
            Z_list.append(Zi)
        Z_concat = np.hstack(Z_list)

        # 生成增强节点
        H = self._tansig(np.dot(Z_concat, self.weights_enhance) + self.bias_enhance)

        # 合并
        A = np.hstack([Z_concat, H])

        # 预测
        y_pred_onehot = np.dot(A, self.output_weights)

        # 转回类别索引
        return np.argmax(y_pred_onehot, axis=1)


# ==========================================
# 2. 数据处理与主程序
# ==========================================

def load_and_preprocess_data(file_path):
    print(f"正在读取文件: {file_path}")

    if not os.path.exists(file_path):
        print(f"错误：找不到文件 {file_path}")
        return None, None, None

    try:
        df = pd.read_excel(file_path, header=0)
        feature_data = df.iloc[0:60, 1:31]
        target_data = df.iloc[0:60, 31]

        if feature_data.isnull().values.any():
            feature_data = feature_data.fillna(0)

        feature_names = feature_data.columns.tolist()
        return feature_data, target_data, feature_names

    except Exception as e:
        print(f"读取文件出错: {e}")
        return None, None, None


def generate_labels(target_series):
    target_series = pd.to_numeric(target_series, errors='coerce')
    p20 = np.percentile(target_series, 20)
    p80 = np.percentile(target_series, 80)

    print(f"抗性划分阈值: 高抗(HR) <= {p20:.4f}, 高感(HS) >= {p80:.4f}")

    labels = []
    for val in target_series:
        if val <= p20:
            labels.append(0)  # HR
        elif val >= p80:
            labels.append(2)  # HS
        else:
            labels.append(1)  # MR

    return np.array(labels), ["HR (高抗)", "MR (中抗)", "HS (高感)"]


def calculate_permutation_importance(model, X_val, y_val, feature_names):
    X_val_np = np.array(X_val)
    baseline_pred = model.predict(X_val_np)
    baseline_acc = accuracy_score(y_val, baseline_pred)

    importances = []

    print("\n正在计算特征重要性...")
    for i in range(X_val_np.shape[1]):
        X_permuted = X_val_np.copy()
        np.random.shuffle(X_permuted[:, i])
        perm_pred = model.predict(X_permuted)
        perm_acc = accuracy_score(y_val, perm_pred)
        importance = baseline_acc - perm_acc
        importances.append(importance)

    imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })

    return imp_df.sort_values(by='Importance', ascending=False)


# ==========================================
# 主执行逻辑
# ==========================================
if __name__ == "__main__":
    physio_file = r"C:\Users\Mayn\Desktop\海南项目\物理-生理指标.xlsx"

    # 1. 加载数据
    X, y_raw, feature_names = load_and_preprocess_data(physio_file)

    if X is not None:
        # 2. 生成标签
        y_labels, class_names = generate_labels(y_raw)

        # 3. 划分数据集
        X_train, X_test, y_train, y_test = train_test_split(X, y_labels, test_size=0.2, random_state=42,
                                                            stratify=y_labels)

        # 4. 训练模型
        # 若想提高精度，可尝试调整 n_enhancement_nodes (如 50 -> 100)
        bls = BroadLearningSystem(n_feature_nodes=50, n_feature_mapped=10, n_enhancement_nodes=100, s=0.8, c=2 ** -30)
        bls.fit(X_train, y_train)

        # ==========================================
        # 5. 结果评估 (新增：训练集精度 & Kappa)
        # ==========================================
        print("\n" + "=" * 30)
        print("MODEL EVALUATION")
        print("=" * 30)

        # A. 训练集表现 (Training Set Performance)
        y_train_pred = bls.predict(X_train)
        train_acc = accuracy_score(y_train, y_train_pred)
        train_kappa = cohen_kappa_score(y_train, y_train_pred)

        print(f"【训练集】精度 (Accuracy) : {train_acc:.4f}")
        print(f"【训练集】Kappa 系数       : {train_kappa:.4f}")
        print("-" * 30)

        # B. 测试集表现 (Test Set Performance)
        y_test_pred = bls.predict(X_test)
        test_acc = accuracy_score(y_test, y_test_pred)
        test_kappa = cohen_kappa_score(y_test, y_test_pred)

        print(f"【测试集】精度 (Accuracy) : {test_acc:.4f}")
        print(f"【测试集】Kappa 系数       : {test_kappa:.4f}")
        print("-" * 30)

        print("\n详细分类报告 (测试集):")
        print(classification_report(y_test, y_test_pred, target_names=class_names, zero_division=0))

        # 6. 特征重要性
        importance_df = calculate_permutation_importance(bls, X, y_labels, feature_names)

        print("\n" + "=" * 30)
        print("关键特征分析 (Top 10)")
        print("=" * 30)
        print(importance_df.head(10).to_string(index=False))