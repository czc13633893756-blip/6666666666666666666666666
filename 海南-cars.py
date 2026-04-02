import pandas as pd
import numpy as np

# ==========================================
# 1. 设置 Matplotlib 后端 (防止 PyCharm 报错)
# ==========================================
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score, mean_squared_error
import os
import warnings

warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# 2. CARS 特征筛选算法
# ==========================================
class CARS_Feature_Selector:
    def __init__(self, n_mc=50, n_splits=5):
        self.n_mc = n_mc
        self.n_splits = n_splits

    def fit(self, X, y, n_features_to_select=30):
        print(f"正在运行 CARS 特征筛选 (迭代 {self.n_mc} 次)...")
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape

        Vsel = np.arange(n_features)
        RMSECV = []
        Sel_Variables = []

        # 指数衰减参数
        a = (n_features / 2) ** (1 / (self.n_mc - 1))
        k = (np.log(n_features / 2)) / (self.n_mc - 1)

        model = PLSRegression(n_components=5)

        for i in range(self.n_mc):
            # 1. 蒙特卡洛采样
            sample_idx = np.random.choice(n_samples, int(n_samples * 0.8), replace=False)
            X_cal = X[sample_idx][:, Vsel]
            y_cal = y[sample_idx]

            if X_cal.shape[1] == 0:
                break

            # 2. 计算权重
            try:
                model.fit(X_cal, y_cal)
                coef = np.abs(model.coef_).flatten()
                weights = coef / np.sum(coef)
            except:
                break

            # 3. 变量保留率
            ratio = (0.5 ** (1 / (self.n_mc - 1))) ** i
            n_keep = int(n_features * ratio)
            if n_keep < n_features_to_select:
                n_keep = n_features_to_select

                # 4. ARS 采样
            sort_idx = np.argsort(weights)[::-1]
            keep_idx = sort_idx[:n_keep]
            Vsel = Vsel[keep_idx]
            Sel_Variables.append(Vsel)

            # 5. 交叉验证
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
            rmse_scores = []

            n_comp = min(5, len(Vsel))
            if n_comp < 1: n_comp = 1
            cv_model = PLSRegression(n_components=n_comp)

            X_curr = X[:, Vsel]
            for train_index, test_index in kf.split(X_curr):
                cv_model.fit(X_curr[train_index], y[train_index])
                y_pred = cv_model.predict(X_curr[test_index])
                # 兼容旧版 sklearn
                mse = mean_squared_error(y[test_index], y_pred)
                rmse = np.sqrt(mse)
                rmse_scores.append(rmse)

            RMSECV.append(np.mean(rmse_scores))

            if (i + 1) % 10 == 0:
                print(f"CARS 迭代: {i + 1}/{self.n_mc}, 保留特征: {len(Vsel)}, RMSECV: {RMSECV[-1]:.4f}")

        # 6. 结果提取
        if len(RMSECV) == 0:
            return np.arange(n_features_to_select)

        best_iter = np.argmin(RMSECV)
        best_vars = Sel_Variables[best_iter]
        print(f"CARS 最佳迭代: {best_iter + 1}, RMSECV={min(RMSECV):.4f}")

        # 强制 Top N
        final_pls = PLSRegression(n_components=min(5, len(best_vars)))
        final_pls.fit(X[:, best_vars], y)
        final_coef = np.abs(final_pls.coef_).flatten()

        top_indices_local = np.argsort(final_coef)[::-1][:n_features_to_select]
        final_selected_indices = best_vars[top_indices_local]

        return final_selected_indices


# ==========================================
# 3. 宽度学习系统 (BLS) - 【已修复初始化】
# ==========================================
class BroadLearningSystem:
    def __init__(self, n_feature_nodes=10, n_feature_mapped=10, n_enhancement_nodes=50, s=0.8, c=2 ** -30):
        self.n_feature_nodes = n_feature_nodes
        self.n_feature_mapped = n_feature_mapped
        self.n_enhancement_nodes = n_enhancement_nodes
        self.s = s
        self.c = c
        self.scaler_input = MinMaxScaler()
        self.onehot = OneHotEncoder(sparse_output=False)

        # 【修复点】初始化权重列表
        self.weights_feature = []
        self.weights_enhance = None
        self.bias_enhance = None
        self.output_weights = None

    def _tansig(self, x):
        return np.tanh(x)

    def _sparse_autoencoder_weights(self, X):
        in_dim = X.shape[1]
        out_dim = self.n_feature_mapped
        W = 2 * np.random.random((in_dim, out_dim)) - 1
        b = 2 * np.random.random((1, out_dim)) - 1
        return W, b

    def fit(self, X, y):
        # 重置权重列表，防止重复调用 fit 时累积
        self.weights_feature = []

        X = np.array(X)
        X = self.scaler_input.fit_transform(X)
        if len(y.shape) == 1: y = y.reshape(-1, 1)
        y = self.onehot.fit_transform(y)

        self.Z = []
        for i in range(self.n_feature_nodes):
            W, b = self._sparse_autoencoder_weights(X)
            self.weights_feature.append((W, b))
            Zi = self._tansig(np.dot(X, W) + b)
            self.Z.append(Zi)
        self.Z_concat = np.hstack(self.Z)

        in_dim_enhance = self.Z_concat.shape[1]
        self.weights_enhance = 2 * np.random.random((in_dim_enhance, self.n_enhancement_nodes)) - 1
        self.bias_enhance = 2 * np.random.random((1, self.n_enhancement_nodes)) - 1
        H = self._tansig(np.dot(self.Z_concat, self.weights_enhance) + self.bias_enhance)

        A = np.hstack([self.Z_concat, H])

        try:
            lambda_I = np.eye(A.shape[1]) * self.c
            pseudo_inverse = np.linalg.inv(np.dot(A.T, A) + lambda_I).dot(A.T)
        except:
            pseudo_inverse = np.linalg.pinv(A)
        self.output_weights = np.dot(pseudo_inverse, y)

    def predict(self, X):
        X = np.array(X)
        X = self.scaler_input.transform(X)
        Z_list = []
        for i in range(self.n_feature_nodes):
            W, b = self.weights_feature[i]
            Zi = self._tansig(np.dot(X, W) + b)
            Z_list.append(Zi)
        Z_concat = np.hstack(Z_list)
        H = self._tansig(np.dot(Z_concat, self.weights_enhance) + self.bias_enhance)
        A = np.hstack([Z_concat, H])
        return np.argmax(np.dot(A, self.output_weights), axis=1)


# ==========================================
# 4. 工具函数
# ==========================================
def load_data(spectra_path, physio_path):
    if not os.path.exists(spectra_path) or not os.path.exists(physio_path):
        print("错误：文件未找到。")
        return None, None, None

    df_spectra = pd.read_excel(spectra_path, header=0)
    X = df_spectra.iloc[0:60, 1:]
    feature_names = X.columns.tolist()

    df_physio = pd.read_excel(physio_path, header=0)
    y_continuous = df_physio.iloc[0:60, 31]

    return X, y_continuous, feature_names


def generate_labels(target_series):
    target_series = pd.to_numeric(target_series, errors='coerce')
    p20 = np.percentile(target_series, 20)
    p80 = np.percentile(target_series, 80)
    labels = []
    for val in target_series:
        if val <= p20:
            labels.append(0)  # HR
        elif val >= p80:
            labels.append(2)  # HS
        else:
            labels.append(1)  # MR
    return np.array(labels), ["HR", "MR", "HS"]


def plot_selected_features(X_all, feature_indices, feature_names):
    X_mean = np.mean(X_all, axis=0)
    x_axis = np.arange(len(X_mean))

    plt.figure(figsize=(12, 6))
    plt.plot(x_axis, X_mean, 'b-', alpha=0.6, label='平均光谱曲线')

    selected_x = feature_indices
    selected_y = X_mean.iloc[feature_indices] if hasattr(X_mean, 'iloc') else X_mean[feature_indices]

    plt.scatter(selected_x, selected_y, c='red', s=50, marker='o', label='CARS筛选波段', zorder=5)

    ymin, ymax = plt.ylim()
    plt.vlines(selected_x, ymin, selected_y, colors='red', linestyles='dotted', alpha=0.3)

    plt.title('CARS筛选的30个特征波段在平均光谱上的分布', fontsize=14)
    plt.xlabel('波段索引', fontsize=12)
    plt.ylabel('吸光度/反射率', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('CARS_Feature_Distribution.png', dpi=300, bbox_inches='tight')
    print("图片已保存为: CARS_Feature_Distribution.png")


def save_features_to_excel(feature_indices, feature_names):
    selected_names = [feature_names[i] for i in feature_indices]
    df_out = pd.DataFrame({
        'Index': feature_indices,
        'Band_Name': selected_names,
        'Rank': range(1, len(feature_indices) + 1)
    })
    df_out.to_excel("CARS_Selected_30_Features.xlsx", index=False)
    print("筛选结果已保存为: CARS_Selected_30_Features.xlsx")


# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    spectra_file = r"C:\Users\Mayn\Desktop\海南项目\60条合成光谱曲线.xlsx"
    physio_file = r"C:\Users\Mayn\Desktop\海南项目\物理-生理指标.xlsx"

    # 1. 加载数据
    print("Step 1: 加载数据...")
    X_df, y_continuous, feature_names = load_data(spectra_file, physio_file)

    if X_df is not None:
        # 2. 生成标签
        y_labels, class_names = generate_labels(y_continuous)

        # 3. 运行 CARS
        print("\nStep 2: 运行 CARS 算法筛选特征...")
        cars = CARS_Feature_Selector(n_mc=50, n_splits=5)
        selected_indices = cars.fit(X_df, y_continuous, n_features_to_select=30)

        print(f"\n筛选完成！共筛选出 {len(selected_indices)} 个特征波段。")

        # 4. 保存结果和绘图
        print("\nStep 3: 导出结果与绘图...")
        save_features_to_excel(selected_indices, feature_names)
        plot_selected_features(X_df, selected_indices, feature_names)

        # 5. 建立 BLS 模型
        print("\nStep 4: 使用筛选特征建立 BLS 预测模型...")
        X_selected = X_df.iloc[:, selected_indices]
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y_labels, test_size=0.2, random_state=42,
                                                            stratify=y_labels)

        bls = BroadLearningSystem(n_feature_nodes=10, n_feature_mapped=10, n_enhancement_nodes=50, s=0.8, c=2 ** -30)
        bls.fit(X_train, y_train)

        y_pred = bls.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)

        print("\n" + "=" * 40)
        print(f"基于 CARS 筛选特征 ({len(selected_indices)}个) 的 BLS 模型性能")
        print("=" * 40)
        print(f"测试集精度 (Accuracy): {acc:.4f}")
        print(f"Kappa 系数           : {kappa:.4f}")
        print("-" * 40)
        print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))