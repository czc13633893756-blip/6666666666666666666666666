import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, cohen_kappa_score
import warnings
import os
import random

# 忽略一些警告
warnings.filterwarnings("ignore")


# ==========================================
# 1. 宽度学习系统 (BLS) 实现类
# ==========================================
class BLS_Classifier:
    def __init__(self, Nf=10, Nm=10, Ne=100, s=0.8, C=2 ** -30):
        """
        参数:
        Nf: 每个窗口的特征节点数
        Nm: 窗口(特征组)数量
        Ne: 增强节点数
        s:  收缩系数
        C:  正则化系数
        """
        self.Nf = Nf
        self.Nm = Nm
        self.Ne = Ne
        self.s = s
        self.C = C
        self.W = None
        self.onehot = OneHotEncoder(sparse_output=False)
        self.scaler_input = MinMaxScaler()
        self.scalers_Zi = []
        self.scalers_Zi_test = []

    def _tansig(self, x):
        return np.tanh(x)

    def _pinv(self, A, reg):
        return np.dot(np.linalg.inv(np.dot(A.T, A) + reg * np.eye(A.shape[1])), A.T)

    def _sparse_bls(self, A, h):
        W_random = np.random.randn(A.shape[1], h)
        W_random, _ = np.linalg.qr(W_random)
        return W_random

    def fit(self, X, y):
        X = self.scaler_input.fit_transform(X)
        self.y_encoded = self.onehot.fit_transform(y.reshape(-1, 1))
        X = np.hstack([X, 0.1 * np.ones((X.shape[0], 1))])

        self.Wh = []
        self.Be = []
        self.We = []
        self.scalers_Zi = []
        H = []

        for i in range(self.Nm):
            Wh_i = self._sparse_bls(X, self.Nf)
            self.Wh.append(Wh_i)
            Zi = np.dot(X, Wh_i)
            scaler_Zi = MinMaxScaler()
            Zi = scaler_Zi.fit_transform(Zi)
            self.scalers_Zi.append(scaler_Zi)
            H.append(Zi)

        H = np.hstack(H)
        H_bias = np.hstack([H, 0.1 * np.ones((H.shape[0], 1))])

        We = np.random.randn(H_bias.shape[1], self.Ne)
        We, _ = np.linalg.qr(We)
        self.We = We

        T = np.dot(H_bias, self.We)
        E = self._tansig(T)
        A = np.hstack([H, E])

        reg_matrix = self.C * np.eye(A.shape[1])
        if A.shape[0] < A.shape[1]:
            reg_matrix = reg_matrix * A.shape[1] / A.shape[0]

        self.W = np.dot(self._pinv(A, self.C), self.y_encoded)
        return self

    def predict(self, X):
        X = self.scaler_input.transform(X)
        X = np.hstack([X, 0.1 * np.ones((X.shape[0], 1))])
        H = []
        for i in range(self.Nm):
            Zi = np.dot(X, self.Wh[i])
            Zi = self.scalers_Zi[i].transform(Zi)
            H.append(Zi)

        H = np.hstack(H)
        H_bias = np.hstack([H, 0.1 * np.ones((H.shape[0], 1))])
        T = np.dot(H_bias, self.We)
        E = self._tansig(T)
        A = np.hstack([H, E])

        y_pred_onehot = np.dot(A, self.W)
        return self.onehot.inverse_transform(y_pred_onehot).flatten()


# ==========================================
# 2. 光谱预处理算法库 (包含9种方法)
# ==========================================
class Preprocessor:
    @staticmethod
    def raw(X):
        return X

    @staticmethod
    def sg(X, window_length=15, polyorder=3):
        if window_length > X.shape[1]: window_length = X.shape[1] // 2 * 2 - 1
        if window_length < 3: window_length = 3
        return savgol_filter(X, window_length, polyorder, axis=1)

    @staticmethod
    def d1st(X):
        return np.gradient(X, axis=1)

    @staticmethod
    def msc(X):
        mean_spectrum = np.mean(X, axis=0)
        n_samples, n_features = X.shape
        msc_X = np.zeros_like(X)
        for i in range(n_samples):
            y = X[i, :]
            A = np.vstack([mean_spectrum, np.ones_like(mean_spectrum)]).T
            coeff, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            msc_X[i, :] = (y - coeff[1]) / coeff[0]
        return msc_X

    @staticmethod
    def snv(X):
        mean = np.mean(X, axis=1, keepdims=True)
        std = np.std(X, axis=1, keepdims=True)
        return (X - mean) / (std + 1e-8)

    @staticmethod
    def snv_d1st(X):
        return Preprocessor.d1st(Preprocessor.snv(X))

    @staticmethod
    def sg_snv_d1st(X):
        return Preprocessor.d1st(Preprocessor.snv(Preprocessor.sg(X)))

    @staticmethod
    def sg_d1st(X):
        return Preprocessor.d1st(Preprocessor.sg(X))

    @staticmethod
    def msc_snv(X):
        return Preprocessor.snv(Preprocessor.msc(X))


# ==========================================
# 3. 按品种划分数据集的核心函数
# ==========================================
def split_data_by_variety_groups(X, y, groups, train_ratio=0.8, random_state=42):
    """
    根据品种(groups)进行数据集划分。
    确保同一个品种的所有样本要么都在训练集，要么都在测试集。
    """
    # 获取所有唯一的品种ID
    unique_varieties = np.unique(groups)
    n_varieties = len(unique_varieties)

    # 按照比例计算训练集品种数量 (例如 60 * 0.8 = 48)
    n_train_varieties = int(n_varieties * train_ratio)

    # 设置随机种子并打乱品种顺序
    rng = np.random.RandomState(random_state)
    rng.shuffle(unique_varieties)

    # 选出训练品种和测试品种
    train_varieties = unique_varieties[:n_train_varieties]
    test_varieties = unique_varieties[n_train_varieties:]

    # 创建掩码 (Mask) 来选择样本
    train_mask = np.isin(groups, train_varieties)
    test_mask = np.isin(groups, test_varieties)

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(f"  [数据集划分详情] 总品种数: {n_varieties}")
    print(f"  > 训练集: {len(train_varieties)} 个品种 (样本数: {len(y_train)})")
    print(f"  > 测试集: {len(test_varieties)} 个品种 (样本数: {len(y_test)})")

    # 验证一下训练集和测试集是否有重叠品种
    overlap = np.intersect1d(groups[train_mask], groups[test_mask])
    if len(overlap) > 0:
        raise ValueError(f"严重错误: 训练集和测试集存在重叠品种! {overlap}")

    return X_train, X_test, y_train, y_test


# ==========================================
# 4. 主程序逻辑
# ==========================================
def main():
    # --- 配置区域 ---
    file_path = r"C:\Users\Mayn\Desktop\海南项目\1200条光谱曲线汇总.xlsx"

    # 如果文件不存在，使用模拟数据测试
    use_mock_data = False
    if not os.path.exists(file_path):
        print(f"警告: 文件 {file_path} 不存在，将使用模拟数据进行测试。")
        use_mock_data = True

    # --- 要求的9种预处理方法 ---
    methods = {
        "RAW": Preprocessor.raw,
        "SG": Preprocessor.sg,
        "D1ST": Preprocessor.d1st,
        "MSC": Preprocessor.msc,
        "SNV": Preprocessor.snv,
        "SNV+D1ST": Preprocessor.snv_d1st,
        "SG+SNV+D1ST": Preprocessor.sg_snv_d1st,
        "SG+D1ST": Preprocessor.sg_d1st,
        "MSC+SNV": Preprocessor.msc_snv
    }

    # BLS 网格搜索参数
    grid_params = {
        'Nf': [10, 20],
        'Nm': [10, 20],
        'Ne': [100, 300]
    }

    try:
        if use_mock_data:
            print("正在生成模拟数据 (包含品种信息)...")
            n_samples = 1200
            n_features = 100
            n_classes = 5

            # 生成模拟光谱和标签
            np.random.seed(42)
            X_raw = np.random.randn(n_samples, n_features) * 0.5 + 10
            y = np.zeros(n_samples, dtype=int)

            # 生成模拟品种列 (A列)
            groups = []
            for v in range(1, 61):  # 60个品种
                variety_id = f"F{v}"
                for s in range(1, 21):  # 20个样本
                    groups.append(variety_id)

                # 生成模拟原始等级
                class_label = (v - 1) // 12
                start_idx = (v - 1) * 20
                end_idx = v * 20
                y[start_idx:end_idx] = class_label

            groups = np.array(groups)
            print(f"模拟数据生成完毕: 60个品种，每品种20条数据。")

        else:
            print(f"正在读取文件: {file_path} ...")
            df = pd.read_excel(file_path)

            # 提取品种 ID
            raw_ids = df.iloc[:, 0].astype(str).values
            groups = np.array([x.split('_')[0] for x in raw_ids])

            y = df.iloc[:, 1].values
            X_raw = df.iloc[:, 2:].values

            if np.isnan(X_raw).any():
                X_raw = np.nan_to_num(X_raw)

            # --- 【核心修改: 等级重划分】 ---
            # 规则: 0-1->高抗, 2-4->中抗, 5->感
            print("正在进行抗性等级重分类 (3分类)...")
            y_mapped = np.zeros_like(y)

            # 0-1级 -> 0 (High Resistance)
            mask_high = (y <= 1)
            y_mapped[mask_high] = 0

            # 2-4级 -> 1 (Moderate Resistance)
            mask_mid = (y >= 2) & (y <= 4)
            y_mapped[mask_mid] = 1

            # 5级 -> 2 (Susceptible)
            mask_low = (y >= 5)
            y_mapped[mask_low] = 2

            # 替换原始标签
            y = y_mapped

            # 标签映射
            unique_labels = np.unique(y)
            print(f"当前类别分布: {dict(zip(*np.unique(y, return_counts=True)))}")
            print("类别映射: 0=高抗(0-1级), 1=中抗(2-4级), 2=感(5级)")

            if not np.array_equal(unique_labels, np.arange(len(unique_labels))):
                label_map = {label: i for i, label in enumerate(unique_labels)}
                y = np.array([label_map[label] for label in y])

            print(f"数据加载成功! 样本数: {X_raw.shape[0]}")
            print(f"检测到的唯一品种数: {len(np.unique(groups))} (预期为60)")

    except Exception as e:
        print(f"数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 存储最佳结果
    best_overall_acc = 0
    best_overall_config = {}
    results_list = []

    # 循环遍历预处理方法
    for method_name, method_func in methods.items():
        print(f"\nProcessing: [{method_name}] ...")

        # 1. 预处理
        try:
            X_processed = method_func(X_raw)
        except Exception as e:
            print(f"预处理失败: {e}")
            continue

        # 2. 数据集划分 (按品种划分!)
        X_train, X_test, y_train, y_test = split_data_by_variety_groups(
            X_processed, y, groups, train_ratio=0.8, random_state=42
        )

        best_method_acc = 0
        best_method_params = {}

        # 3. 网格搜索 BLS
        for nf in grid_params['Nf']:
            for nm in grid_params['Nm']:
                for ne in grid_params['Ne']:
                    try:
                        model = BLS_Classifier(Nf=nf, Nm=nm, Ne=ne)
                        model.fit(X_train, y_train)

                        y_pred_test = model.predict(X_test)

                        acc_test = accuracy_score(y_test, y_pred_test)
                        kappa_test = cohen_kappa_score(y_test, y_pred_test)

                        # 记录训练集准确率仅供参考
                        y_pred_train = model.predict(X_train)
                        acc_train = accuracy_score(y_train, y_pred_train)

                        current_res = {
                            'Method': method_name,
                            'Nf': nf, 'Nm': nm, 'Ne': ne,
                            'Train_Acc': acc_train,
                            'Test_Acc': acc_test,
                            'Test_Kappa': kappa_test
                        }

                        if acc_test > best_method_acc:
                            best_method_acc = acc_test
                            best_method_params = current_res

                        if acc_test > best_overall_acc:
                            best_overall_acc = acc_test
                            best_overall_config = current_res

                    except Exception as e:
                        continue

        if best_method_acc > 0:
            print(f"  > [{method_name}] 最佳测试准确率: {best_method_params['Test_Acc']:.4f}")
            results_list.append(best_method_params)

    # ==========================================
    # 5. 输出结果
    # ==========================================
    if results_list:
        res_df = pd.DataFrame(results_list).sort_values(by='Test_Acc', ascending=False)
        print("\n" + "=" * 60)
        print("最终结果汇总 (按品种划分: 48训练 / 12测试)")
        print("类别定义: 高抗(0-1), 中抗(2-4), 感(5)")
        print("=" * 60)
        print(res_df.to_string(index=False))

        # 保存
        res_df.to_csv("bls_variety_split_results.csv", index=False, encoding='utf-8-sig')


if __name__ == "__main__":
    main()