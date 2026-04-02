import os
import sys
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 1. 修复Matplotlib后端问题
# ==========================================
# 设置环境变量，强制使用Agg后端
os.environ['MPLBACKEND'] = 'Agg'

# 在导入matplotlib之前设置
import matplotlib

# 尝试不同的后端设置
try:
    matplotlib.use('Agg', force=True)
    print("成功设置Matplotlib后端为Agg")
except Exception as e:
    print(f"设置Agg后端失败: {e}")
    # 尝试其他后端
    try:
        matplotlib.use('PDF', force=True)
        print("回退到PDF后端")
    except:
        print("无法设置后端，使用默认设置")

# 现在导入matplotlib的其他模块
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.sparse.linalg import svds
from sklearn.model_selection import train_test_split

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# 2. 光谱预处理方法
# ==========================================
class SpectraPreprocessor:
    """光谱预处理方法集合"""

    @staticmethod
    def d1st(spectra):
        """一阶导数 (D1st)"""
        return np.gradient(spectra, axis=1)

    @staticmethod
    def msc(spectra):
        """多元散射校正 (MSC)"""
        # 计算平均光谱
        mean_spectra = np.mean(spectra, axis=0)
        # 对每个样本进行回归
        n_samples, n_features = spectra.shape
        corrected = np.zeros_like(spectra)

        for i in range(n_samples):
            # 线性回归: spectra[i] = a + b * mean_spectra
            b, a = np.polyfit(mean_spectra, spectra[i], 1)
            # 校正
            corrected[i] = (spectra[i] - a) / b

        return corrected

    @staticmethod
    def snv(spectra):
        """标准正态变量变换 (SNV)"""
        n_samples = spectra.shape[0]
        corrected = np.zeros_like(spectra)

        for i in range(n_samples):
            spectrum = spectra[i]
            mean_val = np.mean(spectrum)
            std_val = np.std(spectrum)
            if std_val != 0:
                corrected[i] = (spectrum - mean_val) / std_val

        return corrected

    @staticmethod
    def sg(spectra, window_length=5, polyorder=2):
        """Savitzky-Golay平滑 (SG)"""
        return savgol_filter(spectra, window_length=window_length,
                             polyorder=polyorder, axis=1, mode='mirror')

    @staticmethod
    def msc_snv(spectra):
        """MSC + SNV"""
        spectra_msc = SpectraPreprocessor.msc(spectra)
        return SpectraPreprocessor.snv(spectra_msc)

    @staticmethod
    def snv_d1st(spectra):
        """SNV + 一阶导数"""
        spectra_snv = SpectraPreprocessor.snv(spectra)
        return SpectraPreprocessor.d1st(spectra_snv)

    @staticmethod
    def sg_d1st(spectra, window_length=5, polyorder=2):
        """SG平滑 + 一阶导数"""
        spectra_sg = SpectraPreprocessor.sg(spectra, window_length, polyorder)
        return SpectraPreprocessor.d1st(spectra_sg)

    @staticmethod
    def sg_snv_d1st(spectra, window_length=5, polyorder=2):
        """SG平滑 + SNV + 一阶导数"""
        spectra_sg = SpectraPreprocessor.sg(spectra, window_length, polyorder)
        spectra_snv = SpectraPreprocessor.snv(spectra_sg)
        return SpectraPreprocessor.d1st(spectra_snv)

    @staticmethod
    def snv_sg(spectra, window_length=5, polyorder=2):
        """SNV + SG平滑"""
        spectra_snv = SpectraPreprocessor.snv(spectra)
        return SpectraPreprocessor.sg(spectra_snv, window_length, polyorder)


# ==========================================
# 3. 简化的CARS特征筛选算法
# ==========================================
class SimpleFeatureSelector:
    def __init__(self, n_features_to_select=30):
        self.n_features_to_select = n_features_to_select
        self.selected_indices_ = None

    def fit(self, X, y):
        """简化版特征选择，基于方差和与y的相关性"""
        X = np.array(X)
        y = np.array(y).reshape(-1, 1) if len(y.shape) == 1 else y

        n_samples, n_features = X.shape

        # 计算每个特征的方差
        variances = np.var(X, axis=0)

        # 计算每个特征与目标的相关性
        correlations = np.zeros(n_features)
        for i in range(n_features):
            if np.std(X[:, i]) > 0:
                correlations[i] = np.abs(np.corrcoef(X[:, i], y.flatten())[0, 1])
            else:
                correlations[i] = 0

        # 组合得分：方差 + 相关性
        scores = variances + correlations

        # 选择得分最高的特征
        selected_idx = np.argsort(scores)[::-1][:self.n_features_to_select]

        self.selected_indices_ = selected_idx
        print(f"选择特征数: {len(selected_idx)}")

        return self

    def transform(self, X):
        """使用选择的特征转换数据"""
        if self.selected_indices_ is None:
            raise ValueError("必须先调用fit方法")
        return X[:, self.selected_indices_]


# ==========================================
# 4. 宽度学习系统 (BLS) - 简化版
# ==========================================
class SimpleBroadLearningSystem:
    def __init__(self, n_feature_nodes=10, n_enhancement_nodes=50, c=1e-6):
        self.n_feature_nodes = n_feature_nodes
        self.n_enhancement_nodes = n_enhancement_nodes
        self.c = c

        # 权重和偏置
        self.weights_feature = []
        self.bias_feature = []
        self.weights_enhance = None
        self.bias_enhance = None
        self.output_weights = None

    def _relu(self, x):
        """ReLU激活函数"""
        return np.maximum(0, x)

    def fit(self, X, y):
        """训练BLS模型"""
        X = np.array(X, dtype=np.float32)

        # 处理y为one-hot编码
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        n_classes = len(np.unique(y))

        if n_classes > 2:
            # 多分类
            y_onehot = np.zeros((len(y), n_classes), dtype=np.float32)
            for i, label in enumerate(y):
                y_onehot[i, int(label)] = 1
        else:
            # 二分类
            y_onehot = y.reshape(-1, 1).astype(np.float32)

        n_samples, n_features = X.shape

        # 生成特征节点
        self.weights_feature = []
        self.bias_feature = []
        Z_list = []

        for _ in range(self.n_feature_nodes):
            # 随机初始化权重
            W = np.random.randn(n_features, 10) * 0.1
            b = np.random.randn(1, 10) * 0.1
            self.weights_feature.append(W)
            self.bias_feature.append(b)

            # 计算特征节点
            Z = self._relu(np.dot(X, W) + b)
            Z_list.append(Z)

        Z = np.hstack(Z_list)

        # 生成增强节点
        n_feature_nodes_total = Z.shape[1]
        self.weights_enhance = np.random.randn(n_feature_nodes_total, self.n_enhancement_nodes) * 0.1
        self.bias_enhance = np.random.randn(1, self.n_enhancement_nodes) * 0.1
        H = self._relu(np.dot(Z, self.weights_enhance) + self.bias_enhance)

        # 组合特征
        A = np.hstack([Z, H]).astype(np.float32)

        # 计算输出权重
        try:
            # 使用岭回归
            lambda_I = np.eye(A.shape[1]) * self.c
            pseudo_inverse = np.linalg.inv(A.T @ A + lambda_I) @ A.T
            self.output_weights = pseudo_inverse @ y_onehot
        except np.linalg.LinAlgError:
            # 如果矩阵不可逆，使用伪逆
            self.output_weights = np.linalg.pinv(A) @ y_onehot

        return self

    def predict(self, X):
        """预测"""
        X = np.array(X, dtype=np.float32)

        # 计算特征节点
        Z_list = []
        for i in range(self.n_feature_nodes):
            Z = self._relu(np.dot(X, self.weights_feature[i]) + self.bias_feature[i])
            Z_list.append(Z)

        Z = np.hstack(Z_list)

        # 计算增强节点
        H = self._relu(np.dot(Z, self.weights_enhance) + self.bias_enhance)

        # 组合特征
        A = np.hstack([Z, H])

        # 预测
        y_pred = A @ self.output_weights

        if y_pred.shape[1] > 1:
            # 多分类
            return np.argmax(y_pred, axis=1)
        else:
            # 二分类
            return (y_pred > 0.5).astype(int).flatten()

    def predict_proba(self, X):
        """预测概率"""
        X = np.array(X, dtype=np.float32)

        # 计算特征节点
        Z_list = []
        for i in range(self.n_feature_nodes):
            Z = self._relu(np.dot(X, self.weights_feature[i]) + self.bias_feature[i])
            Z_list.append(Z)

        Z = np.hstack(Z_list)

        # 计算增强节点
        H = self._relu(np.dot(Z, self.weights_enhance) + self.bias_enhance)

        # 组合特征
        A = np.hstack([Z, H])

        # 预测
        y_pred = A @ self.output_weights

        if y_pred.shape[1] > 1:
            # 多分类，使用softmax
            exp_y = np.exp(y_pred - np.max(y_pred, axis=1, keepdims=True))
            return exp_y / np.sum(exp_y, axis=1, keepdims=True)
        else:
            # 二分类，使用sigmoid
            proba = 1 / (1 + np.exp(-y_pred))
            return np.hstack([1 - proba, proba])


# ==========================================
# 5. 评估工具函数
# ==========================================
def evaluate_model(y_true, y_pred, y_pred_proba=None):
    """评估模型性能"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.metrics import confusion_matrix, roc_auc_score

    results = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }

    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    results['confusion_matrix'] = cm

    # 如果有概率预测，计算AUC
    if y_pred_proba is not None and len(np.unique(y_true)) > 1:
        try:
            if y_pred_proba.shape[1] > 2:
                # 多分类AUC
                results['auc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
            else:
                # 二分类AUC
                results['auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
        except:
            results['auc'] = 0.0

    return results


# ==========================================
# 6. 主程序
# ==========================================
def main():
    """主程序：比较不同预处理方法的效果"""

    # 1. 生成模拟数据
    print("=" * 60)
    print("茶叶光谱数据分析 - 简化版")
    print("=" * 60)

    np.random.seed(42)
    n_samples = 100
    n_features = 200

    # 生成模拟光谱数据
    X = np.random.randn(n_samples, n_features)
    for i in range(n_samples):
        # 添加一些光谱特征
        X[i, 50:80] += np.sin(np.linspace(0, 2 * np.pi, 30)) * 2
        X[i, 120:150] += np.cos(np.linspace(0, 2 * np.pi, 30)) * 1.5

    # 生成连续目标变量
    y_continuous = 5 + X[:, 50] * 3 + X[:, 120] * 2 + np.random.randn(n_samples) * 0.5

    # 生成分类标签
    p20 = np.percentile(y_continuous, 20)
    p80 = np.percentile(y_continuous, 80)
    y_labels = []
    for val in y_continuous:
        if val <= p20:
            y_labels.append(0)  # HR
        elif val >= p80:
            y_labels.append(2)  # HS
        else:
            y_labels.append(1)  # MR
    y_labels = np.array(y_labels)

    print(f"数据形状: X={X.shape}, y={y_labels.shape}")
    print(f"类别分布: HR={np.sum(y_labels == 0)}, MR={np.sum(y_labels == 1)}, HS={np.sum(y_labels == 2)}")

    # 2. 定义预处理方法
    preprocess_methods = {
        '原始数据': lambda x: x,
        'D1st': SpectraPreprocessor.d1st,
        'MSC': SpectraPreprocessor.msc,
        'SNV': SpectraPreprocessor.snv,
        'SG': lambda x: SpectraPreprocessor.sg(x, window_length=5, polyorder=2),
        'SNV+D1st': SpectraPreprocessor.snv_d1st,
        'SG+D1st': lambda x: SpectraPreprocessor.sg_d1st(x, window_length=5, polyorder=2),
        'MSC+SNV': SpectraPreprocessor.msc_snv,
        'SG+SNV+D1st': lambda x: SpectraPreprocessor.sg_snv_d1st(x, window_length=5, polyorder=2)
    }

    # 3. 存储结果
    results = []

    # 4. 对每种预处理方法进行实验
    for method_name, preprocess_func in preprocess_methods.items():
        print(f"\n{'=' * 60}")
        print(f"预处理方法: {method_name}")
        print(f"{'=' * 60}")

        # 应用预处理
        try:
            X_processed = preprocess_func(X)
            print(f"预处理后数据形状: {X_processed.shape}")

            # 分割数据
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y_labels, test_size=0.2, random_state=42, stratify=y_labels
            )

            # 4.1 全波段BLS建模
            print(f"\n[全波段BLS建模]")
            print(f"-" * 40)

            bls_full = SimpleBroadLearningSystem(
                n_feature_nodes=10,
                n_enhancement_nodes=50,
                c=1e-6
            )

            bls_full.fit(X_train, y_train)
            y_pred_full = bls_full.predict(X_test)
            y_pred_proba_full = bls_full.predict_proba(X_test)

            metrics_full = evaluate_model(y_test, y_pred_full, y_pred_proba_full)

            print(f"准确率: {metrics_full['accuracy']:.4f}")
            print(f"精确率: {metrics_full['precision']:.4f}")
            print(f"召回率: {metrics_full['recall']:.4f}")
            print(f"F1分数: {metrics_full['f1']:.4f}")
            if 'auc' in metrics_full:
                print(f"AUC: {metrics_full['auc']:.4f}")

            # 4.2 特征选择后BLS建模
            print(f"\n[特征选择+BLS建模]")
            print(f"-" * 40)

            # 使用简化特征选择
            selector = SimpleFeatureSelector(n_features_to_select=30)

            # 特征选择使用训练集
            selector.fit(X_train, y_train)

            # 转换数据
            X_train_selected = selector.transform(X_train)
            X_test_selected = selector.transform(X_test)

            print(f"选择特征数: {X_train_selected.shape[1]}")

            # 使用BLS建模
            bls_selected = SimpleBroadLearningSystem(
                n_feature_nodes=10,
                n_enhancement_nodes=50,
                c=1e-6
            )

            bls_selected.fit(X_train_selected, y_train)
            y_pred_selected = bls_selected.predict(X_test_selected)
            y_pred_proba_selected = bls_selected.predict_proba(X_test_selected)

            metrics_selected = evaluate_model(y_test, y_pred_selected, y_pred_proba_selected)

            print(f"准确率: {metrics_selected['accuracy']:.4f}")
            print(f"精确率: {metrics_selected['precision']:.4f}")
            print(f"召回率: {metrics_selected['recall']:.4f}")
            print(f"F1分数: {metrics_selected['f1']:.4f}")
            if 'auc' in metrics_selected:
                print(f"AUC: {metrics_selected['auc']:.4f}")

            # 存储结果
            results.append({
                '预处理方法': method_name,
                '建模方式': '全波段',
                '特征数': X_train.shape[1],
                '准确率': metrics_full['accuracy'],
                '精确率': metrics_full['precision'],
                '召回率': metrics_full['recall'],
                'F1分数': metrics_full['f1'],
                'AUC': metrics_full.get('auc', 0)
            })

            results.append({
                '预处理方法': method_name,
                '建模方式': '特征选择',
                '特征数': X_train_selected.shape[1],
                '准确率': metrics_selected['accuracy'],
                '精确率': metrics_selected['precision'],
                '召回率': metrics_selected['recall'],
                'F1分数': metrics_selected['f1'],
                'AUC': metrics_selected.get('auc', 0)
            })

        except Exception as e:
            print(f"处理{method_name}时出错: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 5. 结果汇总
    print(f"\n{'=' * 60}")
    print("结果汇总")
    print(f"{'=' * 60}")

    if not results:
        print("没有获得任何结果，请检查错误信息。")
        return None

    # 创建结果DataFrame
    results_df = pd.DataFrame(results)
    print("\n详细结果:")
    print(results_df.to_string(index=False))

    # 保存结果到Excel
    try:
        results_df.to_excel('预处理方法比较结果.xlsx', index=False)
        print("\n结果已保存到: 预处理方法比较结果.xlsx")
    except Exception as e:
        print(f"保存Excel文件失败: {e}")

    # 6. 可视化比较
    try:
        # 提取每种预处理方法的结果
        methods = list(preprocess_methods.keys())
        full_band_acc = []
        selected_acc = []

        for method in methods:
            method_full = results_df[(results_df['预处理方法'] == method) & (results_df['建模方式'] == '全波段')]
            method_selected = results_df[(results_df['预处理方法'] == method) & (results_df['建模方式'] == '特征选择')]

            if not method_full.empty:
                full_band_acc.append(method_full['准确率'].values[0])
            else:
                full_band_acc.append(0)

            if not method_selected.empty:
                selected_acc.append(method_selected['准确率'].values[0])
            else:
                selected_acc.append(0)

        # 创建图表
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(methods))
        width = 0.35

        ax.bar(x - width / 2, full_band_acc, width, label='全波段', alpha=0.8, color='skyblue')
        ax.bar(x + width / 2, selected_acc, width, label='特征选择', alpha=0.8, color='lightcoral')
        ax.set_xlabel('预处理方法')
        ax.set_ylabel('准确率')
        ax.set_title('不同预处理方法的准确率比较')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # 在柱状图上添加数值
        for i, v in enumerate(full_band_acc):
            ax.text(i - width / 2, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
        for i, v in enumerate(selected_acc):
            ax.text(i + width / 2, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.savefig('预处理方法比较图.png', dpi=300, bbox_inches='tight')
        print("可视化结果已保存到: 预处理方法比较图.png")
        plt.close()

    except Exception as e:
        print(f"创建图表失败: {e}")

    # 7. 输出分析报告
    print(f"\n{'=' * 60}")
    print("分析报告")
    print(f"{'=' * 60}")

    if not results_df.empty:
        # 找出最佳方法
        best_full = results_df[results_df['建模方式'] == '全波段']
        best_selected = results_df[results_df['建模方式'] == '特征选择']

        if not best_full.empty:
            best_full_row = best_full.loc[best_full['准确率'].idxmax()]
            print(f"\n最佳全波段方法: {best_full_row['预处理方法']}")
            print(f"准确率: {best_full_row['准确率']:.4f}, F1分数: {best_full_row['F1分数']:.4f}")

        if not best_selected.empty:
            best_selected_row = best_selected.loc[best_selected['准确率'].idxmax()]
            print(f"\n最佳特征选择方法: {best_selected_row['预处理方法']}")
            print(f"准确率: {best_selected_row['准确率']:.4f}, F1分数: {best_selected_row['F1分数']:.4f}")
            print(f"特征数: {best_selected_row['特征数']}/{X.shape[1]}")

    return results_df


# ==========================================
# 7. 运行主程序
# ==========================================
if __name__ == "__main__":
    # 运行主程序
    print("开始运行光谱预处理与BLS建模分析...")
    results = main()

    print(f"\n{'=' * 60}")
    print("程序执行完成！")
    print(f"{'=' * 60}")

    if results is not None:
        print("\n生成的文件:")
        print("1. 预处理方法比较结果.xlsx - 详细的实验结果表格")
        print("2. 预处理方法比较图.png - 可视化比较图")
        print("\n分析总结:")
        print("1. 比较了9种不同的光谱预处理方法")
        print("2. 每种方法都进行了全波段和特征选择两种建模策略")
        print("3. 使用宽度学习系统(BLS)进行分类建模")
    else:
        print("程序执行过程中出现问题，请检查错误信息。")
