import matplotlib

# ================= 修复核心 =================
# 强制使用 TkAgg 后端，解决 PyCharm 报错 "module 'backend_interagg' has no attribute..."
# 这行代码必须在 import matplotlib.pyplot 之前运行
try:
    matplotlib.use('TkAgg')
except:
    pass  # 如果 TkAgg 不可用，尝试继续（通常 Windows 都支持 TkAgg）
# ===========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
import os

# =================配置区域=================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 文件路径
SPECTRA_FILE = r"C:\Users\Mayn\Desktop\海南项目\60条合成光谱曲线.xlsx"
PHYSIO_FILE = r"C:\Users\Mayn\Desktop\海南项目\物理-生理指标.xlsx"
OUTPUT_DIR = r"C:\Users\Mayn\Desktop\海南项目\分析结果"

if not os.path.exists(OUTPUT_DIR):
    try:
        os.makedirs(OUTPUT_DIR)
    except:
        pass


# =================数据读取=================

def load_data():
    print(f"正在读取光谱文件: {SPECTRA_FILE}")
    if not os.path.exists(SPECTRA_FILE):
        raise FileNotFoundError(f"找不到光谱文件: {SPECTRA_FILE}")

    # 读取光谱
    df_spec = pd.read_excel(SPECTRA_FILE, index_col=0)

    print(f"正在读取指标文件: {PHYSIO_FILE}")
    if not os.path.exists(PHYSIO_FILE):
        raise FileNotFoundError(f"找不到指标文件: {PHYSIO_FILE}")

    # 读取指标
    df_phys = pd.read_excel(PHYSIO_FILE, index_col=0)

    # 数据对齐
    common_index = df_spec.index.intersection(df_phys.index)

    if len(common_index) == 0:
        raise ValueError("错误：两个文件的 A 列（品种名称）没有重合，无法分析。")

    print(f"匹配成功：共有 {len(common_index)} 个共同样本。")

    df_spec = df_spec.loc[common_index]
    df_phys = df_phys.loc[common_index]

    return df_spec, df_phys


# =================核心分析=================

def run_analysis():
    try:
        X_df, Y_df = load_data()
    except Exception as e:
        print("错误:", e)
        return

    wavelengths = X_df.columns
    indicators = Y_df.columns

    print(f"开始计算... (波段数: {len(wavelengths)}, 指标数: {len(indicators)})")

    correlation_results = pd.DataFrame(index=wavelengths, columns=indicators)
    plsr_loadings_results = pd.DataFrame(index=wavelengths, columns=indicators)

    # 预处理
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_df)
    scaler_Y = StandardScaler()
    Y_scaled = scaler_Y.fit_transform(Y_df)

    # 循环计算
    for i, ind_name in enumerate(indicators):
        print(f"正在分析: {ind_name} ...")

        # 1. 相关性
        y_col = Y_df[ind_name]
        corrs = []
        for w in wavelengths:
            # 简单处理：如果有缺失值(NaN)，填0或者跳过，这里假设数据完整
            r = np.corrcoef(X_df[w], y_col)[0, 1]
            corrs.append(r)
        correlation_results[ind_name] = corrs

        # 2. PLSR 载荷
        pls = PLSRegression(n_components=2)
        pls.fit(X_scaled, Y_scaled[:, i])
        plsr_loadings_results[ind_name] = pls.x_loadings_[:, 0]

    # =================绘图=================
    print("计算完成，正在生成图谱（会弹出新窗口）...")

    # 尝试将波段转为数字
    try:
        plot_wavelengths = [float(x) for x in wavelengths]
        x_label_text = "Wavelength (nm)"
    except:
        plot_wavelengths = range(len(wavelengths))
        x_label_text = "Band Index"

    colors = sns.color_palette("husl", len(indicators))

    # --- 图1 ---
    plt.figure(figsize=(12, 6))
    for i, ind_name in enumerate(indicators):
        plt.plot(plot_wavelengths, correlation_results[ind_name],
                 label=ind_name, color=colors[i], linewidth=1.5, alpha=0.8)
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.title('相关系数图谱', fontsize=14)
    plt.xlabel(x_label_text)
    plt.ylabel('Correlation (r)')
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '1_相关系数图谱.png'), dpi=300)
    plt.show()

    # --- 图2 ---
    plt.figure(figsize=(14, 8))
    sns.heatmap(correlation_results.T, cmap="coolwarm", center=0)
    plt.title('相关系数热力图', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '2_相关系数热力图.png'), dpi=300)
    plt.show()

    # --- 图3 ---
    plt.figure(figsize=(12, 6))
    for i, ind_name in enumerate(indicators):
        plt.plot(plot_wavelengths, plsr_loadings_results[ind_name],
                 label=ind_name, color=colors[i], linewidth=1.5, alpha=0.8)
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.title('PLSR 载荷图谱', fontsize=14)
    plt.xlabel(x_label_text)
    plt.ylabel('Loadings')
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '3_PLSR载荷图谱.png'), dpi=300)
    plt.show()

    print(f"完成！图片保存在: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_analysis()