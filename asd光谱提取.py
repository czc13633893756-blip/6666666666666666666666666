import pandas as pd
import os

def process_spectral_data():
    # ================= 配置区域 =================
    # 输入文件夹路径 (请确认路径中没有多余空格)
    input_folder_path = r"C:\Users\Mayn\Desktop\60+大叶+22品种光谱数据（）\2024.9.11CHAI02"
    
    # 输出文件保存路径 (默认保存在原文件夹下)
    output_excel_path = os.path.join(input_folder_path, "1200条光谱曲线汇总.xlsx")
    
    # 文件数量配置
    total_files = 60
    curves_per_file = 200
    merge_step = 10  # 每10条合并为1条
    # ===========================================

    print(f"开始处理，目标文件夹: {input_folder_path}")
    print(f"处理逻辑: 每{merge_step}条光谱合并为1条平均光谱")

    all_merged_data = [] # 用于存放所有处理后的数据框
    wavelength_index = None # 用于存放波段信息 (350-2500)

    # 遍历 1 到 60 个文件
    for i in range(1, total_files + 1):
        file_name = f"{i}.txt"
        file_path = os.path.join(input_folder_path, file_name)

        if not os.path.exists(file_path):
            print(f"[警告] 文件不存在，跳过: {file_name}")
            continue

        try:
            # 读取文本文件
            # 假设文件是以制表符(\t)或空格分隔的。如果报错，可以尝试 sep='\s+'
            # header=0 表示第一行是标题
            df = pd.read_csv(file_path, sep='\t', header=0) 
            
            # 如果读取后列数不对 (比如全部挤在一列)，尝试自动检测分隔符
            if df.shape[1] < 2:
                df = pd.read_csv(file_path, sep=None, engine='python', header=0)

            # 第一列通常是波段 (Wavelength)，后面是光谱数据
            # 提取波段列作为索引 (只在处理第一个文件时提取一次，作为基准)
            if wavelength_index is None:
                wavelength_index = df.iloc[:, 0].values
            
            # 提取仅包含光谱数据的部分 (去掉第一列波段)
            # 假设结构是: [Wavelength, Curve1, Curve2, ... Curve200]
            spectral_data = df.iloc[:, 1:]

            # 检查列数是否足够
            if spectral_data.shape[1] < curves_per_file:
                print(f"[错误] 文件 {file_name} 的数据列不足 {curves_per_file} 列，跳过。")
                continue

            # 开始合并处理：每10列求平均值
            merged_file_data = pd.DataFrame()
            
            # 循环生成 20 条新曲线 (200 / 10 = 20)
            num_new_curves = int(curves_per_file / merge_step)
            
            for group in range(num_new_curves):
                start_col = group * merge_step
                end_col = start_col + merge_step
                
                # 提取当前的10列数据
                subset = spectral_data.iloc[:, start_col:end_col]
                
                # 计算平均值 (axis=1 表示按行计算，即不同波段下10条曲线的平均)
                # 列名命名规则: File{文件号}_Group{组号}
                new_col_name = f"F{i}_G{group+1}"
                merged_file_data[new_col_name] = subset.mean(axis=1)

            # 将当前文件处理好的20列数据加入列表
            all_merged_data.append(merged_file_data)
            
            print(f"[成功] 文件 {file_name} 处理完毕，生成 {num_new_curves} 条合并曲线。")

        except Exception as e:
            print(f"[异常] 处理文件 {file_name} 时发生错误: {e}")

    # ================= 汇总与导出 =================
    if all_merged_data:
        print("正在汇总所有数据并写入 Excel，这可能需要一点时间...")
        
        # 横向合并所有数据框
        final_df = pd.concat(all_merged_data, axis=1)
        
        # 插入波段列到第一列
        final_df.insert(0, 'Wavelength', wavelength_index)
        
        # 写入 Excel
        final_df.to_excel(output_excel_path, index=False)
        print(f"处理完成！文件已保存至: {output_excel_path}")
        print(f"总计生成曲线数: {final_df.shape[1] - 1} (不含波段列)")
    else:
        print("未处理任何数据，请检查文件路径或格式。")

if __name__ == '__main__':
    process_spectral_data()