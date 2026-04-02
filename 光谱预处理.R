# 读取波段名称（第一行，从D列开始）
cat("  读取波段名称...\n")
band_names <- read.xlsx(file_path, rows = 1, cols = 4:1000, colNames = FALSE)
band_names <- as.character(unlist(band_names))

# 移除空值
band_names <- band_names[!is.na(band_names) & band_names != ""]

# 将波段名称转换为数值
wavelengths <- as.numeric(band_names)

# 如果转换失败，使用索引
if (any(is.na(wavelengths))) {
  cat("  警告: 部分波段名称无法转换为数值，使用索引作为波长\n")
  wavelengths <- 1:length(band_names)
}

cat("  波段数量:", length(wavelengths), "\n")
cat("  波长范围:", min(wavelengths), "-", max(wavelengths), "\n")

# 读取光谱数据
cat("  读取光谱数据...\n")
spectral_data <- read.xlsx(file_path, startRow = 2, cols = 4:(3 + length(wavelengths)))

# 转换为矩阵
spectra_matrix <- as.matrix(spectral_data)

# 转换为数值矩阵
spectra_numeric <- apply(spectra_matrix, 2, as.numeric)

cat("  样本数量:", nrow(spectra_numeric), "\n")
cat("  数据读取完成！\n\n")

# 定义预处理函数
cat("步骤2: 定义预处理函数...\n")

# 原始光谱
raw_preprocess <- function(data_matrix) {
  return(data_matrix)
}

# SNV预处理
snv_preprocess <- function(data_matrix) {
  cat("  应用SNV预处理...\n")
  result <- matrix(0, nrow = nrow(data_matrix), ncol = ncol(data_matrix))
  
  for (i in 1:nrow(data_matrix)) {
    spectrum <- data_matrix[i, ]
    mean_val <- mean(spectrum, na.rm = TRUE)
    sd_val <- sd(spectrum, na.rm = TRUE)
    
    if (is.na(sd_val) || sd_val == 0) {
      result[i, ] <- spectrum
    } else {
      result[i, ] <- (spectrum - mean_val) / sd_val
    }
  }
  
  return(result)
}

# SG平滑
sg_preprocess <- function(data_matrix) {
  cat("  应用SG平滑...\n")
  
  if (!require("signal", quietly = TRUE)) {
    cat("  警告: signal包未加载，跳过SG平滑\n")
    return(data_matrix)
  }
  
  result <- matrix(0, nrow = nrow(data_matrix), ncol = ncol(data_matrix))
  
  for (i in 1:nrow(data_matrix)) {
    spectrum <- data_matrix[i, ]
    
    tryCatch({
      window_size <- min(11, length(spectrum))
      if (window_size %% 2 == 0) window_size <- window_size - 1
      
      result[i, ] <- signal::sgolayfilt(spectrum, p = 2, n = window_size)
    }, error = function(e) {
      result[i, ] <- spectrum
    })
  }
  
  return(result)
}

# SNV + 一阶导数
snv_d1st_preprocess <- function(data_matrix) {
  cat("  应用SNV+一阶导数预处理...\n")
  
  # 先应用SNV
  snv_data <- snv_preprocess(data_matrix)
  
  # 计算一阶导数
  n_samples <- nrow(snv_data)
  n_bands <- ncol(snv_data)
  
  if (n_bands < 2) {
    cat("  警告: 波段数太少，无法计算一阶导数\n")
    return(matrix(0, nrow = n_samples, ncol = 1))
  }
  
  result <- matrix(0, nrow = n_samples, ncol = n_bands - 1)
  
  for (i in 1:n_samples) {
    spectrum <- snv_data[i, ]
    result[i, ] <- diff(spectrum)
  }
  
  return(result)
}

# 应用预处理
cat("步骤3: 应用预处理方法...\n")

preprocessed_data <- list()
preprocessed_data$raw <- raw_preprocess(spectra_numeric)
preprocessed_data$snv <- snv_preprocess(spectra_numeric)
preprocessed_data$sg <- sg_preprocess(spectra_numeric)
preprocessed_data$snv_d1st <- snv_d1st_preprocess(spectra_numeric)

# 调整波长向量
cat("步骤4: 调整波长向量...\n")

wavelength_list <- list()
wavelength_list$raw <- wavelengths
wavelength_list$snv <- wavelengths
wavelength_list$sg <- wavelengths

# 一阶导数的波长（使用中点）
if (length(wavelengths) >= 2) {
  wavelength_mid <- (wavelengths[-length(wavelengths)] + wavelengths[-1]) / 2
  wavelength_list$snv_d1st <- wavelength_mid
} else {
  wavelength_list$snv_d1st <- wavelengths
}

# 创建输出目录
output_dir <- "spectral_analysis_matlab_style"
if (!dir.exists(output_dir)) {
  dir.create(output_dir)
  cat("创建输出目录:", output_dir, "\n")
}

# 3. 绘制图形 - 只生成每种预处理的单独图片
cat("步骤5: 绘制预处理后的图形...\n")

# 定义颜色函数 - 随机颜色
generate_random_colors <- function(n) {
  # 使用彩虹色，但随机化顺序
  colors <- rainbow(n)
  return(sample(colors))
}

# 3.1 绘制原始光谱
cat("  3.1 绘制原始光谱...\n")
data_matrix <- preprocessed_data$raw
n_samples <- nrow(data_matrix)

# 创建数据框
plot_data <- data.frame()
for (i in 1:n_samples) {
  temp_df <- data.frame(
    Sample = paste("样本", i),
    Wavelength = wavelength_list$raw,
    Intensity = data_matrix[i, ]
  )
  plot_data <- rbind(plot_data, temp_df)
}

# 生成随机颜色
sample_colors <- generate_random_colors(n_samples)

# 绘制图形
p_raw <- ggplot(plot_data, aes(x = Wavelength, y = Intensity, group = Sample, color = Sample)) +
  geom_line(linewidth = 0.2) +  # 设置线宽为0.2
  theme_minimal() +
  labs(
    title = "原始光谱 - 所有样本",
    x = "波长 (nm)",
    y = "反射率"
  ) +
  theme(
    plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
    axis.title = element_text(size = 14),
    axis.text = element_text(size = 12),
    legend.position = "none",  # 不显示图例
    panel.grid.major = element_line(color = "gray90", size = 0.2),
    panel.grid.minor = element_blank(),
    panel.background = element_rect(fill = "white", color = NA)
  ) +
  scale_color_manual(values = sample_colors) +
  scale_x_continuous(expand = c(0, 0)) +
  scale_y_continuous(expand = c(0, 0))

# 保存图形
png_raw <- paste0(output_dir, "/raw_spectra_all.png")
png(png_raw, width = 3200, height = 2400, res = 300)
print(p_raw)
dev.off()

pdf_raw <- paste0(output_dir, "/raw_spectra_all.pdf")
pdf(pdf_raw, width = 12, height = 9)
print(p_raw)
dev.off()

cat("    原始光谱图已保存:", png_raw, "\n")

# 3.2 绘制SNV预处理光谱
cat("  3.2 绘制SNV预处理光谱...\n")
data_matrix <- preprocessed_data$snv
n_samples <- nrow(data_matrix)

# 创建数据框
plot_data <- data.frame()
for (i in 1:n_samples) {
  temp_df <- data.frame(
    Sample = paste("样本", i),
    Wavelength = wavelength_list$snv,
    Intensity = data_matrix[i, ]
  )
  plot_data <- rbind(plot_data, temp_df)
}

# 生成随机颜色
sample_colors <- generate_random_colors(n_samples)

# 绘制图形
p_snv <- ggplot(plot_data, aes(x = Wavelength, y = Intensity, group = Sample, color = Sample)) +
  geom_line(linewidth = 0.2) +  # 设置线宽为0.2
  theme_minimal() +
  labs(
    title = "SNV预处理 - 所有样本",
    x = "波长 (nm)",
    y = "反射率"
  ) +
  theme(
    plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
    axis.title = element_text(size = 14),
    axis.text = element_text(size = 12),
    legend.position = "none",  # 不显示图例
    panel.grid.major = element_line(color = "gray90", size = 0.2),
    panel.grid.minor = element_blank(),
    panel.background = element_rect(fill = "white", color = NA)
  ) +
  scale_color_manual(values = sample_colors) +
  scale_x_continuous(expand = c(0, 0)) +
  scale_y_continuous(expand = c(0, 0))

# 保存图形
png_snv <- paste0(output_dir, "/snv_spectra_all.png")
png(png_snv, width = 3200, height = 2400, res = 300)
print(p_snv)
dev.off()

pdf_snv <- paste0(output_dir, "/snv_spectra_all.pdf")
pdf(pdf_snv, width = 12, height = 9)
print(p_snv)
dev.off()

cat("    SNV预处理光谱图已保存:", png_snv, "\n")

# 3.3 绘制SG平滑光谱
cat("  3.3 绘制SG平滑光谱...\n")
data_matrix <- preprocessed_data$sg
n_samples <- nrow(data_matrix)

# 创建数据框
plot_data <- data.frame()
for (i in 1:n_samples) {
  temp_df <- data.frame(
    Sample = paste("样本", i),
    Wavelength = wavelength_list$sg,
    Intensity = data_matrix[i, ]
  )
  plot_data <- rbind(plot_data, temp_df)
}

# 生成随机颜色
sample_colors <- generate_random_colors(n_samples)

# 绘制图形
p_sg <- ggplot(plot_data, aes(x = Wavelength, y = Intensity, group = Sample, color = Sample)) +
  geom_line(linewidth = 0.2) +  # 设置线宽为0.2
  theme_minimal() +
  labs(
    title = "SG平滑 - 所有样本",
    x = "波长 (nm)",
    y = "反射率"
  ) +
  theme(
    plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
    axis.title = element_text(size = 14),
    axis.text = element_text(size = 12),
    legend.position = "none",  # 不显示图例
    panel.grid.major = element_line(color = "gray90", size = 0.2),
    panel.grid.minor = element_blank(),
    panel.background = element_rect(fill = "white", color = NA)
  ) +
  scale_color_manual(values = sample_colors) +
  scale_x_continuous(expand = c(0, 0)) +
  scale_y_continuous(expand = c(0, 0))

# 保存图形
png_sg <- paste0(output_dir, "/sg_spectra_all.png")
png(png_sg, width = 3200, height = 2400, res = 300)
print(p_sg)
dev.off()

pdf_sg <- paste0(output_dir, "/sg_spectra_all.pdf")
pdf(pdf_sg, width = 12, height = 9)
print(p_sg)
dev.off()

cat("    SG平滑光谱图已保存:", png_sg, "\n")

# 3.4 绘制SNV+一阶导数光谱
cat("  3.4 绘制SNV+一阶导数光谱...\n")
data_matrix <- preprocessed_data$snv_d1st
n_samples <- nrow(data_matrix)

# 创建数据框
plot_data <- data.frame()
for (i in 1:n_samples) {
  temp_df <- data.frame(
    Sample = paste("样本", i),
    Wavelength = wavelength_list$snv_d1st,
    Intensity = data_matrix[i, ]
  )
  plot_data <- rbind(plot_data, temp_df)
}

# 生成随机颜色
sample_colors <- generate_random_colors(n_samples)

# 绘制图形
p_snv_d1st <- ggplot(plot_data, aes(x = Wavelength, y = Intensity, group = Sample, color = Sample)) +
  geom_line(linewidth = 0.2) +  # 设置线宽为0.2
  theme_minimal() +
  labs(
    title = "SNV+一阶导数 - 所有样本",
    x = "波长 (nm)",
    y = "一阶导数值"
  ) +
  theme(
    plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
    axis.title = element_text(size = 14),
    axis.text = element_text(size = 12),
    legend.position = "none",  # 不显示图例
    panel.grid.major = element_line(color = "gray90", size = 0.2),
    panel.grid.minor = element_blank(),
    panel.background = element_rect(fill = "white", color = NA)
  ) +
  scale_color_manual(values = sample_colors) +
  scale_x_continuous(expand = c(0, 0)) +
  scale_y_continuous(expand = c(0, 0))

# 保存图形
png_snv_d1st <- paste0(output_dir, "/snv_d1st_spectra_all.png")
png(png_snv_d1st, width = 3200, height = 2400, res = 300)
print(p_snv_d1st)
dev.off()

pdf_snv_d1st <- paste0(output_dir, "/snv_d1st_spectra_all.pdf")
pdf(pdf_snv_d1st, width = 12, height = 9)
print(p_snv_d1st)
dev.off()

cat("    SNV+一阶导数光谱图已保存:", png_snv_d1st, "\n")

cat("\n================================================\n")
cat("高光谱数据分析完成！\n")
cat("所有图形已保存到目录:", output_dir, "\n")
cat("参数设置: 颜色随机, 线条宽度=0.2, 显示所有样本\n")
cat("================================================\n")

return(TRUE)