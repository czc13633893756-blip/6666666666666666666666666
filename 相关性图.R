# 完整版本 - 包含错误处理和图片保存
library("corrplot")
library(openxlsx)
library(psych)
library(RColorBrewer)

# 读取Excel文件
file_path <- "C:/Users/Mayn/Desktop/实验MAX/鲜叶-儿茶素对应/儿茶素数据/数据-儿茶素-毫克每克 - 副本 - 副本.xlsx"

# 检查文件是否存在
if (!file.exists(file_path)) {
  stop("文件不存在，请检查路径是否正确")
}

tryCatch({
  # 读取数据
  data <- read.xlsx(file_path, startRow = 2)
  
  # 提取D1-M1列的数据
  catechin_data <- data[, 4:12]
  
  # 设置列名为儿茶素名称
  col_names <- read.xlsx(file_path, rows = 1, cols = 4:12, colNames = FALSE)
  colnames(catechin_data) <- as.character(col_names[1, ])
  
  # 删除包含缺失值的行
  catechin_data <- na.omit(catechin_data)
  
  # 检查数据是否有效
  if (nrow(catechin_data) < 2) {
    stop("有效数据不足，无法计算相关性")
  }
  
  # 计算相关性矩阵
  cor_result <- corr.test(catechin_data, method = 'pearson', adjust = 'none')
  
  # 设置颜色
  addcol <- colorRampPalette(rev(brewer.pal(9,"RdBu")))
  
  # 定义绘制相关性图的函数
  plot_correlation <- function() {
    # 设置图形边距，底部留出更多空间用于显示显著性标记说明
    par(mar = c(6, 4, 4, 2) + 0.1)
    
    # 创建一个布局，主图占90%，说明占10%
    layout(matrix(c(1, 2), nrow = 2, ncol = 1), heights = c(0.9, 0.1))
    
    # 绘制相关性图
    corrplot(cor_result$r, 
             col = addcol(100), 
             method = "circle",  
             tl.col = "black", 
             tl.cex = 0.8, 
             tl.srt = 45,
             tl.pos = "lt",
             p.mat = cor_result$p, 
             diag = TRUE, 
             type = 'upper', 
             sig.level = c(0.001, 0.01, 0.05), 
             pch.cex = 1.5,
             insig = 'label_sig', 
             pch.col = 'grey20', 
             order = 'original',
             mar = c(0, 0, 0, 0))
    
    # 添加下三角（数字）
    corrplot(cor_result$r,
             col = addcol(100), 
             method = "number", 
             type = "lower", 
             tl.col = "n", 
             tl.cex = 0.1,
             cl.pos = 'n',
             number.cex = 0.9, 
             tl.pos = "n",
             order = 'original',
             add = TRUE,
             insig = 'blank',
             number.digits = 2,
             mar = c(0, 0, 0, 0))
    
    # 在底部添加显著性标记说明
    par(mar = c(0, 0, 0, 0))
    plot.new()
    legend("center", 
           legend = c("显著性标记说明:", 
                      "*** p < 0.001 (极显著)", 
                      "**  p < 0.01  (非常显著)", 
                      "*   p < 0.05  (显著)",
                      "空白 p ≥ 0.05  (不显著)"),
           bty = "n", 
           cex = 1.2,
           text.col = "black")
  }
  
  # 保存为PNG格式
  png("catechin_correlation_plot.png", 
      width = 3000, 
      height = 3400,  # 增加高度以容纳显著性说明
      res = 300)
  plot_correlation()
  dev.off()
  cat("PNG相关性图已保存为: catechin_correlation_plot.png\n")
  
  # 保存为PDF格式
  pdf("catechin_correlation_plot.pdf", 
      width = 10, 
      height = 12)  # 增加高度以容纳显著性说明
  plot_correlation()
  dev.off()
  cat("PDF相关性图已保存为: catechin_correlation_plot.pdf\n")
  
  # 输出统计摘要
  cat("\n====================== 统计摘要 ======================\n")
  cat("样本数量: ", nrow(catechin_data), "\n")
  cat("变量数量: ", ncol(catechin_data), "\n")
  cat("\n================ 显著性水平说明 ================\n")
  cat("显著性标记:\n")
  cat("  *** 表示 p < 0.001 (极显著)\n")
  cat("  **  表示 p < 0.01  (非常显著)\n")
  cat("  *   表示 p < 0.05  (显著)\n")
  cat("  空白表示 p ≥ 0.05  (不显著)\n")
  cat("\n================ 相关性矩阵 ===================\n")
  print(round(cor_result$r, 3))
  cat("\n=============== 显著性水平(p值) ===============\n")
  print(round(cor_result$p, 4))
  
  # 输出显著性关系统计
  cat("\n=========== 显著性关系统计 ===========\n")
  p_values <- cor_result$p
  p_values[lower.tri(p_values, diag = TRUE)] <- NA  # 去除对角线和下三角
  
  sig_001 <- sum(p_values < 0.002, na.rm = TRUE)
  sig_01 <- sum(p_values < 0.01 & p_values >= 0.001, na.rm = TRUE)
  sig_05 <- sum(p_values < 0.05 & p_values >= 0.01, na.rm = TRUE)
  non_sig <- sum(p_values >= 0.05, na.rm = TRUE)
  total_pairs <- sum(!is.na(p_values))
  
  cat("变量对总数: ", total_pairs, "\n")
  cat("极显著关系(p < 0.002): ", sig_001, 
      sprintf("(%.1f%%)", sig_001/total_pairs*100), "\n")
  cat("非常显著关系(p < 0.01):  ", sig_01, 
      sprintf("(%.1f%%)", sig_01/total_pairs*100), "\n")
  cat("显著关系(p < 0.05):      ", sig_05, 
      sprintf("(%.1f%%)", sig_05/total_pairs*100), "\n")
  cat("不显著关系(p ≥ 0.05):    ", non_sig, 
      sprintf("(%.1f%%)", non_sig/total_pairs*100), "\n")
  
  cat("\n分析完成！\n")
  cat("================================================\n")
  
}, error = function(e) {
  cat("错误: ", e$message, "\n")
})
