# 儿茶素数据分析与可视化
# 包含相关性分析和小提琴图绘制
# 修复了grid.draw函数问题

# 1. 自动安装和加载所需包
required_packages <- c("corrplot", "openxlsx", "psych", "RColorBrewer", 
                       "ggplot2", "dplyr", "tidyr", "gridExtra", "grid")

# 检查并安装缺失的包
install_missing_packages <- function(packages) {
  new_packages <- packages[!(packages %in% installed.packages()[,"Package"])]
  if (length(new_packages) > 0) {
    cat("正在安装以下缺失的包:", paste(new_packages, collapse = ", "), "\n")
    tryCatch({
      install.packages(new_packages, dependencies = TRUE, 
                       repos = "https://cloud.r-project.org/")
      cat("安装成功！\n")
    }, error = function(e) {
      cat("安装包时出错:", e$message, "\n")
      cat("请尝试手动安装缺失的包。\n")
    })
  } else {
    cat("所有需要的包都已安装。\n")
  }
}

# 安装缺失的包
install_missing_packages(required_packages)

# 加载所有包
cat("正在加载包...\n")
for (pkg in required_packages) {
  if (require(pkg, character.only = TRUE, quietly = TRUE)) {
    cat("  ", pkg, "加载成功\n")
  } else {
    cat("  ", pkg, "加载失败，尝试重新安装...\n")
    install.packages(pkg, dependencies = TRUE, repos = "https://cloud.r-project.org/")
    if (require(pkg, character.only = TRUE, quietly = TRUE)) {
      cat("  ", pkg, "重新安装并加载成功\n")
    } else {
      cat("  ", pkg, "加载失败，某些功能可能不可用\n")
    }
  }
}

# 特别检查grid和gridExtra包
if (!require("grid", character.only = TRUE, quietly = TRUE)) {
  cat("警告: grid包加载失败，尝试重新安装...\n")
  install.packages("grid", repos = "https://cloud.r-project.org/")
  library(grid)
}

if (!require("gridExtra", character.only = TRUE, quietly = TRUE)) {
  cat("警告: gridExtra包加载失败，尝试重新安装...\n")
  install.packages("gridExtra", repos = "https://cloud.r-project.org/")
  library(gridExtra)
}

cat("\n所有包加载完成！\n\n")

# 2. 主分析函数
perform_catechin_analysis <- function(file_path) {
  # 检查文件是否存在
  if (!file.exists(file_path)) {
    stop("文件不存在，请检查路径是否正确: ", file_path)
  }
  
  tryCatch({
    # 读取数据
    cat("正在读取数据...\n")
    data <- read.xlsx(file_path, startRow = 2)
    
    # 提取Maturity列(第二列)和D1-M1列(4-12列)的数据
    # 注意：4:12是9列，对应9种儿茶素物质
    maturity_data <- data[, 2]  # 第二列是Maturity
    catechin_data <- data[, 4:12]  # 4:12列是儿茶素数据，共9列
    
    # 设置儿茶素列的列名
    col_names <- read.xlsx(file_path, rows = 1, cols = 4:12, colNames = FALSE)
    colnames(catechin_data) <- as.character(col_names[1, ])
    
    # 添加Maturity列到数据框中
    catechin_data$Maturity <- maturity_data
    
    # 删除包含缺失值的行
    catechin_data <- na.omit(catechin_data)
    
    cat("数据读取完成。总样本数:", nrow(catechin_data), "\n")
    
    # 获取Maturity的唯一值
    maturity_types <- unique(catechin_data$Maturity)
    maturity_types <- maturity_types[!is.na(maturity_types)]
    
    cat("找到的Maturity类型: ", paste(maturity_types, collapse = ", "), "\n\n")
    
    # 设置颜色方案
    addcol <- colorRampPalette(rev(brewer.pal(9, "RdBu")))
    
    # 为小提琴图创建颜色方案
    violin_colors <- addcol(3)
    
    # 确保Maturity顺序为A、B、C
    catechin_data$Maturity <- factor(catechin_data$Maturity, levels = c("A", "B", "C"))
    
    # 定义绘制相关性图的函数
    plot_correlation <- function(cor_result, maturity_type, sample_count) {
      # 设置图形边距
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
               title = paste("成熟度:", maturity_type, "(样本数:", sample_count, ")"),
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
    
    # 定义处理单个成熟度类型的函数
    process_maturity_type <- function(maturity_type) {
      cat("\n================================================\n")
      cat("正在处理成熟度类型:", maturity_type, "\n")
      cat("================================================\n")
      
      # 筛选当前类型的数据
      type_data <- catechin_data[catechin_data$Maturity == maturity_type, ]
      
      # 移除Maturity列，只保留儿茶素数据
      type_catechin_data <- type_data[, 1:9]
      
      # 检查数据是否有效
      if (nrow(type_catechin_data) < 2) {
        cat("警告: 成熟度类型", maturity_type, "的有效数据不足，跳过此类型\n")
        return(FALSE)
      }
      
      # 计算相关性矩阵
      cor_result <- corr.test(type_catechin_data, method = 'pearson', adjust = 'none')
      
      # 保存为PNG格式
      png_filename <- paste0("catechin_correlation_plot_", maturity_type, ".png")
      png(png_filename, 
          width = 3000, 
          height = 3400,
          res = 300)
      plot_correlation(cor_result, maturity_type, nrow(type_catechin_data))
      dev.off()
      cat("PNG相关性图已保存为:", png_filename, "\n")
      
      # 保存为PDF格式
      pdf_filename <- paste0("catechin_correlation_plot_", maturity_type, ".pdf")
      pdf(pdf_filename, 
          width = 10, 
          height = 12)
      plot_correlation(cor_result, maturity_type, nrow(type_catechin_data))
      dev.off()
      cat("PDF相关性图已保存为:", pdf_filename, "\n")
      
      # 输出统计摘要
      cat("\n====================== 统计摘要 ======================\n")
      cat("成熟度类型: ", maturity_type, "\n")
      cat("样本数量: ", nrow(type_catechin_data), "\n")
      cat("变量数量: ", ncol(type_catechin_data), "\n")
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
      p_values[lower.tri(p_values, diag = TRUE)] <- NA
      
      sig_001 <- sum(p_values < 0.001, na.rm = TRUE)
      sig_01 <- sum(p_values < 0.01 & p_values >= 0.001, na.rm = TRUE)
      sig_05 <- sum(p_values < 0.05 & p_values >= 0.01, na.rm = TRUE)
      non_sig <- sum(p_values >= 0.05, na.rm = TRUE)
      total_pairs <- sum(!is.na(p_values))
      
      cat("变量对总数: ", total_pairs, "\n")
      cat("极显著关系(p < 0.001): ", sig_001, 
          sprintf("(%.1f%%)", sig_001/total_pairs*100), "\n")
      cat("非常显著关系(p < 0.01):  ", sig_01, 
          sprintf("(%.1f%%)", sig_01/total_pairs*100), "\n")
      cat("显著关系(p < 0.05):      ", sig_05, 
          sprintf("(%.1f%%)", sig_05/total_pairs*100), "\n")
      cat("不显著关系(p ≥ 0.05):    ", non_sig, 
          sprintf("(%.1f%%)", non_sig/total_pairs*100), "\n")
      
      cat("\n成熟度类型", maturity_type, "分析完成！\n")
      cat("================================================\n\n")
      
      return(TRUE)
    }
    
    # 对每个成熟度类型进行分析
    cat("=================== 开始相关性分析 ===================\n")
    successful_analyses <- 0
    for (maturity_type in maturity_types) {
      if (process_maturity_type(maturity_type)) {
        successful_analyses <- successful_analyses + 1
      }
    }
    
    # 输出相关性分析总体统计
    cat("\n=============== 相关性分析总体总结 ===============\n")
    cat("总共找到成熟度类型: ", length(maturity_types), "\n")
    cat("成功分析的类型数: ", successful_analyses, "\n")
    cat("各类型样本数统计:\n")
    for (maturity_type in maturity_types) {
      type_count <- sum(catechin_data$Maturity == maturity_type, na.rm = TRUE)
      cat("  ", maturity_type, ": ", type_count, "个样本\n")
    }
    cat("\n相关性分析完成！\n")
    cat("================================================\n\n")
    
    # 制作小提琴图
    cat("=================== 开始制作小提琴图 ===================\n")
    
    # 将数据转换为长格式，便于ggplot绘图
    catechin_long <- catechin_data %>%
      pivot_longer(cols = 1:9, 
                   names_to = "Catechin", 
                   values_to = "Concentration")
    
    # 确保Catechin顺序与原始数据一致
    catechin_long$Catechin <- factor(catechin_long$Catechin, 
                                     levels = colnames(catechin_data)[1:9])
    
    # 创建9个小提琴图（每个儿茶素物质一个）
    violin_plots <- list()
    
    for (i in 1:9) {
      catechin_name <- colnames(catechin_data)[i]
      cat("正在制作", catechin_name, "的小提琴图...\n")
      
      # 筛选当前儿茶素物质的数据
      catechin_subset <- catechin_data %>%
        select(all_of(catechin_name), Maturity) %>%
        rename(Concentration = 1)
      
      # 创建小提琴图
      p <- ggplot(catechin_subset, aes(x = Maturity, y = Concentration, fill = Maturity)) +
        geom_violin(trim = TRUE, scale = "width", alpha = 0.7) +
        geom_boxplot(width = 0.1, fill = "white", alpha = 0.5, outlier.shape = NA) +
        scale_fill_manual(values = setNames(violin_colors, c("A", "B", "C"))) +
        labs(title = catechin_name,
             x = "成熟度 (Maturity)",
             y = "浓度 (毫克/克)") +
        theme_minimal() +
        theme(
          plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
          axis.title = element_text(size = 12),
          axis.text = element_text(size = 10),
          legend.position = "none"
        ) +
        stat_summary(fun = median, geom = "point", shape = 18, 
                     size = 3, color = "red")
      
      violin_plots[[i]] <- p
    }
    
    # 检查grid.draw函数是否可用
    cat("检查grid.draw函数是否可用...\n")
    if (exists("grid.draw")) {
      cat("grid.draw函数存在\n")
    } else if (exists("draw", where = asNamespace("grid"))) {
      cat("grid::draw函数存在\n")
      # 将grid::draw赋值给grid.draw
      grid.draw <- get("draw", envir = asNamespace("grid"))
    } else {
      cat("警告: 无法找到grid.draw函数，将使用备用方法\n")
    }
    
    # 将所有小提琴图组合在一个图中
    cat("正在组合所有小提琴图...\n")
    combined_plot <- grid.arrange(grobs = violin_plots, ncol = 3, nrow = 3)
    
    # 保存组合小提琴图为PNG格式
    png("combined_catechin_violin_plots.png", 
        width = 4000, 
        height = 3200, 
        res = 300)
    
    # 使用安全的绘图方法
    if (exists("grid.draw")) {
      grid.draw(combined_plot)
    } else if (exists("print", where = asNamespace("gridExtra"))) {
      # 备用方法：使用print
      print(combined_plot)
    } else {
      # 最简单的方法：直接绘制
      plot(combined_plot)
    }
    dev.off()
    cat("组合小提琴图已保存为: combined_catechin_violin_plots.png\n")
    
    # 保存组合小提琴图为PDF格式
    pdf("combined_catechin_violin_plots.pdf", 
        width = 16, 
        height = 12)
    
    if (exists("grid.draw")) {
      grid.draw(combined_plot)
    } else if (exists("print", where = asNamespace("gridExtra"))) {
      print(combined_plot)
    } else {
      plot(combined_plot)
    }
    dev.off()
    cat("组合小提琴图已保存为: combined_catechin_violin_plots.pdf\n")
    
    # 为每个儿茶素物质单独保存小提琴图
    cat("\n正在为每个儿茶素物质单独保存小提琴图...\n")
    for (i in 1:9) {
      catechin_name <- colnames(catechin_data)[i]
      
      # 筛选当前儿茶素物质的数据
      catechin_subset <- catechin_data %>%
        select(all_of(catechin_name), Maturity) %>%
        rename(Concentration = 1)
      
      # 创建更详细的小提琴图（单独保存）
      p_single <- ggplot(catechin_subset, aes(x = Maturity, y = Concentration, fill = Maturity)) +
        geom_violin(trim = TRUE, scale = "width", alpha = 0.7) +
        geom_boxplot(width = 0.1, fill = "white", alpha = 0.5, outlier.shape = NA) +
        geom_jitter(width = 0.2, size = 1.5, alpha = 0.5) +
        scale_fill_manual(values = setNames(violin_colors, c("A", "B", "C"))) +
        labs(title = paste("儿茶素:", catechin_name),
             subtitle = paste("样本数: A=", sum(catechin_data$Maturity == "A"), 
                              ", B=", sum(catechin_data$Maturity == "B"),
                              ", C=", sum(catechin_data$Maturity == "C")),
             x = "成熟度 (Maturity)",
             y = "浓度 (毫克/克)",
             caption = "红色菱形表示中位数") +
        theme_minimal() +
        theme(
          plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
          plot.subtitle = element_text(hjust = 0.5, size = 12),
          axis.title = element_text(size = 14),
          axis.text = element_text(size = 12),
          plot.caption = element_text(size = 10, face = "italic"),
          legend.position = "none"
        ) +
        stat_summary(fun = median, geom = "point", shape = 18, 
                     size = 4, color = "red")
      
      # 保存为PNG
      png_filename <- paste0("violin_plot_", gsub("[^[:alnum:]]", "_", catechin_name), ".png")
      png(png_filename, width = 2000, height = 1600, res = 300)
      print(p_single)
      dev.off()
      
      # 保存为PDF
      pdf_filename <- paste0("violin_plot_", gsub("[^[:alnum:]]", "_", catechin_name), ".pdf")
      pdf(pdf_filename, width = 8, height = 6)
      print(p_single)
      dev.off()
      
      cat("  ", catechin_name, "的小提琴图已保存\n")
    }
    
    # 输出数据统计摘要
    cat("\n=================== 数据统计摘要 ===================\n")
    cat("总样本数: ", nrow(catechin_data), "\n")
    cat("儿茶素物质数量: 9\n")
    cat("成熟度类型分布:\n")
    maturity_summary <- table(catechin_data$Maturity)
    for (maturity_type in names(maturity_summary)) {
      cat("  ", maturity_type, ": ", maturity_summary[maturity_type], 
          sprintf("(%.1f%%)", maturity_summary[maturity_type]/nrow(catechin_data)*100), "\n")
    }
    
    cat("\n各儿茶素物质在不同成熟度下的描述性统计:\n")
    for (i in 1:9) {
      catechin_name <- colnames(catechin_data)[i]
      cat("\n  ", catechin_name, ":\n")
      
      for (maturity_type in c("A", "B", "C")) {
        if (maturity_type %in% catechin_data$Maturity) {
          subset_data <- catechin_data[catechin_data$Maturity == maturity_type, catechin_name]
          cat("    ", maturity_type, ": n=", length(subset_data), 
              ", 均值=", round(mean(subset_data, na.rm = TRUE), 3),
              ", 标准差=", round(sd(subset_data, na.rm = TRUE), 3),
              ", 中位数=", round(median(subset_data, na.rm = TRUE), 3), "\n")
        }
      }
    }
    
    cat("\n================================================\n")
    cat("所有分析完成！\n")
    cat("生成的文件:\n")
    cat("1. 相关性分析图: catechin_correlation_plot_[A/B/C].png/pdf\n")
    cat("2. 组合小提琴图: combined_catechin_violin_plots.png/pdf\n")
    cat("3. 单独小提琴图: violin_plot_[儿茶素名称].png/pdf\n")
    cat("================================================\n")
    
    return(TRUE)
    
  }, error = function(e) {
    cat("错误: ", e$message, "\n")
    cat("错误发生位置:", conditionCall(e), "\n")
    return(FALSE)
  })
}

# 3. 主程序
cat("儿茶素数据分析与可视化程序\n")
cat("================================================\n")

# 设置文件路径
file_path <- "C:/Users/Mayn/Desktop/实验MAX/鲜叶-儿茶素对应/儿茶素数据/数据-儿茶素-毫克每克 - 副本 - 副本.xlsx"

# 运行分析
cat("开始分析，文件路径:", file_path, "\n")
result <- perform_catechin_analysis(file_path)

if (result) {
  cat("\n分析成功完成！\n")
} else {
  cat("\n分析过程中出现错误。\n")
}

cat("程序结束。\n")
