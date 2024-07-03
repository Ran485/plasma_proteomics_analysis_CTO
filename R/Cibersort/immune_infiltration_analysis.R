setwd("/Volumes/Samsung_T5/script/results/CTO-data/validation-cohort/Cibersort-res")

rm(list = ls(all = TRUE))

# ------------------------- Install and load necessary packages --------------------------------
# install.packages('devtools')
# library(devtools)
# if(!require(CIBERSORT)) devtools::install_github("Moonerss/CIBERSORT")
library(CIBERSORT)
library(ggplot2)
library(pheatmap)
library(ggpubr)
library(reshape2)
library(tidyverse)

# ---------------- Prepare data: expression matrix + immune cell types + group data ----------------
DEG_expr <- read.csv("/Volumes/Samsung_T5/script/results/CTO-data/validation-cohort/protein-matrix-V2.csv", row.names = 1)
group <- read.csv("/Volumes/Samsung_T5/script/results/CTO-data/validation-cohort/DEP_ana.csv", row.names = 1)

# Data inspection
boxplot(DEG_expr, outline = FALSE, notch = FALSE, las = 2)
qx <- as.numeric(quantile(DEG_expr, c(0., 0.25, 0.5, 0.75, 0.99, 1.0), na.rm = TRUE))
LogC <- (qx[5] > 100) || (qx[6] - qx[1] > 50 && qx[2] > 0) || (qx[2] > 0 && qx[2] < 1 && qx[4] > 1 && qx[4] < 2)
if (LogC) {
  DEG_expr <- log2(DEG_expr + 1)
  print("log2 transform is finished")
} else {
  print("log2 transform is not needed")
}

# --------------------------- Normalize data if means are inconsistent ----------------------------------
library(limma)
DEG_expr <- normalizeBetweenArrays(DEG_expr)
boxplot(DEG_expr, outline = FALSE, notch = FALSE, las = 2)

quartile_normalize <- function(df) {
  apply(df, 2, function(col) {
    q1 <- quantile(col, probs = 0.25, na.rm = TRUE)
    q3 <- quantile(col, probs = 0.75, na.rm = TRUE)
    iqr <- q3 - q1
    norm_values <- (col - q1) / iqr
    norm_values[is.na(norm_values)] <- 0
    return(norm_values)
  })
}

DEG_expr_norm <- quartile_normalize(DEG_expr)

# -------------------------- Immune cell type list ---------------------------------
LM22_local <- read.table("/Volumes/Samsung_T5/script/Cibersort/LM22.txt", header = TRUE, row.names = 1, sep = "\t")
data(LM22)
all(LM22 == LM22_local)

#  ------------------------- Perform Cibersort immune cell infiltration analysis -------------------------
DEG_expr[is.na(DEG_expr)] <- 0
result <- cibersort(sig_matrix = LM22, mixture_file = DEG_expr, perm = 500, QN = TRUE)

result <- as.data.frame(result)
write.csv(result, "cibersort_result.csv")

# --------------------------------- Visualize results --------------------------
result1 <- result[, 1:ncol(LM22)]
result1 <- result1[, apply(result1, 2, function(x) {
  sum(x) > 0
})]
write.csv(result1, "cibersort_result_FILTER.csv")

# Heatmap
pdf(file = "Heatmap.pdf", width = 10, height = 8)
pheatmap(result1,
  color = colorRampPalette(c("#4CB43C", "#FEFCFB", "#ED5467"))(100),
  border = "skyblue",
  main = "Heatmap",
  show_rownames = TRUE,
  show_colnames = TRUE,
  cexCol = 1,
  scale = "row",
  cluster_col = TRUE,
  cluster_row = FALSE,
  angle_col = 45,
  legend = FALSE,
  legend_breaks = c(-3, 0, 3),
  fontsize_row = 10,
  fontsize_col = 10
)
dev.off()

# ---------------------- Function for data transformation ----------------------
# In R, the melt function is used to convert a dataset to a "long format".
# The "long format" refers to placing multiple variable values into one column.

# -------------------- ggplot2 to draw stacked proportion chart ---------------------
# Data processing
identical(rownames(result1), group$Samples)
data <- cbind(rownames(result1), result1)
colnames(data)[1] <- "Samples"
data <- melt(data, id.vars = c("Samples"))
data <- cbind(data, group)
colnames(data) <- c("Samples", "celltype", "proportion", "group")

# Sorting the data by group and B cell proportion
data_subset <- data[data$celltype == "B cells memory", ]
data_subset <- data_subset[order(data_subset$group, -data_subset$proportion), ]
data_subset <- data_subset %>% mutate(index = row_number())
data_subset$index <- factor(data_subset$Samples, levels = data_subset$Samples)
data$Samples <- factor(data$Samples, levels = data_subset$Samples)

# Plotting
mycolors <- c(
  "#D4E2A7", "#88D7A4", "#A136A1", "#BAE8BC", "#C757AF",
  "#DF9FCE", "#D5E1F1", "#305691", "#B6C2E7", "#E8EFF7",
  "#9FDFDF", "#EEE0F5", "#267336", "#98CEDD", "#CDE2EE",
  "#DAD490", "#372E8A", "#4C862D", "#81D5B0", "#BAE8C9",
  "#A7DCE2", "#AFDE9C"
)
pdf(file = "stacked_bar_chart.pdf", width = 18, height = 4)
ggplot(data, aes(Samples, proportion, fill = celltype)) +
  geom_bar(stat = "identity", position = "fill") +
  scale_fill_manual(values = mycolors) +
  ggtitle("Proportion of immune cells") +
  theme_gray() +
  theme(axis.ticks.length = unit(3, "mm"), axis.title.x = element_text(size = 11)) +
  theme(axis.text.x = element_text(angle = 90, hjust = 0.5, vjust = 0.5)) +
  guides(fill = guide_legend(title = "Types of immune cells"))
dev.off()

# ------------------------ Function for data transformation --------------------
# pivot_longer from the tidyr package converts multiple columns in a dataframe into two columns:
# one for the original column names and one for the corresponding values.

# ------------------------ ggplot2 to draw grouped boxplot --------------------------
# Data processing
data1 <- cbind(result1, group)
data1 <- data1[, c(1:21)]
rownames(data1) <- NULL
data1 <- pivot_longer(data = data1, cols = 1:20, names_to = "celltype", values_to = "proportion")

# Plotting
pdf(file = "grouped_boxplot.pdf", width = 10, height = 8)
ggboxplot(
  data = data1,
  x = "celltype",
  y = "proportion",
  combine = TRUE,
  merge = FALSE,
  color = "black",
  fill = "group",
  palette = NULL,
  title = "TME Cell composition",
  xlab = NULL,
  ylab = "Cell composition",
  bxp.errorbar = FALSE,
  bxp.errorbar.width = 0.2,
  facet.by = NULL,
  panel.labs = NULL,
  short.panel.labs = TRUE,
  linetype = "solid",
  size = NULL,
  width = 0.8,
  notch = FALSE,
  outlier.shape = 20,
  select = NULL,
  remove = NULL,
  order = NULL,
  error.plot = "pointrange",
  label = NULL,
  font.label = list(size = 12, color = "black"),
  label.select = NULL,
  repel = FALSE,
  label.rectangle = TRUE,
  ggtheme = theme_pubr()
) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 1))
dev.off()

# -------------------------- Extract and plot a single group (good-CCC/bad-CCC) ------------------------------
data2 <- cbind(result1, group)
data2 <- data2[, c("Plasma cells", "B cells naive", "group")]
data2 <- pivot_longer(data = data2, cols = 1:2, names_to = "celltype", values_to = "proportion")
pdf(file = "single_group_boxplot.pdf", width = 10, height = 8)
ggboxplot(
  data = data2,
  x = "celltype",
  y = "proportion",
  color = "black",
  xlab = "Types of immune cells",
  ylab = NULL,
  title = "TME Cell composition",
  fill = "group",
  palette = c("#ED5462", "#81D5B0"),
  legend.position = "bottom",
  ggtheme = theme_pubr()
) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 1)) +
  stat_compare_means(
    label = "p.signif",
    method = "wilcox.test",
    hide.ns = TRUE
  )
dev.off()

#  ------------------ ggplot2 to draw grouped boxplot with significance p-values ------------------
# Data processing
data3 <- cbind(result1, group)
data3 <- data3[, c(1:21)]
data3 <- pivot_longer(data = data3, cols = 1:20, names_to = "celltype", values_to = "proportion")

# Plotting
pdf(file = "grouped_boxplot_with_pvalues.pdf", width = 12, height = 5)
ggboxplot(
  data = data3,
  x = "celltype",
  y = "proportion",
  combine = TRUE,
  merge = FALSE,
  color = "black",
  fill = "group",
  palette = c("#ED5462", "#81D5B0"),
  title = "TME Cell composition",
  xlab = NULL,
  ylab = "Cell composition",
  bxp.errorbar = FALSE,
  bxp.errorbar.width = 0.2,
  facet.by = NULL,
  panel.labs = NULL,
  short.panel.labs = TRUE,
  linetype = "solid",
  size = NULL,
  width = 0.8,
  notch = FALSE,
  outlier.shape = 20,
  select = NULL,
  remove = NULL,
  order = NULL,
  error.plot = "pointrange",
  label = NULL,
  font.label = list(size = 12, color = "black"),
  label.select = NULL,
  repel = TRUE,
  label.rectangle = TRUE,
  ggtheme = theme_pubr()
) +
  theme(axis.text.x = element_text(angle = 30, hjust = 1, vjust = 1)) +
  stat_compare_means(
    label = "p.signif",
    method = "t.test",
    ref.group = ".all.",
    hide.ns = TRUE
  )
dev.off()
