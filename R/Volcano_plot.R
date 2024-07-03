## ------------ Volcano Plot ---------------
library(ggplot2)
library(ggrepel)

# Load data
genes <- read.csv("/Volumes/Samsung_T5/xinjiang_CTO/results/DEP/Good_Circulation/diff_gene_results_pvalue.csv", header = TRUE)

# Check column names
colnames(genes)

# Determine significance based on P-value and logFC
genes$Significant <- ifelse(genes$P.Value < 0.05 & genes$logFC > 0.58, "Good_circulation up",
      ifelse(genes$P.Value < 0.05 & genes$logFC < -0.58, "Bad_circulation up", "Not Significant")
)

# Adjust point sizes for plotting
genes$size <- ifelse(genes$Significant == "Good_circulation up" | genes$Significant == "Bad_circulation up", abs(genes$P.Value), 0.2)

# Generate volcano plot
ggplot(genes, aes(x = logFC, y = -log10(P.Value), size = size)) +
      geom_point(aes(color = Significant)) +
      scale_size_continuous(range = c(2, 4)) +
      scale_color_manual(values = c("#EA686B", "#7AA9CE", "gray")) +
      theme_bw(base_size = 12) +
      theme(legend.position = "right") +
      geom_text_repel(
            data = subset(genes, Significant == "Good_circulation up" | Significant == "Bad_circulation up"),
            aes(label = gene),
            size = 5,
            box.padding = unit(0.35, "lines"),
            point.padding = unit(0.3, "lines")
      ) +
      theme(
            panel.border = element_blank(),
            panel.grid.major = element_blank(),
            panel.grid.minor = element_blank(),
            axis.line = element_line(colour = "black"),
            axis.text.x = element_text(angle = 0, hjust = 0.6, colour = "black", family = "Arial", size = 16),
            axis.title.x = element_text(angle = 0, hjust = 0.5, colour = "black", family = "Arial", size = 16),
            axis.text.y = element_text(family = "Arial", size = 16, face = "plain"),
            axis.title.y = element_text(family = "Arial", size = 20, face = "plain"),
            legend.text = element_text(face = "plain", family = "Arial", colour = "black", size = 12),
            legend.title = element_text(face = "plain", family = "Arial", colour = "black", size = 12)
      ) +
      geom_hline(yintercept = 1.301, color = "gray", linetype = "dashed") +
      geom_vline(xintercept = c(-0.57, 0.58), color = "gray", linetype = "dashed") +
      scale_x_continuous(limits = c(-2.5, 2.5))

## ------------ Volcano Plot for Correlation ---------------
genes <- read.csv("/Volumes/Samsung_T5/xinjiang_CTO/results/Cox-model/discovery_cohort/correlation_results.csv", header = TRUE)

# Determine significance based on P-value and Correlation
genes$Significant <- ifelse(genes$P.value < 0.05 & genes$Correlation > 0.2, "PFS Positive",
      ifelse(genes$P.value < 0.05 & genes$Correlation < -0.2, "PFS Negative", "Not Significant")
)

# Adjust point sizes for plotting
genes$size <- abs(genes$Correlation)

# Generate volcano plot
ggplot(genes, aes(x = Correlation, y = -log10(P.value), size = size)) +
      geom_point(aes(color = Significant)) +
      scale_size_continuous(range = c(2, 4)) +
      scale_color_manual(values = c("gray", "#7AA9CE", "#EA686B")) +
      theme_bw(base_size = 12) +
      theme(legend.position = "right") +
      geom_text_repel(
            data = subset(genes, Significant == "PFS Positive" | Significant == "PFS Negative"),
            aes(label = Gene),
            size = 5,
            box.padding = unit(0.35, "lines"),
            point.padding = unit(0.3, "lines")
      ) +
      theme(
            panel.border = element_blank(),
            panel.grid.major = element_blank(),
            panel.grid.minor = element_blank(),
            axis.line = element_line(colour = "black"),
            axis.text.x = element_text(angle = 0, hjust = 0.6, colour = "black", family = "Arial", size = 16),
            axis.title.x = element_text(angle = 0, hjust = 0.5, colour = "black", family = "Arial", size = 16),
            axis.text.y = element_text(family = "Arial", size = 16, face = "plain"),
            axis.title.y = element_text(family = "Arial", size = 20, face = "plain"),
            legend.text = element_text(face = "plain", family = "Arial", colour = "black", size = 12),
            legend.title = element_text(face = "plain", family = "Arial", colour = "black", size = 12)
      ) +
      geom_hline(yintercept = 1.301, color = "gray", linetype = "dashed") +
      geom_vline(xintercept = c(-0.2, 0.2), color = "gray", linetype = "dashed") +
      scale_x_continuous(limits = c(-0.5, 0.6))

## ------------ HR Volcano Plot ---------------
genes <- read.csv("/Volumes/Samsung_T5/xinjiang_CTO/results/Cox-model/HR_volcano_plot.csv", header = TRUE)

# Calculate log2 HR
genes$log2_HR <- log2(genes$HR)

# Determine significance based on log-rank p-value and log2 HR
genes$Significant <- ifelse(genes$log_rank_p < 0.05 & genes$log2_HR > 0.5, "MACEs Positive",
      ifelse(genes$log_rank_p < 0.05 & genes$log2_HR < -0.5, "MACEs Negative", "Not Significant")
)

# Adjust point sizes for plotting
genes$size <- abs(genes$log2_HR)

# Generate volcano plot
ggplot(genes, aes(x = log2_HR, y = -log10(log_rank_p), size = size)) +
      geom_point(aes(color = Significant)) +
      scale_size_continuous(range = c(2, 4)) +
      scale_color_manual(values = c("gray", "#7AA9CE", "#EA686B")) +
      theme_bw(base_size = 12) +
      theme(legend.position = "right") +
      geom_text_repel(
            data = subset(genes, Significant == "MACEs Positive" | Significant == "MACEs Negative"),
            aes(label = Gene),
            size = 5,
            box.padding = unit(0.35, "lines"),
            point.padding = unit(0.3, "lines")
      ) +
      theme(
            panel.border = element_blank(),
            panel.grid.major = element_blank(),
            panel.grid.minor = element_blank(),
            axis.line = element_line(colour = "black"),
            axis.text.x = element_text(angle = 0, hjust = 0.6, colour = "black", family = "Arial", size = 16),
            axis.title.x = element_text(angle = 0, hjust = 0.5, colour = "black", family = "Arial", size = 16),
            axis.text.y = element_text(family = "Arial", size = 16, face = "plain"),
            axis.title.y = element_text(family = "Arial", size = 20, face = "plain"),
            legend.text = element_text(face = "plain", family = "Arial", colour = "black", size = 12),
            legend.title = element_text(face = "plain", family = "Arial", colour = "black", size = 12)
      ) +
      geom_hline(yintercept = 1.301, color = "gray", linetype = "dashed") +
      geom_vline(xintercept = c(-0.2, 0.2), color = "gray", linetype = "dashed") +
      scale_x_continuous(limits = c(-0.5, 0.6)) +
      scale_y_continuous(limits = c(0, 3.5)) +
      scale_y_continuous(breaks = seq(0, 4, 1))
