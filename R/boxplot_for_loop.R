library(ggplot2)
library(magrittr)
library(ggpubr)
library(cowplot)
library(tidyr)
library("openxlsx")

## ---------------- Defined functions ------------------------

# Function to handle outliers using IQR method
capping_outliers_IQR <- function(df, category_var, factor = 1.5) {
  qnt <- quantile(df[, category_var], probs = c(.25, .75), na.rm = TRUE)
  caps <- quantile(df[, category_var], probs = c(.10, .90), na.rm = TRUE)
  H <- factor * IQR(df[, category_var], na.rm = TRUE)
  lower_bound <- qnt[1] - H
  upper_bound <- qnt[2] + H
  df[, category_var] <- ifelse(df[, category_var] < lower_bound, caps[1], df[, category_var])
  df[, category_var] <- ifelse(df[, category_var] > upper_bound, caps[2], df[, category_var])
  return(df)
}

# Function to plot custom boxplots with optional violin plot and outlier capping
customed_boxplot <- function(data, target = "Feature", category_var_name = "SOD1", method = "t-test", log2_transform = FALSE, capping_outliers = FALSE, violin_plot = FALSE) {
  df <- data[, c(target, category_var_name)]

  if (capping_outliers) {
    df1 <- capping_outliers_IQR(df, category_var = category_var_name)
  } else {
    df1 <- df
  }

  if (log2_transform) {
    df1[, category_var_name] <- log2(df1[, category_var_name] + 1)
  }

  if (violin_plot) {
    ggplot(df1, aes_string(x = target, y = category_var_name, color = target)) +
      geom_violin(trim = FALSE) +
      geom_boxplot(width = 0.6, position = position_dodge(0.9), outlier.colour = "red") +
      scale_color_manual(values = c("#0E9F87", "#3C5588", "#FF9E29", "#86AA00")) +
      theme_bw() +
      stat_summary(fun = mean, geom = "point", shape = 23, size = 2, position = position_dodge(width = 0.9)) +
      theme(
        panel.border = element_blank(), panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")
      ) +
      ggtitle(category_var_name) +
      theme(
        axis.text.x = element_text(angle = 30, hjust = 0.6, colour = "black", family = "ArialMT", size = 16),
        axis.title.x = element_text(angle = 0, hjust = 0.5, colour = "black", family = "ArialMT", size = 16),
        axis.text.y = element_text(family = "ArialMT", size = 16, face = "plain"),
        axis.title.y = element_text(family = "ArialMT", size = 16, face = "plain"),
        panel.border = element_blank(), axis.line = element_line(colour = "black", size = 0.8),
        legend.text = element_text(face = "plain", family = "ArialMT", colour = "black", size = 12),
        legend.title = element_text(face = "plain", family = "ArialMT", colour = "black", size = 12),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank()
      ) +
      ylab("Relative expression [log2]") +
      xlab("") +
      stat_compare_means(aes_string(group = target), label = "p.signif")
  } else {
    ggboxplot(df1,
      x = target, y = category_var_name, combine = FALSE, color = target, fill = target, alpha = 0.12,
      order = c("Good_Circulation", "Bad_Circulation"),
      palette = c("#0E9F87", "#3C5588", "#FF9E29", "#86AA00", "#F94F21", "#916CA0", "#599BAD", "#DBD289"), width = 0.5, x.text.angle = 45
    ) +
      rotate_x_text(angle = 45, hjust = 0.5, vjust = 1) +
      stat_compare_means(label.y = c(), paired = FALSE, method = "t.test") +
      labs(title = "", x = "", y = "Relative expression [log2]") +
      geom_jitter(alpha = 0.95, aes_string(colour = target), width = 0.15, height = 0) +
      theme_test() +
      theme(
        panel.border = element_blank(), panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")
      ) +
      ggtitle(category_var_name) +
      theme(
        axis.text.x = element_text(angle = 18, hjust = 0.6, colour = "black", family = "ArialMT", size = 16),
        axis.title.x = element_text(angle = 0, hjust = 0.5, colour = "black", family = "ArialMT", size = 16),
        axis.text.y = element_text(family = "ArialMT", size = 16, face = "plain"),
        axis.title.y = element_text(family = "ArialMT", size = 16, face = "plain"),
        panel.border = element_blank(), axis.line = element_line(colour = "black", size = 0.8),
        legend.text = element_text(face = "plain", family = "ArialMT", colour = "black", size = 12),
        legend.title = element_text(face = "plain", family = "ArialMT", colour = "black", size = 14),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank()
      )
  }
}

## ------------------------ Input Data -----------------------------
data <- read.csv("/Volumes/Samsung_T5/ro_junyi/script/results/XGBoost/XGB_script/Data/Combined/clinical_proteome_combined_validation_input.csv", header = TRUE)
data$Feature <- factor(data$Feature, levels = c(0, 1), labels = c("Good_Circulation", "Bad_Circulation"))

## ----------------- Analyze specified category_var_name --------------------
customed_boxplot(data, target = "Feature", category_var_name = "MIA3", log2_transform = FALSE, capping_outliers = FALSE, violin_plot = FALSE)

## --------- Loop to analyze multiple category_var_name from a list -----------
category_list <- c("Basophil_count", "Medical_History_Year", "PT_activity", "Chlorine", "Serum_HDL_cholesterol", "Erythrocyte_distribution_width")
category_var_plots <- list()

for (category_var in category_list) {
  category_var_plots[[category_var]] <- customed_boxplot(data, target = "Feature", category_var_name = category_var, log2_transform = FALSE, capping_outliers = TRUE, violin_plot = TRUE)
  print(category_var_plots[[category_var]])
  # ggsave(category_var_plots[[category_var]], file=paste0("plot_", category_var, ".png"), width = 44.45, height = 27.78, units = "cm", dpi = 300)
}

## Combine multiple plots into one
plot_grid(plotlist = category_var_plots)
