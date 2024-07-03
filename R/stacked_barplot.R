# Load necessary libraries
library(vcd)
library(ggplot2)
library(Hmisc)

# Function to create a stacked barplot
stacked_barplot <- function(data, target = "LBC_circulation", category_var_colname = "Lateral_branch_blood_supply") {
  # Ensure columns exist in the data
  if (!all(c(target, category_var_colname) %in% colnames(data))) {
    stop("Columns specified do not exist in the data")
  }

  # Create a new column to indicate presence (1) or absence (0)
  col_1 <- paste0(category_var_colname, "_delNA")
  data[, col_1] <- ifelse(data[, category_var_colname] == 0, 0, 1)
  data[, col_1] <- factor(data[, col_1], levels = c("0", "1"), labels = c(paste0("Non_", category_var_colname), capitalize(category_var_colname)), ordered = TRUE)

  # Basic stacked barplot
  ggplot(data = data, mapping = aes_string(x = target, fill = col_1)) +
    geom_bar(stat = "count", width = 0.5, position = "fill") +
    scale_fill_manual(values = c("#FF9E29", "#86AA00", "#F94F21", "#916CA0", "#599BAD", "#DBD289")) +
    geom_text(stat = "count", aes(label = ..count..), color = "white", size = 3.5, position = position_fill(0.5)) +
    theme_minimal()

  # Remove rows with NA values for the analysis
  temp <- data[, c(target, col_1)]
  temp <- na.omit(temp)

  # Fisher's Exact Test
  res <- table(temp[, target], temp[, col_1])
  p_val <- fisher.test(res)$p.value
  fisher_test_pval <- paste0("fisher.test p_value = ", signif(p_val, 4))

  # Percentage stacked barplot
  ggplot(data = temp, mapping = aes_string(x = target, fill = col_1)) +
    geom_bar(stat = "count", width = 0.7, position = "fill") +
    scale_fill_manual(values = c("#FF9E29", "#86AA00", "#F94F21", "#916CA0", "#599BAD", "#DBD289")) +
    geom_text(stat = "count", aes(label = scales::percent(..count.. / sum(..count..))), color = "white", size = 5, position = position_fill(0.5)) +
    theme(
      panel.border = element_blank(), panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")
    ) +
    labs(
      title = category_var_colname,
      subtitle = fisher_test_pval,
      x = "",
      y = "Frequency [%]"
    ) +
    theme(
      axis.text.x = element_text(angle = 18, hjust = 0.6, colour = "black", family = "Arial", size = 16),
      axis.title.x = element_text(angle = 0, hjust = 0.5, colour = "black", family = "Arial", size = 16),
      axis.text.y = element_text(family = "Arial", size = 16, face = "plain"),
      axis.title.y = element_text(family = "Arial", size = 20, face = "plain"),
      legend.text = element_text(face = "plain", family = "Arial", colour = "black", size = 12),
      legend.title = element_text(face = "plain", family = "Arial", colour = "black", size = 14),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank()
    )
}

# Load the data
data <- read.csv("/Volumes/Samsung_T5/xinjiang_CTO/results/Cox-model/MACCE/MACCE_meta.csv")

# Create a stacked barplot for a specific category variable
stacked_barplot(data, target = "LBC_circulation", category_var_colname = "status")

# List of category variables to analyze
category_list <- c("Heart_failure", "Gender", "Ethnicity", "Comorbidities_modified", "Smoking", "Family_History", "Lateral_branch_blood_supply", "Processing", "ACEIorARB", "CCB", "ARNI", "Nitrates")

# Initialize a list to store plots
category_var_plots <- list()

# Loop through category variables and create stacked barplots
for (category_var in category_list) {
  category_var_plots[[category_var]] <- stacked_barplot(data, target = "LBC_circulation", category_var_colname = category_var)
  print(category_var_plots[[category_var]])
  # Uncomment the line below to save the plots
  # ggsave(category_var_plots[[category_var]], file=paste0("plot_", category_var, ".png"), width = 44.45, height = 27.78, units = "cm", dpi = 300)
}

# Combine multiple plots into one plot
plot_grid(plotlist = category_var_plots)
