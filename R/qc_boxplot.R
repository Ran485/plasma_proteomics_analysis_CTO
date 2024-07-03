# Load necessary libraries
library(ggplot2)

# Load data
df <- read.csv("/Volumes/Samsung_T5/xinjiang_CTO/results/discovery_qc.csv")
meta <- read.csv("/Volumes/Samsung_T5/xinjiang_CTO/results/discovery_qc_group.csv", header = TRUE)

# Function to plot custom boxplot with group colors
plot_custom_boxplot <- function(data_matrix, meta_data, color_dict = NULL) {
  # Validate that 'meta_data' contains 'Sample' and 'Group' columns
  if (!all(c("Sample", "Group") %in% colnames(meta_data))) {
    stop("Meta data must contain 'Sample' and 'Group' columns")
  }

  # Ensure there are no missing values in 'Sample' column
  if (any(is.na(meta_data$Sample))) {
    stop("Meta$Sample contains NA values")
  }

  # Get sample order
  sample_order <- meta_data$Sample

  # Check if sample order matches columns in 'data_matrix'
  if (!all(sample_order %in% colnames(data_matrix))) {
    stop("Meta$Sample contains samples not present in 'data_matrix'")
  }

  # Reorder 'data_matrix' columns according to sample order
  data_matrix <- data_matrix[, sample_order]

  # Get unique group names
  group_names <- unique(meta_data$Group)

  # Prepare color dictionary
  if (!is.null(color_dict)) {
    # Check if 'color_dict' contains all group names
    if (any(!group_names %in% names(color_dict))) {
      warning("color_dict is missing some group color definitions, default colors will be used")
      missing_groups <- setdiff(group_names, names(color_dict))
      group_colors <- rainbow(length(group_names))
      names(group_colors)[names(group_colors) %in% missing_groups] <- missing_groups
      color_dict <- setNames(group_colors, group_names)
    }
  } else {
    # Generate default color dictionary
    group_colors <- rainbow(length(group_names))
    color_dict <- setNames(group_colors, group_names)
  }

  # Assign colors to each sample
  sample_colors <- color_dict[meta_data$Group]

  # Plot boxplot
  boxplot(
    data_matrix,
    las = 2,
    col = sample_colors,
    outline = FALSE,
    main = "Custom Boxplot with Group Colors"
  )

  # Add legend
  legend(
    "topright",
    legend = names(color_dict),
    fill = color_dict,
    title = "Group",
    cex = 0.8
  )
}

# Example call
# Assuming 'data_matrix' and 'meta_data' are defined, and using a custom color dictionary
plot_custom_boxplot(df, meta)

# Using a custom color dictionary
custom_color_dict <- c("Group1" = "red", "Group2" = "blue")
plot_custom_boxplot(log2(df), meta, color_dict = custom_color_dict)
