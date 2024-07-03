library("factoextra")
library("FactoMineR")
library("ggthemes")
library("missMDA")

# Load data
matrix <- read.csv("/Volumes/Samsung_T5/xinjiang_CTO/results/results_20211209/PCA/plasma_data_PCA.csv",
  row.names = 1, header = TRUE
)

# Drop rows with excessive NAs
matrix <- matrix[rowSums(is.na(matrix)) < (length(matrix) - 1), ]

# Drop outliers if necessary
drop_outliers <- c("Exp112151", "Exp112170")
matrix <- matrix[, !names(matrix) %in% drop_outliers]

# Transpose matrix
matrix <- t(matrix)

# Load group annotations
group_annotation <- read.csv("/Volumes/Samsung_T5/xinjiang_CTO/results/results_20211209/PCA/PCA_anatation_sorted.csv")

# Drop outliers from group annotations
group_annotation <- group_annotation[!group_annotation$id %in% drop_outliers, ]

# Define groups
group <- group_annotation$type

# Log-transform data and replace non-positive values with NA
matrix[matrix <= 0] <- NA
matrix <- log10(matrix)

# Convert to matrix
matrix <- as.matrix(matrix)

# Perform PCA
res.pca <- PCA(matrix, graph = FALSE, scale.unit = FALSE)

# Plot PCA results
fviz_pca_ind(res.pca,
  label = "none", # Hide individual labels
  repel = TRUE,
  col.ind = group,
  legend.title = "Group",
  palette = c("#177cb0", "#dc3023"), # Set custom colors
  addEllipses = TRUE, # Add concentration ellipses
  ellipse.level = 0.95,
  title = "PCA - Two Cohorts of Plasma Raw Data"
) + theme_base(base_size = 12)

# Prepare the PCA results for plotting
pca_results <- as.data.frame(res.pca$ind$coord)

# Add the group and batch information
pca_results$group <- group_annotation$type
pca_results$batch <- group_annotation$batch

# Plot using ggplot2
ggplot(pca_results, aes(x = Dim.1, y = Dim.2, shape = group, fill = batch, color = batch)) +
  geom_point(size = 3) +
  scale_shape_manual(values = c(16, 17)) + # Set shapes manually
  scale_color_brewer(palette = "Paired") + # Set colors manually
  theme_minimal() +
  labs(title = "PCA - Two Cohorts of Plasma Raw Data") +
  theme(
    axis.text.x = element_text(angle = 0, hjust = 0.6, colour = "black", family = "ArialMT", size = 16),
    axis.title.x = element_text(angle = 0, hjust = 0.5, colour = "black", family = "ArialMT", size = 16),
    axis.text.y = element_text(family = "ArialMT", size = 16, face = "plain"),
    axis.title.y = element_text(family = "ArialMT", size = 16, face = "plain"),
    panel.border = element_blank(),
    axis.line = element_line(colour = "black", size = 0.8),
    legend.text = element_text(face = "plain", family = "ArialMT", colour = "black", size = 12),
    legend.title = element_text(face = "plain", family = "ArialMT", colour = "black", size = 14)
  )
