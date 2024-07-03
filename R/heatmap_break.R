library(pheatmap)
library(RColorBrewer)
matrix <- read.csv("/Volumes/Samsung_T5/xinjiang_CTO/results/Cox-model/heatmap.csv",
  header = T, row.names = 1
)

# matrix = data.frame(t(matrix))
matrix[is.na(matrix)] <- 0
matrix <- log2(matrix + 1)

# color for subtype
annotation_col <- read.csv("/Volumes/Samsung_T5/xinjiang_CTO/results/Cox-model/heatmap_ana.csv", header = T, row.names = 1)
# rownames(annotation_col) = colnames(matrix)

chending_color <- c(
  colorRampPalette(c("#1E90FF", "white"))(30),
  colorRampPalette(c("white", "red"))(30)
)

breaks <- unique(c(seq(-1.5, 0, length = 31), 0, seq(0, 1.5, length = 31)))

pheatmap(matrix,
  scale = "row",
  cluster_rows = 0, cluster_cols = 0,
  clustering_distance_cols = "correlation", fill = T, breaks = breaks,
  clustering_distance_rows = "correlation", border_color = "gray", na_col = "#EDEEEF",
  col = chending_color, show_rownames = T, show_colnames = F, display_numbers = F,
  width = 4.85, height = 5, fontsize_number = 18, number_color = "black", number_format = "%.2f",
  annotation_col = annotation_col
) # , annotation_colors = ann_colors) #, annotation_row = annotation_row)
