library(limma)
library(ComplexHeatmap)
library(Hmisc)
library(circlize)
library(BBmisc)

options(stringsAsFactors = FALSE)
warnings("off")
rm(list = ls())

## --------- multi_group_DEP function ----------------------------
# Function to calculate Z-Score
Z_Score <- function(x) {
  res <- (x - mean(x)) / sd(x)
  return(res)
}

# Function to get subtype information from metadata
Get_subtype_info <- function(meta) {
  f <- factor(meta$type)
  table(f)
  subtype_info <- data.frame(table(f))
  rownames(subtype_info) <- subtype_info[, 1]
  subtype_list <- rownames(subtype_info) ## Get group information
  return(subtype_list)
}

### —————————————— wilcox.test and p.val-adjust function  ——————————————————————
# Function to perform Wilcoxon test and adjust p-values
wilcox_group <- function(data, ana_index, group1, group2) {
  index <- ana_index[ana_index[, 2] == group1 | ana_index[, 2] == group2, ]
  df <- data.frame(table(ana_index$type))
  rownames(df) <- df[, 1]
  sep1 <- df[1, 2]
  sep2 <- sep1 + 1
  index_col <- intersect(index$id, colnames(data))
  expset <- data[, index_col]
  index_group1 <- index[index[, 2] == group1, ]$id
  group1_expset <- expset[, index_group1]
  index_group2 <- index[index[, 2] == group2, ]$id
  group2_expset <- expset[, index_group2]
  expset <- cbind(group1_expset, group2_expset)

  # t-test p-values
  tstat.val <- apply(expset, 1, function(x) {
    t.test(x[1:sep1], x[sep2:ncol(expset)], paired = FALSE)$statistic
  })
  tstat.pval <- apply(expset, 1, function(x) {
    t.test(x[1:sep1], x[sep2:ncol(expset)], paired = FALSE)$p.value
  })

  # Combine results
  p_val <- data.frame(tstat.val, tstat.pval)
  p_val[, "FDR_t-test"] <- p.adjust(p_val[, "tstat.pval"], method = "BH", length(p_val[, "tstat.pval"]))
  p_val[, group1] <- apply(expset[, 1:sep1], 1, mean)
  p_val[, group2] <- apply(expset[, sep2:ncol(expset)], 1, mean)
  p_val$count_1 <- rowSums(expset[, 1:sep1] > 0.00001) ## Count non-NA values in each group
  p_val$count_2 <- rowSums(expset[, sep2:ncol(expset)] > 0.00001) ## Count non-NA values in each group
  p_val$FC <- p_val[, group2] / p_val[, group1]
  p_val$log2_FC <- log2(p_val$FC)

  cat("group1 =", group1, "\n")
  cat("group2 =", group2, "\n")
  cat("FC = group2/group1:", group2, "/", group1)

  return(p_val)
}

## -------------- Load files ----------------------
if (TRUE) {
  input_path <- "/Volumes/Samsung_T5/xinjiang_CTO/results/validation_data/merge-data/DEP/matrix_data_FOT_sorted.csv"
  anno_path <- "/Volumes/Samsung_T5/xinjiang_CTO/results/validation_data/merge-data/DEP/matrix_ana.csv"
  input_data <- read.csv(input_path, header = TRUE, row.names = 1, stringsAsFactors = FALSE)
  meta <- read.csv(anno_path)
  meta$id <- gsub("-", ".", meta$id)
  rownames(meta) <- meta[, 1]
  exprSet <- as.matrix(input_data)
  index <- intersect(rownames(meta), colnames(exprSet))
  exprSet <- exprSet[, index]

  # Handling missing values
  exprSet[is.na(exprSet)] <- 0
  exprSet <- log2(exprSet + 1)

  # Set working directory
  setwd("/Volumes/Samsung_T5/xinjiang_CTO/results/validation_data/merge-data/DEP/result1")
  outpath <- getwd()
}

### Set parameters for differential expression analysis
AdjPvalueCutoff <- 1
topNum <- 20
height <- 16
width <- 14
heatmap_breakNum <- 1.4

## Set colors for heatmap annotations
Tar_group <- "#2080C3"
The_remaining <- "#89BCDF"

## Set colors for heatmap
heatmap_col <- c("#2080C3", "#f7f7f7", "red")

### Get subtype information
subtype_list <- Get_subtype_info(meta)

## ----------- Differential Expression Analysis using limma ------------
for (i in subtype_list) {
  print("-----------------------------------------------------")
  cat(i, "differential protein analysis is starting", sep = " ")
  print(i)

  Group <- meta
  Group$type <- ifelse(Group$type == i, i, "The_remaining") ## Compare subtype vs remaining subtypes
  group <- factor(Group$type)
  design <- model.matrix(~ 0 + group)
  colnames(design) <- c(i, "The_remaining") ## Rename columns
  rownames(design) <- colnames(exprSet)

  fit <- lmFit(exprSet, design)
  Contrasts <- paste(i, "The_remaining", sep = "-")
  cont.matrix <- makeContrasts(Contrasts, levels = design)
  fit2 <- contrasts.fit(fit, cont.matrix)
  fit2 <- eBayes(fit2)
  diff_gene <- topTable(fit2, adjust = "BH", number = Inf, p.value = AdjPvalueCutoff)
  diff_gene$cluster <- ifelse(diff_gene$t > 0, i, "The_remaining")

  ## Perform Wilcoxon rank-sum test
  ana_index <- Group
  wilcox_result <- wilcox_group(exprSet, ana_index, group1 = "The_remaining", group2 = i)

  ## Select top 20 pathways by logFC
  diff_gene <- sortByCol(diff_gene, c("logFC"))
  topGene <- c(rev(rownames(diff_gene))[1:topNum], rownames(diff_gene)[1:topNum])
  mat <- exprSet[topGene, ]
  top_Gene <- data.frame(topGene)
  topGeneset <- data.frame(mat)

  ## Create output directory
  outpath <- paste(getwd(), i, sep = "/")
  if (!file.exists(outpath)) {
    dir.create(file.path(outpath))
  }

  ## Save results
  write.csv(diff_gene, paste(getwd(), i, "diff_gene_results_pvalue.csv", sep = "/"))
  write.csv(top_Gene, paste(getwd(), i, "top_Gene.csv", sep = "/"))
  write.csv(topGeneset, paste(getwd(), i, "topGeneset_results.csv", sep = "/"))
  write.csv(wilcox_result, paste(getwd(), i, "wilcox_results.csv", sep = "/"))

  ## Format pathway names
  rownames(mat) <- gsub("KEGG ", "", rownames(mat))
  pathway_name <- rownames(mat)
  rownames(mat) <- pathway_name

  ## Heatmap annotation
  if (TRUE) {
    Group$type <- ifelse(Group$type != "The_remaining", "Tar_group", "The_remaining")
    topAnn <- HeatmapAnnotation(
      df = Group[, "type", drop = FALSE],
      col = list(type = c("Tar_group" = Tar_group, "The_remaining" = The_remaining)),
      annotation_height = unit(1, "cm")
    )

    heat.col <- colorRamp2(c(-heatmap_breakNum, 0, heatmap_breakNum), heatmap_col)

    ## Calculate Z-score
    mat <- t(apply(mat, 1, Z_Score))
    ht <- Heatmap(mat,
      name = "Z-score", col = heat.col, top_annotation = topAnn,
      cluster_rows = FALSE, cluster_columns = FALSE, show_column_names = FALSE, show_row_names = TRUE,
      row_names_side = "right"
    )

    ## Save heatmap as PDF
    pdf(paste(getwd(), i, "top_20_pathway.pdf", sep = "/"), height = height, width = width)
    maxChar <- rownames(mat)[nchar(rownames(mat)) == max(nchar(rownames(mat)))]

    padding <- unit.c(
      unit(2, "mm"),
      grobWidth(textGrob(maxChar)) - unit(50, "mm"),
      unit(c(2, 2), "mm")
    )
    draw(ht, padding = padding, merge_legends = TRUE)
    dev.off()
  }

  cat(i, "differential protein analysis is Done", sep = " ", "\n")
}
