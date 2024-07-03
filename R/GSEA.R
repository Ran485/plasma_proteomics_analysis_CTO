## GSEA

library(DESeq2)
library(clusterProfiler)
library(dplyr)
library(ggplot2)
library(pheatmap)

## geneID 转换
data <- read.csv("/Users/ranpeng/Desktop/Desktop/项目文件/对外服务/罗俊一/results/DEP/GSEA_input.csv")
names(data)[1] <- "SYMBOL"
rownames(data) <- data[, 1]
geneid <- rownames(data)
genes <- clusterProfiler::bitr(geneid, fromType = "SYMBOL", toType = c("ENSEMBL", "ENTREZID"), OrgDb = "org.Hs.eg.db")


genes <- genes[!duplicated(genes$SYMBOL), ]
genes <- genes[!duplicated(genes$SYMBOL), ] %>% dplyr::inner_join(data, "SYMBOL")
# （3）建立输入对象并分析表达谱在GO BP和KEGG中的富集程度。
input_GSEA <- genes$FC
names(input_GSEA) <- genes$ENTREZID
input_GSEA <- sort(input_GSEA, decreasing = T)
GSEGO_BP <- gseGO(input_GSEA, ont = "BP", OrgDb = org.Hs.eg.db, nPerm = 1000, pvalueCutoff = 0.5)
GSEGO_BP <- setReadable(GSEGO_BP, org.Hs.eg.db, keyType = "ENTREZID")
GSEA_KEGG <- gseKEGG(input_GSEA, organism = "hsa", keyType = "ncbi-geneid", nPerm = 1000, pvalueCutoff = 0.5)
GSEA_KEGG <- setReadable(GSEA_KEGG, org.Hs.eg.db, keyType = "ENTREZID")
# （4）GSEA结果的可视化和解读。

# further filter out the significant gene sets and order them by NES scores.
# GO
GSEA_BP_df <- as.data.frame(GSEGO_BP) %>% dplyr::filter(abs(NES) > 1 & pvalue < 0.05 & qvalues < 0.5)
GSEA_BP_df <- GSEA_BP_df[order(GSEA_BP_df$NES, decreasing = T), ]
p <- gseaplot(GSEGO_BP, GSEA_BP_df$ID[1], by = "all", title = GSEA_BP_df$Description[1], color.vline = "gray50", color.line = "red", color = "black") # by='runningScore/preranked/all'
p <- p + annotate(
  geom = "text", x = 0.87, y = 0.85, color = "red", fontface = "bold", size = 4,
  label = paste0("NES= ", round(GSEA_BP_df$NES[1], 1), "\n", "p.adj= ", round(GSEA_BP_df$p.adjust[1], 2))
) +
  theme(panel.grid = element_line(colour = "white"))
p
# KEGG
GSEA_KEGG_df <- as.data.frame(GSEA_KEGG) %>% dplyr::filter(abs(NES) > 1 & pvalue < 0.05 & qvalues < 0.5)
GSEA_KEGG_df <- GSEA_KEGG_df[order(GSEA_KEGG_df$NES, decreasing = T), ]
i <- 1
p <- enrichplot::gseaplot2(GSEA_KEGG, geneSetID = GSEA_KEGG_df$ID[i], pvalue_table = F, ES_geom = "line") +
  annotate("text",
    x = 0.87, y = 0.85, color = "red", fontface = "bold", size = 4,
    label = paste0("NES= ", round(GSEA_KEGG_df$NES[i], 1), "\n", "p.adj= ", round(GSEA_KEGG_df$p.adjust[1], 2))
  ) +
  labs(title = GSEA_KEGG_df$Description[i]) + theme(plot.title = element_text(hjust = 0.5))
p

GSEA_KEGG$Description
# enrichplot::gseaplot2(GSEA_KEGG, c(1,5,6,8))
enrichplot::gseaplot2(GSEA_KEGG, geneSetID = c(9, 17, 12), color = c("red", "#56AF61", "#3C679E")) #
