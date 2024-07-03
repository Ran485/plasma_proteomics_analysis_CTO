library(ggplot2)
library(scales)
library(ggrepel)
require(extrafont)

# Import fonts, only needs to be done once
# font_import(pattern="[A/a]rial", prompt=T)

## --------------- Scatter CV plot ------------
# Load data
QC_data_cv <- read.csv("/Volumes/Samsung_T5/xinjiana_CTO/results/CV_protein_abundance_scatter/QC_data_cv.csv", check.names = FALSE)
colnames(QC_data_cv)
genes <- QC_data_cv

# Categorize Coefficient of Variation (CV)
genes$CV_threshold <- ifelse(genes$`Coefficient of Variation [%]` < 5, "CV% < 5",
      ifelse(genes$`Coefficient of Variation [%]` < 10, "CV% < 10",
            ifelse(genes$`Coefficient of Variation [%]` < 20, "CV% < 20",
                  ifelse(genes$`Coefficient of Variation [%]` < 30, "CV% < 30", "CV% >= 30")
            )
      )
)

# Scatter plot of CV against LFQ intensity
p1 <- ggplot(genes, aes(x = LFQ_intensity, y = `Coefficient of Variation [%]`)) +
      geom_point(aes(color = CV_threshold), shape = 1) + # Change the shape of the point
      scale_size_continuous(range = c(1, 4)) +
      scale_color_manual(values = c("#515151", "red", "gray", "#7AA9CE", "#EA686B")) +
      theme_bw(base_size = 12) +
      theme(legend.position = "right") +
      geom_text_repel(
            data = subset(genes, CV_threshold == "select"),
            aes(label = Symbol),
            size = 5,
            box.padding = unit(0.35, "lines"),
            point.padding = unit(0.3, "lines")
      ) +
      scale_x_continuous(
            trans = "log10", breaks = trans_breaks("log10", function(x) 10^x),
            labels = trans_format("log10", math_format(10^.x))
      ) +
      theme(
            panel.border = element_blank(), panel.grid.major = element_blank(),
            panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")
      ) +
      ggtitle("Quality control for the mass spectrometry operation status") +
      theme(
            axis.text.x = element_text(angle = 0, hjust = 0.6, colour = "black", family = "Arial", size = 16),
            axis.title.x = element_text(angle = 0, hjust = 0.5, colour = "black", family = "Arial", size = 16),
            axis.text.y = element_text(family = "Arial", size = 16, face = "plain"),
            axis.title.y = element_text(family = "Arial", size = 20, face = "plain"),
            panel.border = element_blank(), axis.line = element_line(colour = "black", size = 0.8),
            legend.text = element_text(face = "plain", family = "Arial", colour = "black", size = 12),
            legend.title = element_text(face = "plain", family = "Arial", colour = "black", size = 12),
            panel.grid.major = element_blank(), panel.grid.minor = element_blank()
      ) +
      geom_hline(yintercept = c(5, 10, 20, 30), color = "gray", linetype = "dashed")

p1
# ggsave(p1, file = "./figure/CV_protein_abundance_scatter.pdf", width = 8, height = 5.5)

## ------ Identification of protein numbers by different CV thresholds ----------
protein_num <- as.data.frame(table(genes$CV_threshold))
colnames(protein_num) <- c("CV_thresholds", "Quantitative protein numbers")
protein_num$`Quantitative protein numbers [%]` <- round((protein_num$`Quantitative protein numbers` / sum(protein_num$`Quantitative protein numbers`) * 100), 1)
protein_num$CV_thresholds <- factor(protein_num$CV_thresholds, levels = c("CV% < 5", "CV% < 10", "CV% < 20", "CV% < 30", "CV% >= 30"))

# Bar plot of protein numbers by CV thresholds
p2 <- ggplot(data = protein_num, aes(x = CV_thresholds, y = `Quantitative protein numbers [%]`)) +
      geom_bar(stat = "identity", fill = "steelblue") +
      geom_text(aes(label = `Quantitative protein numbers [%]`), vjust = 1.6, color = "white", size = 6) +
      theme(
            panel.border = element_blank(), panel.grid.major = element_blank(),
            panel.grid.minor = element_blank(), axis.line = element_line(colour = "black")
      ) +
      ggtitle("The identification protein numbers of different CV thresholds") +
      theme(
            axis.text.x = element_text(angle = 0, hjust = 0.6, colour = "black", family = "Arial", size = 16),
            axis.title.x = element_text(angle = 0, hjust = 0.5, colour = "black", family = "Arial", size = 16),
            axis.text.y = element_text(family = "Arial", size = 16, face = "plain"),
            axis.title.y = element_text(family = "Arial", size = 20, face = "plain"),
            panel.border = element_blank(), axis.line = element_line(colour = "black", size = 0.8),
            legend.text = element_text(face = "plain", family = "Arial", colour = "black", size = 12),
            legend.title = element_text(face = "plain", family = "Arial", colour = "black", size = 12),
            panel.grid.major = element_blank(), panel.grid.minor = element_blank()
      ) +
      geom_hline(yintercept = 25, color = "gray", linetype = "dashed")

p2
# ggsave(p2, file = "./figure/protein_num_barplot.pdf", width = 7, height = 5.5)
