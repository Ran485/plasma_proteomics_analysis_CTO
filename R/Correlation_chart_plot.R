library("PerformanceAnalytics")
library("psych")

# Load data
data <- read.csv("/Volumes/Samsung_T5/xinjiana_CTO/CTO-data/discovery-QC-data_dropna5.csv", header = T, row.names = 1)
head(data)

data <- log2(data + 1)

pairs.panels(data,
    hist.col = "gray",
    method = "pearson",
    show.points = TRUE,
    stars = FALSE,
    gap = 0.05,
    pch = ".",
    ellipses = FALSE,
    scale = FALSE,
    jiggle = TRUE,
    factor = 6,
    main = "Spearmans correlation coefficients for quality control",
    col = "#ADFF2F",
    pty = "m",
    font = 2
)
