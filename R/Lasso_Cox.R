library(glmnet)
library(survival)
library(pheatmap)
library(dplyr)
library(survminer)

# Load data
df <- read.csv("/Volumes/Samsung_T5/xinjiang_CTO/results/Cox-model/discovery_cohort/single-cox.csv", col.names = 1)
exprSet <- read.csv("/Volumes/Samsung_T5/xinjiang_CTO/results/Cox-model/discovery_cohort/protein-matrix-cox-v2.csv")
genes <- rownames(df)
x <- as.matrix(exprSet[, genes])

# Scale the data
x <- scale(x, center = TRUE, scale = TRUE)
y <- Surv(exprSet$time, exprSet$status == 1)

# Fit Lasso Cox model
lasso <- glmnet(x, y, family = "cox", alpha = 1, nlambda = 1000)
plot(lasso)

# Cross-validation for the Lasso Cox model
fitCV <- cv.glmnet(x, y, family = "cox", type.measure = "deviance", nfolds = 10)
plot(fitCV)

# Extract genes selected by Lasso
fitCV$lambda.min
coefficient <- coef(fitCV, s = "lambda.min")
Active.index <- which(as.numeric(coefficient) != 0)
Active.coefficient <- as.numeric(coefficient)[Active.index]
sig_gene_mult_cox <- rownames(coefficient)[Active.index]
sig_gene_mult_cox

# Build Cox model with selected genes
DEG_met_expr.lasso_cox <- exprSet %>%
  dplyr::select(status, time, all_of(sig_gene_mult_cox))
multiCox <- coxph(Surv(time, status) ~ ., data = DEG_met_expr.lasso_cox)
summary(multiCox)

# Calculate risk score
riskScore <- predict(multiCox, type = "risk", newdata = DEG_met_expr.lasso_cox)
riskScore <- as.data.frame(riskScore)
riskScore$sample <- rownames(riskScore)
head(riskScore, 2)

# Divide risk scores into high and low risk groups
riskScore_cli <- cbind(riskScore, exprSet)
riskScore_cli$riskScore2 <- ifelse(riskScore_cli$riskScore > median(riskScore_cli$riskScore), "High", "Low")

# Kaplan-Meier survival analysis
fit <- survfit(Surv(time, status) ~ riskScore2, data = riskScore_cli)
lasso_KM <- ggsurvplot(fit,
  data = riskScore_cli,
  pval = TRUE,
  risk.table = TRUE,
  surv.median.line = "hv",
  legend.title = "RiskScore",
  title = "Overall survival",
  ylab = "Cumulative survival (percentage)", xlab = "Time (Days)",
  censor.shape = 124, censor.size = 2, conf.int = FALSE
)
lasso_KM

# ROC curve analysis
library(timeROC)
with(
  riskScore_cli,
  ROC_riskscore <<- timeROC(T = time, delta = status, marker = riskScore, cause = 1, weighting = "marginal", times = c(12, 24, 48), ROC = TRUE, iid = TRUE)
)
plot(ROC_riskscore, time = 12, col = "#3C5588", add = FALSE, title = "The ROC curve for predicting MACEs")
plot(ROC_riskscore, time = 24, col = "#0E9F87", add = TRUE)
plot(ROC_riskscore, time = 48, col = "#F94F21", add = TRUE)
legend("bottomright", c("1-Year", "3-Year", "5-Year"), col = c("#3C5588", "#0E9F87", "#F94F21"), lty = 1, lwd = 2)
text(0.5, 0.2, paste("1-Year AUC =", round(ROC_riskscore$AUC[1], 3)))
text(0.5, 0.15, paste("2-Year AUC =", round(ROC_riskscore$AUC[2], 3)))
text(0.5, 0.1, paste("4-Year AUC =", round(ROC_riskscore$AUC[3], 3)))

# Plot risk score, survival time, and heatmap
dt <- riskScore_cli
dt <- dt[order(dt$riskScore, decreasing = FALSE), ]
dt$id <- c(1:length(dt$riskScore))
dt$status <- ifelse(dt$status == 0, "alive", "death")
dt$status <- factor(dt$status, levels = c("death", "alive"))
dt$Risk_Group <- ifelse(dt$riskScore < quantile(dt$riskScore, 0.5), "Low Risk", "High Risk")
dt$Risk_Group <- factor(dt$Risk_Group, levels = c("Low Risk", "High Risk"))
head(dt)

# Risk score scatter plot
p1 <- ggplot(dt, aes(x = id, y = log10(riskScore))) +
  geom_point(aes(col = Risk_Group)) +
  scale_colour_manual(values = c("blue", "red")) +
  geom_hline(yintercept = quantile(dt$riskScore, 0.1), colour = "grey", linetype = "dashed", size = 0.8) +
  geom_vline(xintercept = sum(dt$Risk_Group == "Low Risk"), colour = "grey", linetype = "dashed", size = 0.8) +
  theme_bw() +
  theme(
    panel.border = element_blank(), panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"),
    axis.text.x = element_text(angle = 0, hjust = 0.6, colour = "black", family = "ArialMT", size = 16),
    axis.title.x = element_text(angle = 0, hjust = 0.5, colour = "black", family = "ArialMT", size = 16),
    axis.text.y = element_text(family = "ArialMT", size = 16, face = "plain"),
    axis.title.y = element_text(family = "ArialMT", size = 16, face = "plain"),
    legend.text = element_text(face = "plain", family = "ArialMT", colour = "black", size = 12),
    legend.title = element_text(face = "plain", family = "ArialMT", colour = "black", size = 14)
  )
p1

# Survival time scatter plot
p2 <- ggplot(dt, aes(x = id, y = time)) +
  geom_point(aes(col = status)) +
  scale_colour_manual(values = c("red", "blue")) +
  geom_vline(xintercept = sum(dt$Risk_Group == "Low Risk"), colour = "grey", linetype = "dashed", size = 0.8) +
  theme_bw() +
  theme(
    panel.border = element_blank(), panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"),
    axis.text.x = element_text(angle = 18, hjust = 0.6, colour = "black", family = "ArialMT", size = 16),
    axis.title.x = element_text(angle = 0, hjust = 0.5, colour = "black", family = "ArialMT", size = 16),
    axis.text.y = element_text(family = "ArialMT", size = 16, face = "plain"),
    axis.title.y = element_text(family = "ArialMT", size = 16, face = "plain"),
    legend.text = element_text(face = "plain", family = "ArialMT", colour = "black", size = 12),
    legend.title = element_text(face = "plain", family = "ArialMT", colour = "black", size = 14)
  )
p2

# Heatmap of gene expression
genes <- c("ABI3BP", "AP3B1", "IGHV1.46", "MASP1", "ITGB4", "MAVS", "NDUFAF3", "F10", "CPM")
exp <- as.matrix(DEG_met_expr.lasso_cox[, genes])
mycol <- c(colorRampPalette(c("#1E90FF", "white"))(30), colorRampPalette(c("white", "red"))(30))
breaks <- unique(c(seq(-1.5, 0, length = 31), 0, seq(0, 1.5, length = 31)))
exp2 <- cbind(exp, riskScore)
exp2 <- exp2[, -11]
exp2 <- exp2[order(exp2[, "riskScore"]), ]
exp2 <- t(exp2)

annotation <- data.frame(Type = as.vector(dt[, "Risk_Group"]))
rownames(annotation) <- colnames(exp2)
annotation$Type <- factor(annotation$Type, levels = c("Low Risk", "High Risk"))
ann_colors <- list(Type = c("Low Risk" = "blue", "High Risk" = "red"))

pheatmap(exp2,
  scale = "row",
  col = mycol,
  breaks = breaks,
  cluster_rows = TRUE,
  cluster_cols = FALSE,
  show_colnames = FALSE,
  annotation_col = annotation,
  annotation_colors = ann_colors,
  annotation_legend = FALSE
)

write.csv(riskScore, "/Volumes/Samsung_T5/xinjiang_CTO/results/Cox-model/heatmap_model_riskscore.csv")
