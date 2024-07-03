# Load necessary libraries
library(table1)
library(boot)
library(survival)
library(car)
options(stringsAsFactors = F)

# Custom function for rendering table1 with added functionality for normality and homogeneity of variance tests
rd <- function(x, name, ...) {
  y <- a[[name]]
  m <- any(is.na(y))
  if (is.numeric(y)) {
    normt <- (shapiro.test(y)$p.value > 0.1) + 2
    my.render.cont(x, n = normt, m = m, ...)
  } else {
    render.default(x = x, name = name, ...)
  }
}

# Custom function for rendering continuous variables
my.render.cont <- function(x, n, m, ...) {
  a <- with(stats.apply.rounding(stats.default(x, ...), ...), c("",
    `Mean (±SD)` = sprintf("%s (±%s)", MEAN, SD),
    `Median (IQR)` = sprintf("%s [%s, %s]", MEDIAN, Q1, Q3)
  ))[-n]
  if (m) {
    a <- c(a, with(
      stats.apply.rounding(stats.default(is.na(x), ...), ...)$Yes,
      c(`Missing` = sprintf("%s (%s%%)", FREQ, PCT))
    ))
  }
  a
}

# Function to check normality
normality <- function(x, ...) {
  y <- unlist(x)
  g <- factor(rep(1:length(x), times = sapply(x, length)))

  if (is.numeric(y)) {
    zt <- sapply(unique(g), function(i) shapiro.test(y[g == i])$p.value > 0.1)
    ztjy <- all(zt)
    normality <- if (ztjy) "Normal" else "Non-normal"
  } else {
    normality <- "-"
  }
  normality
}

# Function to check homogeneity of variance
hom.var <- function(x, ...) {
  y <- unlist(x)
  g <- factor(rep(1:length(x), times = sapply(x, length)))

  if (is.numeric(y)) {
    fcq <- leveneTest(y ~ g, center = median)$`Pr(>F)`[1] > 0.1
    hom_of_var <- if (fcq) "Homogeneous" else "Non-homogeneous"
  } else {
    hom_of_var <- "-"
  }
  hom_of_var
}

# Function to compute p-values for different tests
p <- function(x, result, ...) {
  y <- unlist(x)
  g <- factor(rep(1:length(x), times = sapply(x, length)))

  if (is.numeric(y)) {
    zt <- sapply(unique(g), function(i) shapiro.test(y[g == i])$p.value > 0.1)
    ztjy <- all(zt)
    fcq <- leveneTest(y ~ g, center = mean)$`Pr(>F)`[1] > 0.1

    if (ztjy & fcq) {
      p <- if (length(unique(g)) > 2) summary(aov(y ~ g))[[1]]$`Pr(>F)`[1] else t.test(y ~ g)$p.value
    } else {
      p <- if (length(unique(g)) > 2) kruskal.test(y ~ g)$p.value else wilcox.test(y ~ g)$p.value
    }
  } else {
    p <- chisq.test(table(y, g))$p.value
  }
  format.pval(p, digits = 3, eps = 0.001)
}

# Function to determine the statistical test used
pmethod <- function(x, ...) {
  y <- unlist(x)
  g <- factor(rep(1:length(x), times = sapply(x, length)))

  if (is.numeric(y)) {
    zt <- sapply(unique(g), function(i) shapiro.test(y[g == i])$p.value > 0.1)
    ztjy <- all(zt)
    fcq <- leveneTest(y ~ g, center = mean)$`Pr(>F)`[1] > 0.1

    if (ztjy & fcq) {
      method <- if (length(unique(g)) > 2) "ANOVA" else "t-test"
    } else {
      method <- if (length(unique(g)) > 2) "Kruskal-Wallis" else "Wilcoxon"
    }
  } else {
    method <- "Chi-square"
  }
  method
}

# # Example usage
# a <- read.csv("/path/to/your/data.csv")

# table1(
#   ~ Age + Gender + Ethnicity + Rrentrop_Grade + Heart_Rate + Comorbidities + Smoking + Family_History + Lateral_branch_blood_supply +
#     Processing + Medical_History_Year + Hospitalization_cycle + Height + Weight + Basophil_percentage |
#     Group,
#   data = a,
#   render = rd,
#   render.continuous = my.render.cont,
#   overall = FALSE,
#   droplevels = TRUE,
#   render.categorical = normality,
#   render.nonnumeric = hom.var,
#   extra.col = list(`P.value` = p, `Method` = pmethod)
# )
