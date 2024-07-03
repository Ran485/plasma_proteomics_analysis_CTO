## Loading configuration files
source("clinical_triliner_table_config.R")
library("table1")

## -------------------------------------------------------------------------
#### The section above can be reused without changes. The section below needs to be customized with your own data and some modifications.
## -------------------------------------------------------------------------

# Load data
a <- read.csv("/Volumes/Samsung_T5/xinjiang_CTO/Tables/clinical/merged_clinical_v2.csv")
colnames(a)
str(a)

# Set factors for categorical variables
a$Rrentrop_Grade <- factor(a$Rrentrop_Grade, levels = c(0, 1, 2, 3), labels = c("Grade0", "Grade1", "Grade2", "Grade3"))
a$Heart_failure <- factor(a$Heart_failure, levels = c(0, 1), labels = c("Health", "Heart_failure"))
a$Gender <- factor(a$Gender, levels = c(0, 1), labels = c("Female", "Male"))
a$Ethnicity <- factor(a$Ethnicity, levels = c(0, 1), labels = c("Ethnic Han", "Ethnic Minorities"))
a$Comorbidities <- factor(a$Comorbidities, levels = c(0, 1), labels = c("Non-comorbidities", "Comorbidities"))
a$Smoking <- factor(a$Smoking, levels = c(0, 1), labels = c("Non-Smoking", "Smoking"))
a$Family_History <- factor(a$Family_History, levels = c(0, 1), labels = c("Non-Family_History", "Family_History"))
a$Lateral_branch_blood_supply <- factor(a$Lateral_branch_blood_supply, levels = c(0, 1), labels = c("Non-LBBS", "LBBS"))
a$Processing <- factor(a$Processing, levels = c(0, 1), labels = c("Non-Processing", "Processing"))

# Set units for continuous variables
units(a$Age) <- "years"
units(a$Medical_History_Year) <- "years"
units(a$Hospitalization_cycle) <- "weeks"
units(a$Height) <- "cm"
units(a$Weight) <- "Kg"
units(a$Heart_Rate) <- "Bpm"
units(a$Basophil_percentage) <- "%"

# Generate tables
table1(
    ~ Age + Gender + Ethnicity + Rrentrop_Grade + Heart_Rate + Comorbidities + Smoking + Family_History + Lateral_branch_blood_supply +
        Processing + Medical_History_Year + Hospitalization_cycle + Height + Weight + Basophil_percentage +
        Lymphocyte.count + Serum.HDL.cholesterol + Eosinophil.count + Basophil.count + Albumin.ratio +
        Lactate.Dehydrogenase + Plasma.prothrombin.time + Plasma.fibrinogen + Alkaline.phosphatase +
        Chlorine | LBC_circulation,
    data = a, topclass = "Rtable1-zebra", render = rd,
    droplevels = TRUE, overall = FALSE,
    render.continuous = my.render.cont,
    # You can change the names of the new columns in the list below
    extra.col = list(`P.method` = pmethod, `P.value` = p)
)

table1(
    ~ Age + Gender + Ethnicity + LBC_circulation + Rrentrop_Grade + Heart_Rate + Comorbidities + Smoking + Family_History + Lateral_branch_blood_supply +
        Processing + Medical_History_Year + Hospitalization_cycle + Height + Weight + Basophil_percentage +
        Lymphocyte.count + Serum.HDL.cholesterol + Eosinophil.count + Basophil.count + Albumin.ratio +
        Lactate.Dehydrogenase + Plasma.prothrombin.time + Plasma.fibrinogen + Alkaline.phosphatase +
        Chlorine | Cohort,
    data = a, topclass = "Rtable1-zebra", render = rd,
    droplevels = TRUE, overall = FALSE,
    render.continuous = my.render.cont,
    # You can change the names of the new columns in the list below
    extra.col = list(`P.method` = pmethod, `P.value` = p)
)

table1(~ Age + Gender + Ethnicity + LBC_circulation + Rrentrop_Grade + Heart_Rate + Comorbidities + Smoking + Family_History + Lateral_branch_blood_supply +
    Processing + Medical_History_Year + Hospitalization_cycle + Height + Weight + Basophil_percentage +
    Lymphocyte.count + Serum.HDL.cholesterol + Eosinophil.count + Basophil.count + Albumin.ratio +
    Lactate.Dehydrogenase + Plasma.prothrombin.time + Plasma.fibrinogen + Alkaline.phosphatase +
    Chlorine | Cohort, data = a)
