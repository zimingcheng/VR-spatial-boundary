library(tidyverse)
library(rstudioapi)
library(lme4)
library(ggplot2)
library(ggiraphExtra)
library(emmeans)
library(effsize)

# ==== Load dataset ====
# change working directory 
path <- dirname(getSourceEditorContext()$path)
setwd(path)

# ==== TE_route ====
df_TE_route <- read.csv('cleaned/df_TE_route.csv')

df_TE_route_group <- df_TE_route %>%
  group_by(ptp,condition) %>%
  summarise(mean_diff_TE_route_centered = mean(diff_TE_route_centered),
            mean_diff_TE_route = mean(diff_TE_route)) %>%
  ungroup()

model <- aov(mean_diff_TE_route_centered ~ condition, data=df_TE_route_group)
summary(model)

pairs(emmeans(model, ~condition))
emmeans(aov(mean_diff_TE_route ~ condition, data = df_TE_route_group), ~condition)

# Cohen's d for SLSR vs SLDR
cohen.d(mean_diff_TE_route_centered ~ condition, data=df_TE_route_group%>%filter(condition!='DLDR'))
# Cohen's d for SLDR vs DLDR
cohen.d(mean_diff_TE_route_centered ~ condition, data=df_TE_route_group%>%filter(condition!='SLSR'))
# Cohen's d for SLSR vs DLDR
cohen.d(mean_diff_TE_route_centered ~ condition, data=df_TE_route_group%>%filter(condition!='SLDR'))

# ==== TE_trial ====
df_TE <- read.csv('cleaned/df_TE.csv')

df_TE_group <- df_TE %>%
  group_by(ptp,TaskName) %>%
  summarise(mean_TE_diff_centered = mean(TE_diff_centered),
            mean_TE_diff = mean(TE_diff)) %>%
  ungroup()

model <- aov(mean_TE_diff_centered ~ TaskName, data=df_TE_group)
summary(model)

pairs(emmeans(model, ~TaskName))
emmeans(aov(mean_TE_diff ~ TaskName, data=df_TE_group), ~TaskName)

# Cohen's d for SLSR vs SLDR
cohen.d(mean_TE_diff ~ TaskName, data=df_TE_group%>%filter(TaskName!='DLDR'))
# Cohen's d for SLDR vs DLDR
cohen.d(mean_TE_diff ~ TaskName, data=df_TE_group%>%filter(TaskName!='SLSR'))
# Cohen's d for SLSR vs DLDR
cohen.d(mean_TE_diff ~ TaskName, data=df_TE_group%>%filter(TaskName!='SLDR'))

# trial order effect
df_TE$ObjectOrder <- as.factor(df_TE$ObjectOrder)
model <- aov(TE_diff ~ TaskName * ObjectOrder, data=df_TE)
summary(model)
emmeans(model, ~TaskName)
emmeans(model, ~ObjectOrder)
pairs(emmeans(model, ~ObjectOrder))

# condition order effect
df_TE$ConditionOrder_x <- as.factor(df_TE$ConditionOrder_x)
model <- aov(TE_diff ~ TaskName * ConditionOrder_x, data=df_TE)
summary(model)
emmeans(model, ~ConditionOrder_x)
pairs(emmeans(model, ~ConditionOrder_x))

# ==== OrdDisc when both objects were reexperienced ====
df_OrdDisc_R_Acc <- read.csv('cleaned/df_OrdDisc_R_Acc.csv')

df_OrdDisc_R_Acc_group <- df_OrdDisc_R_Acc %>%
  group_by(ptp,TaskName) %>%
  summarise(mean_OrdDisc_Accuracy_centered = mean(OrdDisc_Accuracy_centered),
            mean_OrdDisc_Accuracy = mean(OrdDisc_Accuracy)) %>%
  ungroup()

model <- aov(mean_OrdDisc_Accuracy_centered ~ TaskName, data=df_OrdDisc_R_Acc_group)
summary(model)


pairs(emmeans(model, ~TaskName))
emmeans(aov(mean_OrdDisc_Accuracy~ TaskName, data=df_OrdDisc_R_Acc_group), ~TaskName)


# Cohen's d for SLSR vs SLDR
cohen.d(mean_OrdDisc_Accuracy ~ TaskName, data=df_OrdDisc_R_Acc_group%>%filter(TaskName!='DLDR'))
# Cohen's d for SLDR vs DLDR
cohen.d(mean_OrdDisc_Accuracy ~ TaskName, data=df_OrdDisc_R_Acc_group%>%filter(TaskName!='SLSR'))
# Cohen's d for SLSR vs DLDR
cohen.d(mean_OrdDisc_Accuracy ~ TaskName, data=df_OrdDisc_R_Acc_group%>%filter(TaskName!='SLDR'))

# ==== Primacy effect of OrdDisc ====
df_OrdDisc_R <- read.csv('cleaned/df_OrdDisc_R.csv')
df_OrdDisc_R$ObjectOrder <- as.factor(df_OrdDisc_R$ObjectOrder)

# cannot do mean center here because this is at the trial level 
model <- aov(OrdDisc_Accuracy ~ TaskName * ObjectOrder, data=df_OrdDisc_R)
summary(model)
emmeans(model, ~ObjectOrder)
pairs(emmeans(model, ~ObjectOrder))
pairs(emmeans(model, ~TaskName * ObjectOrder))

model <- aov(OrdDisc_Accuracy ~ TaskName * ConditionOrder, data=df_OrdDisc_R)
summary(model)
emmeans(model, ~ObjectOrder)
pairs(emmeans(model, ~ObjectOrder))
pairs(emmeans(model, ~TaskName * ObjectOrder))

# ==== recognition memory ====
df_ON <- read.csv('cleaned/df_RKN_count_LDI.csv')
df_ON_group <- df_ON %>%
  group_by(ptp,TaskName) %>%
  summarise(mean_LDI_centered = mean(LDI_centered), mean_d_prime_centered = mean(d_prime_centered),
            mean_LDI = mean(LDI), mean_d_prime = mean(d_prime)) %>%
  ungroup()

model <- aov(mean_LDI_centered ~ TaskName, data=df_ON_group)
summary(model)
model <- aov(mean_d_prime_centered ~ TaskName, data=df_ON_group)
summary(model)
emmeans(aov(mean_LDI ~ TaskName, data=df_ON_group), ~TaskName)
emmeans(aov(mean_d_prime ~ TaskName, data=df_ON_group), ~TaskName)

# ==== R percentage ====
df_Rprop <- read.csv('cleaned/df_RKN_Rprop.csv')

df_Rprop_group <- df_Rprop %>%
  group_by(ptp,TaskName,ON_Detection) %>%
  summarise(mean_RK_response = mean(RK_response)) %>%
  ungroup()

model <- aov(mean_RK_response ~ TaskName * ON_Detection, data=df_Rprop_group)
summary(model)

emmeans(model, ~ON_Detection)
pairs(emmeans(model, ~ON_Detection))

# order effect
df_R <- read.csv('cleaned/df_RKN.csv')
df_R$ObjectOrder <- as.factor(df_R$ObjectOrder)
model <- aov(RK_response ~ TaskName * ObjectOrder, data=df_R)
summary(model)
model <- aov(ON_Accuracy ~ TaskName * ObjectOrder, data=df_R)
summary(model)
