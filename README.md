# Methodology and Results

Multidimensional analysis of household and child deprivation in Kazakhstan, 2021–2024.
This document describes **how the study was done** and **what it found**. For the data description, the folder layout and the EDA notebooks, see [`README.md`](README.md).

---

## Contents

- [1. Research questions](#1-research-questions)
- [2. Data and its limits](#2-data-and-its-limits)
- [3. Measuring deprivation: the Alkire–Foster method](#3-measuring-deprivation-the-alkirefoster-method)
- [4. Study pipeline, step by step](#4-study-pipeline-step-by-step)
- [5. Statistical models](#5-statistical-models)
- [6. Main findings](#6-main-findings)
- [7. Demographic simulation (AnyLogic)](#7-demographic-simulation-anylogic)
- [8. Robustness checks](#8-robustness-checks)
- [9. Limitations](#9-limitations)
- [10. Where to find the results in this repository](#10-where-to-find-the-results-in-this-repository)

---

## 1. Research questions

Income does not always reflect how households actually live. In 2023, 7.8% of children in Kazakhstan lived in households below the subsistence minimum, while multidimensional child poverty reached 28.5%. In rural areas it was 34.8% against 24.4% in cities.

The study asks three questions:

1. How can household and child deprivation be measured in several dimensions at once?
2. What is associated with child deprivation: region, income, adult deprivation, housing, other household traits?
3. If deprived households have more children, what does this mean for the population of Kazakhstan by 2050?

---

## 2. Data and its limits

Four household survey forms for 2021–2024 (about 12,000 households per year):

| Form | Content |
|------|---------|
| D-002 | Subjective assessment, deprivation items (including child items GR45 and GR46) |
| D-004 | Household budget: income (`kv_vopr_11`), spending and debt payments (`kv_vopr_12`) |
| D-006 | Housing conditions, amenities, durable goods |
| D-008 | Household members: age, education, status |

**The data are synthetic.** They were generated from de-identified microdata by decision-tree synthesis. Distributions of single variables are close to the real ones, but relations between variables are weakened. Consequences:

- the direction of an association can be discussed, its strength cannot;
- a null result (for example, income) does not prove that the link is absent in real data;
- standard errors are understated.

**Panel structure.** `HOUSEHOLD_ID` is unique only within a year, so there is **no panel**. The study works with four independent annual cross-sections, and the unit of analysis is a **household-year**.

**Missing values stay missing.** "Don't know" or an empty answer is never treated as "no deprivation". An index is computed only when enough of its weight was answered.

**Age in 2021.** The birth year (`d008__GOD_ROJD`) is empty in 2021. Age classes were restored by a gradient-boosting model (`HistGradientBoostingClassifier`) trained on the other years. It recovers the age class for 94% of persons and the "household has children" flag for 98.6% of households.

---

## 3. Measuring deprivation: the Alkire–Foster method

The study uses the counting approach of Alkire and Foster, which underlies the global Multidimensional Poverty Index.

### 3.1 Deprivation score

Each indicator *j* is a binary deprivation flag *g_j* (1 = deprived, 0 = not deprived, empty = unknown) with weight *w_j*. The score of household *i* is the weighted share of deprivations among the answered indicators:

```
c_i = Σ_j ( w_j · g_ij ) / Σ_j ( w_j · answered_ij )
```

A household is **deprived (multidimensionally poor)** if `c_i ≥ k`, where *k* is the poverty threshold.

### 3.2 Three headline measures

| Measure | Meaning | Formula |
|---------|---------|---------|
| **H** (headcount ratio) | Share of deprived households: how many | `H = q / n` |
| **A** (intensity) | Average score among the deprived: how severe | `A = mean(c_i \| c_i ≥ k)` |
| **M0** (adjusted headcount) | Combines both | `M0 = H · A` |

`q` is the number of deprived households and `n` the number with a valid score. M0 decomposes exactly into contributions of single indicators and dimensions (the sum of contributions equals M0, checked numerically).

H is insensitive to the depth of deprivation: a household just above the threshold and one deprived of almost everything count equally. That is why A and M0 are reported with it.

### 3.3 Child deprivation index

- **Source:** the 14 binary items `GR45_1…14` of D-002 (clothing, shoes, fruit and vegetables, meat or fish, learning books, toys, leisure, activities, holidays with parents, and others) plus access to medical care `GR46`.
- **Structure:** 4 domains, **25% of weight each**, equal shares within a domain.
- **Validity:** a score is computed if at least 70% of the weight was answered.
- **Threshold:** `k = 1/3`, with a sensitivity range of 0.2–0.5.
- **Household level:** a household with children counts as child-deprived if at least one child respondent is deprived in that year.

### 3.4 Household deprivation index

- **Dimensions:** 8 dimensions (food, education, health, housing, finance, social and others) built from D-002, D-004, D-006 and D-008.
- **Two incomparable blocks of years.** Block **A (2023–2024)** and block **B (2021–2022)** have different questionnaires and different sets of criteria. Block B lacks the child, medical and several other items, so weights of the remaining ones are higher. Levels of deprivation are **not comparable between blocks**; compare 2023 with 2024 and 2021 with 2022 only.
- **Coverage rule:** indicators answered for fewer than 5% of households are dropped, except six forced-in indicators. A household score is valid if the observed weight is at least 0.5.
- **Threshold:** a fixed `k` (1/3 and 0.25 were run) and an **adaptive threshold** equal to the 75th percentile of scores within each year. With the adaptive threshold the deprived share is about 25% by construction, so it is used to compare the **structure** of deprivation rather than its level.

### 3.5 Weights

The main run uses **default equal weights** (dimensions of equal weight, indicators of equal weight within a dimension), as in the global MPI. Custom weights (from an expert questionnaire) are an **additional run**, kept in the `custom_weights*` output folders. Main conclusions are made on equal weights and tested against the custom ones.

---

## 4. Study pipeline, step by step

1. **Data audit.** Column structure of D-002, D-004, D-006 and D-008; completeness control of D-004 tables; check that `HOUSEHOLD_ID` is unique only within a year (the same number in different years refers to different regions and household sizes).
2. **Decoding and validation.** Meaning of income columns (`kv_vopr_11`, items `GR1…GR21`) was checked against behaviour of the data; a few synthetic-data quirks were documented rather than corrected.
3. **Variable crosswalk and harmonisation.** The same concept in different years was mapped to one variable. Same-named columns of different forms get a source prefix (`d002_subject__GR1`, `d004_vopr11__GR1`).
4. **Merging the forms and building the analytical sample.** One row per household-year; the "has children" rule is the same for all years (by age of members, `< 18`).
5. **Dependent variable.** Child deprivation by the Alkire–Foster method (section 3.3). An early version used "at least one of 14 items", which gave 39–43% of households with children; it was replaced by the multidimensional score.
6. **Household index** (section 3.4), including the A/B blocks and the thresholds.
7. **Covariates.** Material ladder (self-assessment), income per person, food insecurity (FIES scale), housing (amenities, area per person, owned housing), household structure, employment, alimony.
8. **Descriptive statistics and graphs.** By year, region, settlement type, income quartile, family type, number of children.
9. **Bivariate tests.** Chi-square with Cramér's V; correlations for continuous factors.
10. **Multivariable model.** Logistic regression (section 5).
11. **Collinearity, interactions, year-to-year dynamics.**
12. **Robustness.** Twelve alternative specifications (section 8).
13. **Hypothesis check and interpretation.** All numbers in the text are generated from computed tables.
14. **Profiles of four household groups** (with and without children × with and without deprivation) and of the child-deprived subgroup: income, debt, housing, age, self-assessment.
15. **Demographic simulation** (section 7).

---

## 5. Statistical models

**Child deprivation (household-year level).**

```
logit P(child_deprived = 1) = region fixed effects + income per person (log)
                            + adult deprivation index + housing + household traits + ...
```

- about 10,000 household-years;
- **region fixed effects**, standard errors **clustered by region**;
- **Holm correction** for multiple comparisons;
- VIF for multicollinearity;
- likelihood-ratio test for the joint significance of region.

**Why household-year, not household-quarter.** The first version produced p-values of about 10⁻⁶⁷. This was an artefact: child answers are identical in all four quarters of a year, so four rows were counted as four observations instead of one. After moving to the household-year level the effects disappeared.

**Children in the household (H2).** A separate logit for `has_children` against housing, area, amenities and income.

---

## 6. Main findings

**Level and structure of child deprivation**

- At `k = 1/3`, **6.9% (2023) and 6.7% (2024)** of households with children are deprived. Intensity is about 0.5: the deprived lack almost half of the weighted items.
- The level depends strongly on the threshold: about **14% at k = 0.2** and about **3% at k = 0.5**.
- The most common child deprivations are **development and participation items**: no holiday with parents (17.7% of households), no extracurricular activities (12%), no equipment for active leisure (12%). Basic needs (clothing, meat, medical care) are rarer, from 1.7% to 7%.

**Factors**

- **Region** is the strongest factor. The deprived share differs from under 1% to about 25% across regions, and the model explains 10.9% of variance with region against 1.6% without it.
- **Income per person** is not significant (odds ratio 1.11, p = 0.18); self-assessed well-being and food security are not significant either. Because the data are synthetic, this does not prove the absence of a link in real data.
- The **adult deprivation index** is significant, but it should be read as a common burden of deprivation of the respondent, not as a cause: one respondent answers similar questions about adults and children.
- Households with **income from property** have about 2.5 times lower odds of child deprivation.
- Other added factors have no stable statistical support. The main hypothesis is **partly confirmed**.

**Household groups**

- Households with children have lower income per person (in 2024, households without children had income about 1.8 times higher) and higher debt payments; the gap in debt payments grew from 1.5 to almost 2 times.
- Child-deprived households are larger and have the lowest income and the least housing per person (about 17 m²).
- Self-assessment does not match living conditions: in 2024, 27% of households with both general and child deprivation rated their position above average, against 24% of households with no deprivation at all.

**Household index (blocks)**

- In block A the deprived share falls from 21% to 16% (2023 to 2024, `k = 0.25`); in block B from 49% to 33% (2021 to 2022). The 2021–2022 drop may partly be a survey effect, since the 2021 question wording was not verified.

**Comparison with UNICEF (2023).** Regional patterns are similar (for example in Mangystau and Kyzylorda), with large gaps elsewhere (Turkestan: 25% here against 54% in the UNICEF study). The likely reason is methodology: this index covers a wider range of social aspects, UNICEF's is closer to basic needs.

---

## 7. Demographic simulation (AnyLogic)

**Question.** If deprived families have more children and children inherit the status of the mother, does deprivation persist as the population grows?

**Model type.** System dynamics, deterministic (no random processes), Euler integration with step 0.25 year, horizon 2024–2050.

**Structure.** Eight stocks: four age groups (0–14, 15–49, 50–64, 65+) × status (deprived `D` / non-deprived `N`).

| Flow | Rule |
|------|------|
| Births | `B_s = k_cal · TFR_s · fW · S_s,15-49`, newborns keep the mother's status |
| Aging | `S / w_g`, where `w` = 15, 35, 15 years |
| Deaths | `μ_g · m_s · S` (national mortality by age group) |
| Exit from deprivation | `λ · D` with `λ = 1/3` per year |
| Entry into deprivation | `λ · H0/(1 − H0) · N`, which keeps `H` stationary in the baseline |

**Inputs.**

- fertility `TFR_N = 2.8`, `TFR_D = 3.3` (that is 2.8 × 1.18, the ratio found in the data);
- initial deprived share `H0 ≈ 4.4%`, the population-weighted average of group shares 4.1%, 3.7%, 4.9% and 8.0% (equal weights, `k = 1/3`, 2024). This is the share of the whole population, **not** the child-deprivation rate of section 6;
- population 2024 by age group from the national statistics, about 20.03 million;
- calibration to 2024 births (376,863) through the factor `k_cal`.

**Assumptions that are not measured in the data:** exit rate `λ = 1/3` (about three years on average), the 1.18 fertility ratio carried over to the population, and all policy parameters.

**Results.**

- **Baseline.** Population and births grow, and the deprived share stays at about 4.4%. The population reaches about 25.9 million by 2050.
- **Support scenario** (exit rate ×2, entry ×0.7, reached gradually: starts in year 2, ramp 5 years). The deprived share falls from 4.4% to 1.6%, and the number of children in deprivation from about 286 to about 101 thousand.
- **Demographic side effect.** Births fall by about 3 thousand a year because deprived families have higher fertility, but the population in 2050 changes only by about **−0.2%**. Targeted support does not conflict with demographic goals.

**Interpretation.** Deprivation is worth reducing for the sake of children's well-being, not for demography; the "deprivation → fertility" channel is weak. The size and speed of the policy effect are scenario assumptions, not estimates from data. Sensitivity of results to `λ` (0.2–0.5) can be tested with a Parameter Variation experiment.

The model files (`sd_model.py`, `inputs.json`, the AnyLogic build guide) belong in the `simulation/` folder.

---

## 8. Robustness checks

- **Threshold** `k` from 0.2 to 0.5: the level of H changes strongly, the conclusion about region does not.
- **Fixed versus adaptive threshold** in the household index.
- **Equal versus custom (expert) weights.**
- **Twelve alternative model specifications:** different samples, definition of child deprivation (`any` or `all` child respondents), replacing the income variable, and adding an overall material-deprivation count instead of single regressors.
- **Collinearity:** VIF for all regressors.
- **Placebo logic for synthetic data:** conclusions are stated as directions of association only.

---

## 9. Limitations

- **No causality.** The design is observational on cross-sections, and the data are synthetic.
- **No panel.** Households cannot be followed from year to year.
- **Blocks A and B are not comparable** in level; 2021 age is model-based.
- **Choice of weights and threshold is the authors'.** Sensitivity is shown, but any single level of H depends on this choice.
- **Possible endogeneity.** The same respondent answers about adults and children, and several deprivation items are close in meaning.
- **Unknown child deprivation.** Some households with deprivation and children have no valid child score (code 2.1.3 in the profile tables); this is a data limitation.
- **Simulation parameters** (`λ`, policy strength, the fertility ratio for the whole population) are assumptions.
- **No survey weights, strata or PSU** are provided, so all estimates are unweighted sample means.

---

## 10. Where to find the results in this repository

| What | Where |
|------|-------|
| Final analysis notebook | `Checking_hypothesis/Checking_hypothesis/child_deprivation_analysis_2021_2024_final.ipynb` |
| Column completeness by year | `Checking_hypothesis/Checking_hypothesis/column_audit_master_wide.csv` |
| Graphs of the child-deprivation analysis | `output/` |
| Alkire–Foster graphs (equal weights) | `output_depr/` (block A), `output_depr/block_2021_2022/` (block B) |
| Custom-weights run | `output_depr/custom_weights/`, `output_depr/custom_weights_p75/` |
| Structure of output CSV files | `output_depr/csv_structure.md` |
| Diagnostics of suspicious columns | `output_depr/diagnostics.txt` |
| Simulation | `simulation/` |

Graphs in `output_depr/` per folder: prevalence by indicator, H/A/M0 by threshold, contribution of dimensions, score distribution, deprivation by region and settlement, deprivation by groups.
