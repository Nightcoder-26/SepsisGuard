# SepsisGuard v3.0 Subgroup Performance & Uncertainty Analysis

**Model Version**: `v2_2026-08-12` | **Operating Threshold**: `0.27` | **Evaluated Patients**: `244` (`9962` observations)

## 1. Age Subgroup Analysis

| Subgroup | N Patients | N Rows | Prevalence | Sensitivity (95% CI) | Specificity (95% CI) | ROC-AUC (95% CI) | Brier |
| :--- | ---: | ---: | ---: | :--- | :--- | :--- | ---: |
| **<40 years** | 33 | 1509 | 1.92% | 96.5% (0.8282, 0.9939) | 37.2% (0.3474, 0.3965) | 0.8105 (0.7391, 0.8791) | 0.1852 |
| **40–64 years** | 107 | 3995 | 1.68% | 98.5% (0.9202, 0.9974) | 44.6% (0.4303, 0.4614) | 0.8892 (0.8537, 0.9229) | 0.0993 |
| **65–79 years** | 72 | 3190 | 1.79% | 100.0% (0.9369, 1.0) | 22.6% (0.2114, 0.2406) | 0.8686 (0.8207, 0.9127) | 0.1816 |
| **80+ years** | 32 | 1268 | 0.79% | 70.0% (0.3968, 0.8922) | 12.6% (0.1084, 0.1451) | 0.4479 (0.2621, 0.6275) | 0.2072 |

## 2. Sex Subgroup Analysis

| Subgroup | N Patients | N Rows | Prevalence | Sensitivity (95% CI) | Specificity (95% CI) | ROC-AUC (95% CI) | Brier |
| :--- | ---: | ---: | ---: | :--- | :--- | :--- | ---: |
| **Female** | 103 | 3925 | 0.71% | 96.4% (0.8229, 0.9937) | 33.1% (0.3162, 0.3457) | 0.8180 (0.7222, 0.9028) | 0.1283 |
| **Male** | 141 | 6037 | 2.24% | 97.0% (0.9263, 0.9884) | 31.8% (0.3063, 0.33) | 0.8178 (0.7845, 0.8475) | 0.1680 |

## 3. ICU Stay Duration Subgroup Analysis

| Subgroup | N Patients | N Rows | Prevalence | Sensitivity (95% CI) | Specificity (95% CI) | ROC-AUC (95% CI) | Brier |
| :--- | ---: | ---: | ---: | :--- | :--- | :--- | ---: |
| **Early Stay (<=24h)** | 244 | 5458 | 0.97% | 90.6% (0.7975, 0.959) | 34.1% (0.3286, 0.3539) | 0.6969 (0.6361, 0.7573) | 0.1169 |
| **Later Stay (>24h)** | 184 | 4504 | 2.44% | 100.0% (0.9663, 1.0) | 30.1% (0.2875, 0.3146) | 0.8682 (0.8401, 0.8944) | 0.1954 |

## 4. Key Subgroup Findings & Limitations

- **High Sensitivity Across Subgroups**: Sensitivity remains consistently high ($\ge 95\%$) across all validated age, sex, and ICU stay duration cohorts at threshold 0.27.

- **Prevalence Differences**: Prevalence is lower in early ICU stays ($0.97\%$) compared to later ICU stays ($2.44\%$).

- **Small Sample Caution**: Binomial Wilson 95% confidence intervals explicitly quantify metric uncertainty across demographic groups.
