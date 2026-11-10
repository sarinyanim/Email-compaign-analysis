# Data Dictionary: Bank Marketing Dataset

**File:** `data/bank_marketing.csv` (semicolon-separated, load with `pd.read_csv(path, sep=';')`)
**Source:** UCI Machine Learning Repository, *Bank Marketing* ("bank-additional-full" version). Moro, S., Cortez, P., & Rita, P. (2014). *A Data-Driven Approach to Predict the Success of Bank Telemarketing.* Decision Support Systems. Licensed CC BY 4.0.
**Context:** Direct phone-marketing campaigns run by a Portuguese bank between May 2008 and November 2010. Each row is one contact with one client during a campaign.

| | |
|---|---|
| Rows | 41,188 |
| Columns | 21 (20 features + 1 target) |
| Missing values (NaN) | 0, but several categorical columns use the text value `unknown` instead |
| Exact duplicate rows | 12 |
| Target | `y`: did the client subscribe to a term deposit? |
| Conversion rate | 11.3% `yes` / 88.7% `no` (imbalanced) |

---

## 1. Client profile

| Column | Type | Description | Values / range | `unknown` count |
|---|---|---|---|---|
| `age` | Numeric (int) | Client's age in years | 17 to 98, median 38 | — |
| `job` | Categorical | Type of job | admin., blue-collar, technician, services, management, retired, entrepreneur, self-employed, housemaid, unemployed, student, unknown | 330 |
| `marital` | Categorical | Marital status (`divorced` also covers widowed) | married, single, divorced, unknown | 80 |
| `education` | Categorical | Highest education level | university.degree, high.school, basic.9y, professional.course, basic.4y, basic.6y, illiterate, unknown | 1,731 |
| `default` | Categorical | Has credit in default? | no, yes, unknown | 8,597 (21%) |
| `housing` | Categorical | Has a housing loan? | yes, no, unknown | 990 |
| `loan` | Categorical | Has a personal loan? | no, yes, unknown | 990 |

## 2. Current campaign: last contact

| Column | Type | Description | Values / range | Notes |
|---|---|---|---|---|
| `contact` | Categorical | Communication channel | cellular, telephone | |
| `month` | Categorical | Month of last contact | mar–dec (no jan or feb in the data) | Heavily concentrated in May |
| `day_of_week` | Categorical | Weekday of last contact | mon, tue, wed, thu, fri | No weekend calls |
| `duration` | Numeric (int) | Length of the last call, in seconds | 0 to 4,918, median 180 | ⚠️ Only known *after* the call ends. Median is 449 s for `yes` vs 164 s for `no`. See Exercise 1, question 2. |

## 3. Campaign history

| Column | Type | Description | Values / range | Notes |
|---|---|---|---|---|
| `campaign` | Numeric (int) | Number of contacts with this client during this campaign, including the last one | 1 to 56, median 2 | Right-skewed |
| `pdays` | Numeric (int) | Days since the client was last contacted in a *previous* campaign | 0 to 27, or **999** | ⚠️ `999` is a code meaning "never contacted before" (96% of rows), not a real number of days |
| `previous` | Numeric (int) | Number of contacts before this campaign | 0 to 7, median 0 | |
| `poutcome` | Categorical | Outcome of the previous campaign | nonexistent, failure, success | `nonexistent` = no previous campaign |

## 4. Economic context

These describe the Portuguese economy at the time of the contact, not the client. Many clients share the same period, so these columns have few distinct values (10 to 26 each, except `euribor3m` with 316).

| Column | Type | Description | Frequency | Range |
|---|---|---|---|---|
| `emp.var.rate` | Numeric (float) | Employment variation rate | Quarterly | −3.4 to 1.4 |
| `cons.price.idx` | Numeric (float) | Consumer price index | Monthly | 92.20 to 94.77 |
| `cons.conf.idx` | Numeric (float) | Consumer confidence index | Monthly | −50.8 to −26.9 |
| `euribor3m` | Numeric (float) | Euribor 3-month interest rate (%) | Daily | 0.63 to 5.05 |
| `nr.employed` | Numeric (float) | Number of employees, in thousands | Quarterly | 4,963.6 to 5,228.1 |

## 5. Target

| Column | Type | Description | Values |
|---|---|---|---|
| `y` | Categorical (binary) | Did the client subscribe to a term deposit? | `yes` (4,640 rows, 11.3%), `no` (36,548 rows, 88.7%) |

---

## Data quality notes

1. **`unknown` is a value, not a NaN.** `df.isna().sum()` shows zero, but six columns contain `unknown`. Decide whether to keep it as its own category (often fine, and it can carry signal) or treat it as missing.
2. **`pdays = 999` is a sentinel code.** Leaving it as a number makes 999 look like "a very long time ago." A common fix is a flag column such as `contacted_before = pdays != 999`.
3. **Rows are contacts, ordered by date.** The file runs from May 2008 to November 2010, so the economic columns change over time. A random train/test split is acceptable for practice; a time-based split would be more realistic.
4. **Class imbalance.** With only 11.3% positives, accuracy is a misleading metric. Prefer AUC, precision and recall.
5. **12 exact duplicate rows.** Small enough to ignore, or drop with `df.drop_duplicates()`.
