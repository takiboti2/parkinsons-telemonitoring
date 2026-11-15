# Parkinson's Telemonitoring — Data Dictionary

## Source and interpretation

**Dataset:** `data/raw/parkinsons_updrs.csv`  
**Reference:** the dataset's **ATTRIBUTE INFORMATION** supplied with this project.  
**Dataset landing page:** [UCI Parkinsons Telemonitoring](https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring).

The descriptions below follow that attribute information. Model roles are project-specific decisions, not additional claims from the source. Column names match the raw CSV exactly. Each row represents a recorded voice measurement associated with a subject; repeated measurements from the same subject are expected.

The dictionary describes all **22 columns**: **19 predictors**, one target, one excluded score, and one grouping identifier. Types describe the logical numeric values, not an enforced processing schema or a missing-value policy.

## Subject information and clinical scores

| Raw column | Logical type | Description and encoding | Model role |
| --- | --- | --- | --- |
| `subject#` | Integer | Uniquely identifies each subject. Repeats across that subject's measurements. | Grouping and auditing only; never a predictor. |
| `age` | Numeric, integer-valued | Subject age, in years. | Predictor. |
| `sex` | Integer code | Source-described gender encoding: `0` = male, `1` = female. Retain the raw column name and source coding without inferring additional categories. | Predictor. |
| `test_time` | Numeric | Time since recruitment into the trial. The integer part represents the number of days since recruitment; preserve the fractional part. | Predictor. |
| `motor_UPDRS` | Numeric | Clinician's motor Unified Parkinson's Disease Rating Scale (UPDRS) score, linearly interpolated. | Regression target; never an input feature. |
| `total_UPDRS` | Numeric | Clinician's total UPDRS score, linearly interpolated. | Excluded from model inputs. |

The clinical scores are interpolated labels, not necessarily direct clinician assessments at each voice-recording time. Do not round them to integers or describe every row as an independently assessed clinical visit. The source description does not specify how to interpret fractional `test_time` values beyond the day information above, so do not infer a timestamp or time-of-day convention.

## Voice measurements

### Fundamental-frequency variation: jitter

The source groups these five attributes as measures of variation in fundamental frequency. Each is a separate numeric predictor.

| Raw column | Description | Representation notes |
| --- | --- | --- |
| `Jitter(%)` | Jitter measure of fundamental-frequency variation. | Percentage-labeled measure; preserve the supplied numeric representation without automatically multiplying or dividing by 100. |
| `Jitter(Abs)` | Absolute jitter measure of fundamental-frequency variation. | The supplied attribute information does not state a unit. |
| `Jitter:RAP` | RAP jitter measure of fundamental-frequency variation. | Preserve the source values. |
| `Jitter:PPQ5` | PPQ5 jitter measure of fundamental-frequency variation. | Preserve the source values. |
| `Jitter:DDP` | DDP jitter measure of fundamental-frequency variation. | Preserve the source values. |

### Amplitude variation: shimmer

The source groups these six attributes as measures of variation in amplitude. Each is a separate numeric predictor.

| Raw column | Description | Representation notes |
| --- | --- | --- |
| `Shimmer` | Shimmer measure of amplitude variation. | Preserve the source values. |
| `Shimmer(dB)` | Shimmer measure of amplitude variation, labeled in decibels. | Preserve the dB representation. |
| `Shimmer:APQ3` | APQ3 shimmer measure of amplitude variation. | Preserve the source values. |
| `Shimmer:APQ5` | APQ5 shimmer measure of amplitude variation. | Preserve the source values. |
| `Shimmer:APQ11` | APQ11 shimmer measure of amplitude variation. | Preserve the source values. |
| `Shimmer:DDA` | DDA shimmer measure of amplitude variation. | Preserve the source values. |

### Noise, complexity, and nonlinear measures

Each attribute below is a numeric predictor.

| Raw column | Description |
| --- | --- |
| `NHR` | Noise-to-harmonics ratio; one of the source's measures of noise relative to tonal components in the voice. |
| `HNR` | Harmonics-to-noise ratio; the other measure relating noise and tonal components in the voice. Do not assume it is the simple reciprocal of the supplied `NHR` value. |
| `RPDE` | A nonlinear dynamical complexity measure. |
| `DFA` | Signal fractal scaling exponent. |
| `PPE` | A nonlinear measure of fundamental-frequency variation. |

Except where indicated by the source description or column label, the supplied attribute information does not establish exact units, formulas, valid ranges, or conversion rules. Consult the dataset's technical references before introducing such assumptions into validation or transformations.

## Processing implications

- Preserve the raw CSV unchanged, including original column names and numeric precision.
- Predict `motor_UPDRS`; exclude `total_UPDRS` to avoid using a related clinical score as an input.
- Split and cross-validate by `subject#` so the same subject cannot occur in both fitting and evaluation partitions. Remove the identifier before fitting the estimator.
- The `LinearRegression` baseline uses only `age`, `sex`, and `test_time`. The full-feature `LinearRegression` candidate adds all 16 voice attributes for 19 predictors total.
- Do not assume repeated subject identifiers or repeated measurements are duplicate errors.
- Do not infer permissible ranges from names alone, such as requiring nonnegative `test_time` without checking source documentation.
- Define missing-value handling and schema validation explicitly in the processing pipeline. Fit learned preprocessing only on training partitions.
- If processing introduces canonical column names, maintain a versioned, explicit mapping back to these raw names. This dictionary does not rename the dataset.

See the [implementation plan](implementation-plan.md) for pipeline architecture and acceptance criteria. The system is not a clinical diagnostic service.

## Project processing policy

The raw dictionary above remains authoritative for source interpretation. The
following are versioned project decisions, not additional claims about UCI data:

- Validate all 22 raw headers before normalization; missing, unexpected, duplicate,
  or normalization-colliding names are fatal schema errors.
- Normalize with the notebook's `to_snake_case` convention. The complete mapping is
  exported in each manifest and defined in `src/parkinsons_ml/schema.py`:
  `subject#` becomes `subject`, `motor_UPDRS` becomes `motor_updrs`, `Jitter(%)`
  becomes `jitter`, and `Shimmer(dB)` becomes `shimmer_db`.
- Accepted missing tokens are empty fields, `NA`, `N/A`, `NaN`, `nan`, `NULL`, and
  `null`, after trimming whitespace. Other nonnumeric strings are malformed, not
  missing. Other spellings parsed as NaN or infinity are invalid nonfinite values.
- Drop rows with malformed/nonfinite values, missing subject/target, noninteger
  subject or nonmissing age, or sex other than missing/0/1. Warn and record every
  reason. Validate the excluded total score too; missing total score alone is allowed.
- Retain missing predictors for fold-fitted imputation. Count explicit missingness
  separately from malformed values. Subject IDs remain grouping/audit information.
- Remove later exact duplicate copies across all 22 parsed columns before clipping
  or dropping columns. Keep the first in source order. Preserve distinct repeated
  measurements, including those that become equal only after clipping.
- Clip finite negative `test_time` to zero. Preserve missingness and log the action.
  Apply the same idempotent rule in the serialized model pipeline; reject infinity.
- Report extreme continuous predictors with train-derived IQR bounds; never remove
  them. Skip flagging with an explicit status for fewer than four observed train
  values, zero IQR, or unrepresentable bounds. These are not clinical range checks.
- Export patient-disjoint `train`/`test` data without `total_updrs`; create grouped
  validation folds within train only. Missing continuous predictors use train-fitted
  medians; missing sex uses the train-fitted mode. An entirely missing fitting
  feature is a training error. Target precision is preserved.
