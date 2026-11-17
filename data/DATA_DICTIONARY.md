# Synthetic teaching evaluation data dictionary

Every row is one fictional response. There are 200 rows and six attributes.
Each attribute contains exactly one of `High`, `Medium`, `Low`; there are no IDs,
respondent identities, survey dates or real institutional records.

| Column | Meaning of the simulated rating | Values |
|---|---|---|
| Teaching_Clarity | Clarity of teaching explanations | High, Medium, Low |
| Subject_Knowledge | Perceived knowledge of the subject | High, Medium, Low |
| Student_Interaction | Opportunities for questions and interaction | High, Medium, Low |
| Feedback_Timeliness | Promptness of feedback | High, Medium, Low |
| Learning_Resources | Availability/usefulness of learning material | High, Medium, Low |
| Overall_Satisfaction | Overall simulated satisfaction | High, Medium, Low |

Generator: `scripts/generate_data.py`, Python `random.Random`, seed **4312026**.
First five factors are independently sampled with stated probabilities. Satisfaction
is sampled conditionally using a weighted rating index. This intentionally creates
simulated co-occurrence, rather than copying any rating column into satisfaction.
The probabilities and full mechanism are in the script and paper.

Initial factor probabilities (Low, Medium, High), sampled independently:

| Factor | Low | Medium | High |
|---|---:|---:|---:|
| Teaching_Clarity | 0.20 | 0.30 | 0.50 |
| Subject_Knowledge | 0.12 | 0.28 | 0.60 |
| Student_Interaction | 0.25 | 0.30 | 0.45 |
| Feedback_Timeliness | 0.25 | 0.40 | 0.35 |
| Learning_Resources | 0.20 | 0.40 | 0.40 |

Encode Low=0, Medium=1, High=2. Compute a weighted index with weights
(0.35, 0.15, 0.30, 0.10, 0.10) in the order above and divide by two.
Satisfaction probabilities (Low, Medium, High) are (0.05, 0.15, 0.80) for index
at least 0.70, (0.15, 0.55, 0.30) for 0.45 <= index < 0.70, and
(0.70, 0.25, 0.05) below 0.45. The generator applies Python floating-point
comparisons exactly as shown in the source.
