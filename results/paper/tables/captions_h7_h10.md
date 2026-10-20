# Captions of the H7-H10 tables and figure

H7-H10 were added after the main analysis, with their criteria fixed before running (README, "Criteria H7-H10"). All
use Dreaddit, the 7 models of Table 6, alpha = 0.10 and the 1,000 grouped resplits of calibration + test. For encoders,
each resplit is first averaged over the 5 seeds, and the tables give the median, mean and 2.5-97.5% quantiles (q025,
q975) over resplits, with the between-seed spread as seed_min and seed_max. Workload is the share of posts not
dismissed (flagged or referred) per 1,000 screened at risk prevalence 0.10, computed on the test half.

**Table 7 (table7_hypotheses_h7_h10.csv).** Criteria of H7-H10 in the format of Table 6, with one row per model-level
quantity, per count of models or cells meeting a criterion and per verdict. Rows marked descriptive have no criterion.

**appendix_h7_workload_matched.csv (H7).** M1 at a workload-matched alpha' against M0 at alpha = 0.10, with M1 and M0
calibrated on clean data. In each resplit and for each seed, alpha' is chosen on the calibration half as the value of
the grid 0.02, 0.03, ..., 0.20 nearest to 0.10 at which M1's in-sample clean workload is not lower than M0's (the smaller
alpha on a tie). When no grid value reaches M0's workload, alpha' is 0.02 and the resplit counts as unmatched
(column unmatched, its mean being the share). Metrics per condition are alpha', the calibration-half workloads of M0 and
of M1 at alpha', c1 of M0 and of M1 at alpha' on the test half, their paired difference c1_difference = c1(M1 at alpha')
- c1(M0), both test-half workloads and the workload gap M1(alpha') - M0. identity_max_abs_difference is the largest
difference between the recomputed alpha = 0.10 points and the stored main results.

**appendix_h7_c1_vs_workload (figure and CSV, H7).** Risk-class coverage c1 against workload per 1,000 over the alpha
grid 0.02-0.20 for M0 and M1 (clean calibration) and M3 (calibration in the test condition), on clean texts and under
head_25, as medians over resplits. The enlarged point on each line is alpha = 0.10, and the dotted line marks 0.90.

**appendix_h8_calibration_risk_10pct.csv (H8).** In each resplit the risk rows of the calibration half are thinned to
round(0.10 n1) rows (about 37), drawn without replacement once per resplit and shared by all models, while the test half
is unchanged. Rows give c1 and workload of M0, M1 and M0-LS (label-shift M0 targeting the test half's risk share) in
every condition, and the paired c1(M1) - c1(M0). The threshold comparison columns q_M1_below_q_M0, q_M1_equal_q_M0 and
q_M1_above_q_M0 are 0/1 per seed and resplit, so their mean is the share of resplits (q_M1 is M1's risk-class threshold
and q_M0 the M0 threshold, both log-odds scores). calibration_slack_c1_M0 is the in-sample c1 of M0 on the thinned
calibration half minus 0.90. The threshold and slack columns do not depend on the condition.

**appendix_h9_oracle_cost.csv (H9).** The oracle flags a post when its risk score is at most t, where t is the
ceil(0.9 n1)-th smallest risk score of the calibration half in the same condition, and dismisses it otherwise. This is
the smallest threshold with in-sample c1 of at least 0.90, uncorrected as in B1. oracle_risk_in_set_if_p_at_least is
that threshold as a p(risk) cut-off. Rows give c1 and workload of the oracle, M1 (clean calibration) and M3 (calibration
in the test condition), and workload_ratio_M3_over_oracle and workload_ratio_M1_over_oracle, the ratios of the
seed-averaged workloads within each resplit. For calibration halves of 369-370 risk rows, ceil(0.9 n1) equals the
conformal rank ceil(0.9 (n1 + 1)), so the oracle threshold coincides with M3's risk-class threshold, and the ratio
differs from 1 only through M3's empty sets.

**appendix_h10_weighted_conformal.csv (H10).** Weighted split conformal prediction (Tibshirani et al., 2019) with
density-ratio weights from a logistic-regression domain classifier fitted on the validation split only. Its features
are word count and the share of lower-cased words outside the training vocabulary, computed on validation texts
degraded by the study's degradation code. The source of the classifier is the same validation posts, each in one random
condition as for M3-mix (variant a), or clean (variant b), and its target is those posts in the test condition. Weights
use text features only, and shifts in model scores at fixed length and vocabulary are not corrected. Variant a weights
the M3-mix calibration set and variant b the clean calibration set. Method WCP is the marginal and WCP-class the
class-conditional weighted method. Metrics are c1, workload per 1,000, the share of full sets, the share of infinite
thresholds (over test posts, and over test posts x candidate classes for WCP-class), the effective sample size
ess = (sum w)^2 / sum w^2 of the calibration weights (ess_risk for its risk rows) and max_weight_ratio, the largest over
smallest weight among the calibration and test posts of a resplit. Rows with variant "-" give M0, M1 and M3-mix from the
main results. c1_band_low and c1_band_high are the c1 reference band at the median risk-class calibration and test sizes.
