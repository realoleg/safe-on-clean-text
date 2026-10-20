# Captions of the H11-H13 tables

H11-H13 were added after the main analysis, with their criteria fixed before running (README, "Criteria H11-H13").
Dreaddit excerpts are cut from longer posts: "We partition the posts into contiguous five-sentence chunks for labeling"
(Turcan and McKeown, 2019, section 3.2), so short excerpts are a natural counterpart of truncation. Length is the clean
word count. Long excerpts have at least the median length of the pool (calibration + test, 81 words) and short ones at
most its lower quartile (65 words). The truncation level k = 50 was chosen mechanically, being the level of 75, 50 and
25 whose median truncated length of long excerpts (51 words) is closest to the median length of short excerpts (55
words). Dreaddit, the 7 models of Table 6, alpha = 0.10, 1,000 grouped resplits, encoders averaged over their 5 seeds
within each resplit, then median, mean and 2.5-97.5% quantiles over resplits, with seed_min and seed_max the
between-seed spread. Workload is per 1,000 screened at risk prevalence 0.10.

**Table 8 (table8_hypotheses_h11_h13.csv).** Criteria of H11-H13 in the format of Table 6.

**appendix_natural_shift_descriptives.csv.** Length quantiles of the pool, size, median length, risk share and domain
shares of the long and short groups, the number of short risk excerpts per test half and of long risk excerpts per
calibration half over the resplits, and the number of excerpts of at most 40 words (with their risk count). The
secondary analysis on excerpts of at most 40 words needs at least 30 risk excerpts per test half and is not run, because
there are 12 (median over resplits).

**appendix_natural_shift.csv.** c1 and workload per 1,000 per (analysis, test set, method), with the median risk
counts of calibration (n_calibration_1) and test (n_test_1) and the c1 reference band at those counts. Analysis "long
calibration": M0 and M1 calibrated on the clean long excerpts of the calibration half and M3-mix on those excerpts in
their M3-mix condition, with (length bucket, class) groups, tested on the short clean excerpts of the test half (H11,
H13), on its long clean excerpts and on its long excerpts truncated to head_50 and tail_50. Analysis "cross-fitted
prediction" (H12): within the calibration half the long excerpts are split at random by post into halves A and B, and
each is cut to head_50 or tail_50 with probability 1/2 (both draws once per resplit, shared by all models). M0 or M1
fitted on clean A predicts c1 as the share of B's risk excerpts, in their truncated versions, whose risk label is in
the set, and the two folds are averaged after swapping A and B. Analysis "long calibration, domain-reweighted"
(sensitivity): weighted split conformal prediction with the long calibration excerpts weighted by domain so that their
domain shares match those of the short test excerpts (label-free), for M0, M1 and M3-mix, with domain_weight_ess the
effective sample size (sum w)^2 / sum w^2 of the calibration weights. Analysis "standard calibration" (secondary):
calibration on the whole clean calibration half (M3-mix on its M3-mix assignment), tested on the short clean
excerpts of the test half, which gives coverage conditional on length. Its "all clean" rows equal the main results and
serve as a check.

**appendix_natural_shift_differences.csv.** Paired differences per resplit (within seed, then averaged over seeds): c1
of M1 minus M0 on short excerpts; for each method, short-excerpt c1 minus c1 on the long test excerpts (clean, head_50,
tail_50), all with long calibration; the observed band mass c1(M0) - c1(M1) on short excerpts and the predicted band
mass from the cross-fitted prediction; and the prediction error, predicted minus observed c1 of M1.
