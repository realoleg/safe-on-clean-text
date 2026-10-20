# Captions of the second set of H7-H10 follow-up tables

Descriptive, without criteria; the H7-H10 verdicts stay as fixed. Same data, models, alpha = 0.10, 1,000 resplits and
primary aggregation as captions_follow_up.md.

**appendix_h7_sensitivity_calibration_matched.csv.** As appendix_h7_sensitivity.csv for the trained models (TF-IDF,
DistilBERT, RoBERTa, MentalRoBERTa; moved_method M1), but the workloads are matched on the calibration half's
in-sample clean data, as in H7: alpha_star is where M1's in-sample clean workload, interpolated along the grid
0.02-0.20, equals M0's at alpha = 0.10. c1 and workload on the test half are interpolated at that alpha, so the clean
workload gap on the test half is no longer 0 by construction. Rows with condition "all" give the share of unmatched
resplits, the calibration-half target workload and the range M1 reaches on the grid.

**appendix_m3_cost_ratios.csv.** Workload per 1,000 at pi = 0.10 of M3 (calibration in the test condition) divided by
M1's clean workload (metric workload_ratio_M3_over_M1_clean) and by M1's workload in the same condition (metric
workload_ratio_M3_over_M1_same_condition), both methods at alpha = 0.10. Ratios of the seed-averaged workloads within
each resplit, median and 2.5-97.5% over resplits. On clean texts M3 equals M1, so both ratios are 1.
