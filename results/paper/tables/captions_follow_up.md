# Captions of the H7-H10 follow-up tables and figure

Descriptive follow-ups and sensitivity checks without criteria; the H7-H10 verdicts stay as fixed. Same data, models,
alpha = 0.10, 1,000 resplits and primary aggregation as the H7-H10 tables (captions_h7_h10.md). Encoders are
DistilBERT, RoBERTa and MentalRoBERTa; zero-shot models are the three LLMs with S-tok P1.

**appendix_zero_shot_ranking.csv.** Paired differences per resplit, zero-shot model minus encoder (the encoder averaged
over its seeds within the resplit), under head_25, tail_25, del_30 and typo_10, as the median and 2.5-97.5% range over
resplits. Metric oracle_workload is the H9 oracle workload per 1,000 at c1 = 0.90 (lower is cheaper, see
appendix_h9_oracle_cost.csv); metric auroc is the test-half AUROC of the risk log-odds (higher ranks better). verdict
says which side is better when the interval excludes 0.

**appendix_auroc_by_condition.csv.** Test-half AUROC of all 7 models in all 11 conditions (metric auroc) and the paired
difference to the best encoder of that condition, the one with the highest median AUROC (metric
auroc_minus_best_encoder, column best_encoder).

**appendix_h7_sensitivity.csv.** H7 with exact workload matching on the test half's clean workload by linear
interpolation along an alpha grid. For the trained models (moved_method M1), M1's c1 and workload are interpolated over
the grid 0.02-0.20 to M0's clean workload at alpha = 0.10. For the zero-shot models (moved_method M0), M0's alpha is
raised over the grid 0.10-0.99 until its clean workload equals M1's at alpha = 0.10. alpha_star is the interpolated
alpha (the crossing nearest to 0.10), c1_difference_M1_minus_M0 the paired difference and workload_gap_M1_minus_M0 the
remaining workload gap (0 on clean by construction). Resplits without a crossing are left out. Rows with condition
"all" give their share (metric unmatched), the target clean workload and the lowest and highest clean workload the moved
method reaches on its grid.

**appendix_h10_clipped_weights.csv.** H10 variant a, class-conditional (WCP-class), with calibration and test weights
clipped at 11 = 1/pi_k, the bound on the true target / mixture density ratio when the target condition has weight 1/11
in the mixture, next to the unclipped results (column weights). Metrics as in appendix_h10_weighted_conformal.csv. Rows
with model "all" give, per condition, the shares of calibration and test weights above 11 (the weights do not depend
on the model).

**figure5_risk_coverage_vs_workload_curves (figure and CSV).** Figure 5 with thin c1-vs-workload curves over the alpha
grid 0.02-0.20 for M0 and M1 of every model (colour and line style of the model), from the H7 curves. Markers are the
alpha = 0.10 operating points of Figure 5. The CSV holds the curve points; the markers are in figure5_points.csv.
