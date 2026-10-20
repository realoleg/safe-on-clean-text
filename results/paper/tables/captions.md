# Table captions

Conformal tables give medians and 2.5-97.5% quantiles (q025, q975) over 1,000 grouped resplits of calibration + test.
SDCNL resplits are grouped by author. For encoders, each resplit is first averaged over the 5 seeds (primary
aggregation). The columns seed_min and seed_max give the between-seed spread as the min-max of the per-seed medians and
are empty for single-run models. Results are reported for alpha = 0.10 and 0.05 (column alpha). Methods appear as the
codes of the labels in figures/captions.md, and <model>_ensemble rows are B3 (seed ensemble). Metrics: coverage, c0 / c1
(coverage of the not-risk / risk class), workload_per_1000_pi=0.1 (posts not dismissed, that is flagged or referred, per
1,000 screened at risk prevalence 0.10), uncertainty_referrals_per_1000_pi=0.1 (referrals, meaning full or empty sets,
per 1,000 at prevalence 0.10) and coverage_pi=0.1 (coverage reweighted to prevalence 0.10).

**Table 1 (table1_data.csv).** Data per dataset and split. Columns give rows, posts (Dreaddit) or author groups
(SDCNL), risk rows, risk share (stress on Dreaddit, suicidal ideation on SDCNL) and median words. For Dreaddit the table
adds rows and risk share per domain and rows per annotator-agreement bucket.

**Table 2 (table2_classification.csv).** Classification on clean texts of the fixed validation and test splits. Reported
are macro-F1, AUROC, Brier score, NLL and ECE (15 equal-width and 15 equal-mass bins of p(risk)), each as the median and
min-max over seeds (n_seeds). The column probabilities marks raw or temperature-scaled values. Temperature scaling, with
the temperature fitted on validation, is applied to Dreaddit only.

**Table 3 (table3_conformal.csv).** Dreaddit conformal results per model, method and scenario. The four scenarios are
clean texts, clean texts reweighted to risk prevalence 0.10 (pi=0.1), texts truncated to their first 25% of words
(head_25) and "worst in-distribution domain (min of 5; clean; first 200 resplits)". The last one takes, per resplit, the
lowest coverage among the 5 Dreaddit domains (abuse, anxiety, financial, PTSD, social) on that resplit's test half, over
the first 200 resplits. Every model in it is trained and calibrated on all domains, so it is an in-distribution group
check. The leave-one-domain-out analysis (H4, appendix_lodo.csv) differs in that it removes a domain from training and
calibration.

**Table 4 (table4_llms.csv).** Zero-shot LLMs by confidence method and prompt (P1-P3). S-tok uses answer-token
probabilities, S-verb a verbalised probability and S-cons self-consistency over 10 samples. The table gives validation
and test macro-F1 and test AUROC on the clean fixed split, and the mean probability mass on the answer tokens (S-tok,
clean test). It also lists the conformal operating points on clean texts at alpha = 0.10 under M0 and M1, as the median
and 2.5-97.5% range over resplits for the single run of each.

**Table 5 (table5_sdcnl.csv).** SDCNL replication in the format of Table 3, for TF-IDF + LogReg, DistilBERT (5 seeds)
and Qwen3-4B S-tok P1. It covers methods M0, M1, M3, M3-mix, M5 and B1 and has no domain scenario.

**Table 6 (table6_hypotheses.csv).** Criteria of hypotheses H1-H6 and of their SDCNL replications R1-R3 under the
primary aggregation. The table has one row per model-level quantity, per tally of models meeting a criterion and per
verdict. Column value holds the quantity, and margin is its distance to the criterion's threshold, positive when the
criterion is met. The meets column records whether the criterion is met, and seed_spread_min_max_of_medians gives the
between-seed spread. Reference bands are the 95% beta-binomial bands of the coverage the guarantee implies at the median
calibration and test sizes.

**Appendix tables.**
- appendix_table6_sensitivity_pooled.csv lists the Table 6 tallies and verdicts that differ when an encoder's seeds are
  pooled, with quantiles over all seed x resplit values in place of seed-averaged resplits.
- Leave-one-domain-out results (H4) are in appendix_lodo.csv. It gives coverage, c0 and c1 of the held-out domain under
  M0 and M1 as the median and 2.5-97.5% over 1,000 post-level bootstraps of the four-domain calibration (for DistilBERT,
  3 seeds averaged per bootstrap), with the reference band (band_low, band_high) at that domain's sizes.
- appendix_agreement_buckets.csv reports n, coverage, c1 and workload by annotator-agreement bucket (clean, M0 and M1,
  first 200 resplits).
- The AUROC for a non-unanimous annotation, with a full M0 / M1 prediction set or 1 - max p as the predictor (H6), is in
  appendix_agreement_auroc.csv.
- appendix_agreement_regression.csv holds a logistic regression of a referral (full or empty set) on the agreement
  bucket (reference level unanimous, 1.00), the domain (reference level abuse) and log word count, with coefficients
  over resplits.
- In appendix_unanimous.csv, calibration and test are restricted to unanimously annotated rows (subset unanimous). The
  main metrics appear next to those for all rows, with the remaining sizes.
- appendix_confident_miss_liwc.csv compares stress posts whose set is exactly {not stress} (confident misses) with
  stress posts whose set contains stress, by counts and standardised mean differences (smd_*) of LIWC features.
- Official-test rows without an agreement value are set against rows with one in appendix_missing_agreement.csv.
- appendix_subreddits.csv gives size, stress share and coverage per subreddit (DistilBERT).
- The reliability-diagram bins (15 equal-width bins of p(risk), clean fixed test split, seeds averaged per bin) are
  stored in appendix_reliability_bins.csv.
- appendix_jitter.csv sets the tie-jitter variant, with U(0, 1e-9) added to the scores, beside the main result for the
  discrete-score LLM runs (S-verb, S-cons).
- For each run, appendix_official_test_permutation.csv holds the one-sided p-value of the official-test c0 under M1
  (clean, alpha = 0.10) against the 1,000 resplits.
- appendix_llm_mass.csv summarises LLM answers per split and condition, with unparsed answers, mean answer-token mass,
  the shares of rows with mass below 0.9 and 0.5, and rows with a missing answer-token variant.
- The post hoc slack analysis (figures/captions.md, exploratory figure) and its Spearman correlations are in
  exploratory_slack_points.csv and exploratory_slack_spearman.csv.
- appendix_band_decomposition.csv is exploratory and post hoc. It decomposes the risk-class coverage gap
  c1(M0) - c1(M1) at alpha = 0.10 (resplit protocol, 1,000 resplits, all 11 conditions) for the Table 3 and Table 5
  runs other than the seed ensembles. Both methods are calibrated on clean data and put the risk label in the set when
  its score is at most the threshold, q0 under M0 and q1 under M1. The gap therefore equals the signed share of risk
  test posts whose score lies between q1 and q0 (band_mass), which is negative when the M1 cut-off is looser than the
  M0 cut-off. band_mass_change_vs_clean is the paired difference band mass(condition) - band mass(clean) within each
  resplit. The clean rows also give q0 and q1 and the same thresholds as p(risk) cut-offs, the risk label being in the
  set when p(risk) is at least the cut-off. Rows with seed = seed-averaged use the primary aggregation, and each
  multi-seed model also has one row group per seed. ratio_of_medians is band mass(condition) / band mass(clean) computed
  from the aggregated medians, not per resplit, and is empty when the clean median is below 0.005.
  identity_max_abs_difference is the largest absolute difference between band_mass and the stored c1(M0) - c1(M1) over
  the seeds and resplits of the row group.
