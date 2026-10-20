# Figure captions

Conformal quantities use alpha = 0.10 unless stated otherwise. Each is the median over 1,000 grouped resplits of
calibration + test, and bars or bands show the 2.5-97.5% range over resplits. For encoders, the metric is first averaged
over the 5 seeds within each resplit, and the median and interval are then taken over resplits. The tables report the
between-seed spread. The LLMs are zero-shot with prompt P1 and one run each. The grey band is the 95% beta-binomial
reference band of the coverage that the guarantee implies at the calibration and test sizes of that population. "Risk"
denotes the positive class of both datasets, which is stress on Dreaddit and suicidal ideation on SDCNL. Figures that
show Dreaddit alone say "stress" and "not stress". Method labels: M0 (marginal), M1 (class-conditional), M2 (domain ×
class), M3 (matched degradation), M3-mix (degradation mixture), M5 (risk-only, one-sided), M0-LS (marginal, label-shift
weighted), B1 (validation threshold), B2 (temperature scaling + abstention), B3 (seed ensemble).

**Figure 1.** Triage scheme. The prediction set decides the action ({not risk} -> dismiss, {risk} -> flag, full or
empty set -> refer to a human). Every per-class metric is reweighted to a risk prevalence pi as pi * metric(risk) +
(1 - pi) * metric(not risk).

**Figure 2.** Coverage, not-risk coverage c0 and risk-class coverage c1 of each model on clean Dreaddit under M0
(marginal) and M1 (class-conditional), shown against each metric's reference band. The appendix repeats the figure for
alpha = 0.05 and for SDCNL.

**Figure 3.** Coverage (top) and workload per 1,000 screened (bottom) as functions of the risk prevalence pi. M0
(marginal) and M1 (class-conditional) are calibrated at the original prevalence. M0-LS (marginal, label-shift weighted,
Podkopaev & Ramdas, 2021) knows pi. An appendix version covers SDCNL with M0 and M1 only, since M0-LS was not run there.

**Figure 4.** DistilBERT risk-class coverage c1 (top) and workload per 1,000 at pi = 0.10 (bottom) against degradation
strength (head / tail truncation, word deletion, letter typos). Methods are M0 (marginal) and M1 (class-conditional),
both calibrated on clean texts, plus M3 (matched degradation) and M3-mix (degradation mixture, with length bucket x
class groups). The grey band is the c1 reference band. Appendix versions show the other models, alpha = 0.05 and SDCNL.

**Figure 5.** Risk-class coverage c1 against workload per 1,000 screened at pi = 0.10. The left panel shows clean texts
with the dense region enlarged, and the right panel shows texts truncated to their first 25% of words. Each line joins
the operating points of one model. Markers identify the method among M0 (marginal), M1 (class-conditional), M3-mix
(degradation mixture), M5 (risk-only, one-sided) and B1 (validation threshold). M5 coincides with M1 because it flags
exactly the cases whose M1 set contains the risk class, which makes c1 and workload equal. The dotted line marks the
nominal 0.90.

**Mechanism figure (main text).** DistilBERT p(stress) of stress and not stress posts on clean texts and after
truncation to the first or last 25% of words, pooled over calibration + test with shares averaged over the 5 seeds. The
vertical lines mark the p(stress) at or above which each method puts the stress class in the prediction set. These
cut-offs are averaged over seeds per resplit, and the figure shows their median over 1,000 resplits. M0 (marginal) has
a single threshold and M1 (class-conditional) contributes its stress-class threshold. M3-mix (degradation mixture)
thresholds depend on the text-length bin, so each panel shows the threshold of the bin that contains the condition's
median length (> 80 words for clean, <= 25 words for head/tail 25%). The number at the top of each line is its
cut-off. The appendix shows the other six models.

**Schematic figure (conceptual).** How degradation moves at-risk posts into the band between the two cut-offs. The two
curves are illustrative beta-shaped densities of the model score p(risk) of at-risk posts for clean and degraded text,
not estimates from data. The vertical lines are DistilBERT's median M0 (marginal) and M1 (class-conditional) risk
cut-offs from the mechanism figure. Degraded at-risk posts that land in the shaded band between them are still caught
by M0 (their set contains risk) but missed by M1.

**Exploratory figure (post hoc, mechanism-derived).** Each point is one run, covering every encoder seed and the single
runs of TF-IDF and LLM S-tok P1. Circles are Dreaddit runs and triangles SDCNL runs, with hollow markers for LLMs. The x
axis gives the slack, defined as the clean-data c1 under M0 (median over resplits) minus 0.90. On the y axis is the
paired difference c1(M1) - c1(M0) at head_25, as a median with bars for the 2.5-97.5% range over resplits. The analysis
was defined after the main results and is descriptive only.

**Appendix figures.** Leave-one-domain-out coverage of the held-out domain under M0 and M1, from 1,000 calibration
bootstraps (H4). Coverage by annotator-agreement bucket for DistilBERT over the first 200 resplits. For DistilBERT,
subreddit coverage is plotted against the subreddit's stress share together with the reweighting line. Sentence
occlusion shows the mean change in p(stress) on the test split when the first, a middle or the last sentence is
removed, with +/- SD over seeds. Reliability diagrams use 15 equal-width bins on the fixed test split, with seeds
averaged per bin.
