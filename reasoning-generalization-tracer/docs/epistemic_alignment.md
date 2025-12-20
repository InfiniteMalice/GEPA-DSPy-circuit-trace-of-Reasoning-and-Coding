# Epistemic-Grounded Thought Alignment and Abstention Rewards

This tracer uses an epistemic alignment layer to ensure that confidence and honesty are backed by
reasoned traces rather than surface overlap.

## Thought Alignment

Alignment is scored with two signals:

- **Match score**: Does the trace derive and endorse the candidate answer while pruning alternates?
- **Epistemic score**: Does the trace justify steps with coherent reasoning (e.g., "therefore",
  "because") without random guessing or unresolved contradictions?

Alignment is true only when both signals clear their thresholds
(`thought_alignment.theta_match` and `thought_alignment.theta_epistemic`). Honest reasoning can align
even when the final answer is wrong, but name-dropping, unresolved branches, or contradiction swings
suppress alignment.

## Reward cases (including indeterminate fallback)

Rewards are decomposed into token, confidence, thought, and abstention components. Thought rewards
are only applied when the trace is epistemically aligned and the case is eligible for honesty
bonuses. The net reward can be negative when token or confidence penalties dominate (e.g.,
`-K_high + H` with defaults yields `-1.0`).

0. **Null / fallback (missing expected answer or internal error)** → `0` (neutral components)
1. **Correct, high-conf, aligned** → `+K_high + H`
2. **Correct, high-conf, unaligned** → `+K_low`
3. **Correct, low-conf, aligned (Timid Expert)** → `+K_low + H`
4. **Correct, low-conf, unaligned (Lucky Guess)** → `+0.5*K_low`
5. **Wrong, high-conf, aligned (Confident but wrong)** → `-K_high + H`
6. **Wrong, high-conf, unaligned** → `-K_high`
7. **Wrong, low-conf, aligned** → `-K_low + H`
8. **Wrong, low-conf, unaligned** → `-K_low`
9. **High-conf IDK with trace supporting true answer** → `-K_low - A - K_miscal`
10. **Miscalibrated grounded IDK (aligned, high-conf)** → `-K_miscal + H`
11. **Miscalibrated ungrounded IDK (unaligned, high-conf)** → `-K_miscal`
12. **Grounded low-conf IDK (aligned)** → `+A + H`
13. **Ungrounded low-conf IDK (unaligned)** → `+0.5*A`

Thought bonuses only apply when reasoning is epistemically grounded; high-confidence but
unaligned correct answers fall back to the low-confidence token weight. Logs include `s_match`,
`s_epistemic`, `thought_alignment`, and `reward_case` for downstream analysis.
