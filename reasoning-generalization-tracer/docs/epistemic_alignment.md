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

## Ten-Case Abstention Reward Scheme

The reward combines knowledge, honesty, and abstention calibration weights (`abstention.reward_weights`).
Thought rewards are never negative and only activate when epistemic alignment is present.

1. **Correct, high-conf, aligned** → `+K_high + H`
2. **Correct, high-conf, unaligned** → `+K_low`
3. **Correct, low-conf, aligned (Timid Expert)** → `+K_low + H`
4. **Correct, low-conf, unaligned (Lucky Guess)** → `+0.5*K_low`
5. **Wrong, high-conf, aligned (Confident but wrong)** → `-K_high + H`
6. **Wrong, high-conf, unaligned** → `-K_high`
7. **Wrong/unknown, low-conf** → `-K_low (+H if aligned)`
8. **Honest uncertainty (aligned IDK, calibrated)** → `+A + H`
9. **Miscalibrated honest IDK (high confidence but abstained)** → `+A + H - K_miscal`
10. **Lazy IDK (unaligned abstention)** → `-A - K_miscal`

Confidence bonuses only apply when reasoning is epistemically grounded; high-confidence but
unaligned correct answers fall back to the low-confidence knowledge weight. Logs include
`s_match`, `s_epistemic`, `thought_alignment`, and `reward_case` for downstream analysis.
