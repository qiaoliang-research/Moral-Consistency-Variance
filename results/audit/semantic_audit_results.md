# Semantic Audit Results

Audit set: `semantic_audit_sample.md`  
Reviewer: author audit  
Sample size: 20 random baseline/perturbation prompt pairs  
Judgement criterion: whether the perturbation preserved the scenario facts and the two action options while changing only the surrounding instruction wrapper

Result:

- Semantically equivalent at the scenario-content level: 20 / 20
- Not equivalent at the scenario-content level: 0 / 20

Interpretive note:

- In all sampled cases, the scenario block, action options, and question remained unchanged.
- The wrapper prompts did introduce different discourse cues such as `consistency check`, `careful evaluation`, or `stylistic normalization`.
- The audit therefore supports factual and option-level equivalence, but it does not establish full pragmatic identity.
