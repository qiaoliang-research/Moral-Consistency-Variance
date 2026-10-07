# Revision Summary

## Model-Level Results

### grok-4-fast-non-reasoning
- Scenario count: 50
- Mean MCV: 0.002220 (SD 0.011254; 95% CI [0.000209, 0.005817])
- Median MCV: 0.000038
- Mean JSD: 0.000690 (SD 0.003504; 95% CI [0.000066, 0.001761])
- Mean flip rate: 0.000000

### grok-4-1-fast-non-reasoning
- Scenario count: 50
- Mean MCV: 0.001376 (SD 0.009295; 95% CI [0.000031, 0.004035])
- Median MCV: 0.000000
- Mean JSD: 0.000455 (SD 0.003069; 95% CI [0.000011, 0.001329])
- Mean flip rate: 0.000000

### HuggingFaceTB/SmolLM2-360M-Instruct
- Scenario count: 50
- Mean MCV: 0.005886 (SD 0.002166; 95% CI [0.005281, 0.006457])
- Median MCV: 0.005617
- Mean JSD: 0.001467 (SD 0.000539; 95% CI [0.001326, 0.001618])
- Mean flip rate: 0.292000

### gemini-2.5-flash
- Scenario count: 50
- Mean MCV: 0.022724 (SD 0.062721; 95% CI [0.007520, 0.042246])
- Median MCV: 0.000586
- Mean JSD: 0.005190 (SD 0.014430; 95% CI [0.001762, 0.009521])
- Mean flip rate: 0.012000

## Pairwise Tests

- mcv: grok-4-fast-non-reasoning vs grok-4-1-fast-non-reasoning | Δ=0.000844 | 95% CI [0.000156, 0.001826] | Wilcoxon p=1.61833e-08
- jsd: grok-4-fast-non-reasoning vs grok-4-1-fast-non-reasoning | Δ=0.000235 | 95% CI [0.000046, 0.000522] | Wilcoxon p=2.24518e-08
- mcv: grok-4-fast-non-reasoning vs HuggingFaceTB/SmolLM2-360M-Instruct | Δ=-0.003666 | 95% CI [-0.005861, -0.000085] | Wilcoxon p=8.22989e-09
- jsd: grok-4-fast-non-reasoning vs HuggingFaceTB/SmolLM2-360M-Instruct | Δ=-0.000777 | 95% CI [-0.001444, 0.000336] | Wilcoxon p=8.22989e-09
- mcv: grok-4-fast-non-reasoning vs gemini-2.5-flash | Δ=-0.020504 | 95% CI [-0.039011, -0.006113] | Wilcoxon p=4.85715e-09
- jsd: grok-4-fast-non-reasoning vs gemini-2.5-flash | Δ=-0.004499 | 95% CI [-0.008688, -0.001317] | Wilcoxon p=1.50792e-07
- mcv: grok-4-1-fast-non-reasoning vs HuggingFaceTB/SmolLM2-360M-Instruct | Δ=-0.004510 | 95% CI [-0.006220, -0.001637] | Wilcoxon p=5.61027e-11
- jsd: grok-4-1-fast-non-reasoning vs HuggingFaceTB/SmolLM2-360M-Instruct | Δ=-0.001012 | 95% CI [-0.001546, -0.000082] | Wilcoxon p=5.61027e-11
- mcv: grok-4-1-fast-non-reasoning vs gemini-2.5-flash | Δ=-0.021348 | 95% CI [-0.040277, -0.006581] | Wilcoxon p=6.59028e-13
- jsd: grok-4-1-fast-non-reasoning vs gemini-2.5-flash | Δ=-0.004734 | 95% CI [-0.009083, -0.001408] | Wilcoxon p=9.52127e-13
- mcv: HuggingFaceTB/SmolLM2-360M-Instruct vs gemini-2.5-flash | Δ=-0.016839 | 95% CI [-0.036109, -0.001589] | Wilcoxon p=0.0136869
- jsd: HuggingFaceTB/SmolLM2-360M-Instruct vs gemini-2.5-flash | Δ=-0.003722 | 95% CI [-0.008121, -0.000260] | Wilcoxon p=0.0108796