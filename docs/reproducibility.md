# Reproducibility and Provenance Notes

## Scope

The repository preserves the experiment runners, scenario generator, four final 50-scenario JSON outputs, the analysis scripts, archived aggregate and revision results, the manual semantic-audit record, and the figures from the four-model comparison.

The saved outputs support offline reproduction of the reported statistical analyses. Rerunning third-party model inference can require current credentials, billing, provider access, and available endpoints. No bit-for-bit reproducibility claim is made for external model services.

## Archived Data Reduction

The four source 50-scenario JSONL datasets were byte-identical. One canonical copy is retained as data/scenarios_50.jsonl. Each final run JSON retains per-scenario, per-variant response records and A/B distributions, plus MCV and decision-flip rate. The actual prompt text is built by src/scenarios.py from the saved scenario fields and wrapper definitions.

The archived run JSON metadata uses a repository-relative dataset path. The Gemini run copy omits its provider project identifier and project-specific endpoint. These metadata-only privacy edits do not change scenario records, probability distributions, per-scenario metrics, or aggregate results.

The final comparison directory supplies the four-model aggregate and theme CSVs. The final revision directory supplies model summaries, paired tests, the revision summary, and generated case studies. The audit directory holds one canonical copy of the identical sampled prompts and the recorded author-audit result.

The historical comparison figures `figures/mcv_boxplot.png` and `figures/theme_mcv.png` are preserved byte-for-byte. The README uses `figures/readme_mcv_boxplot.png` and `figures/readme_theme_mcv.png` as presentation renderings generated from the same saved final run data with the existing comparison plotting logic. Their shorter display labels and adjusted layout are visual presentation changes only; the plotted scenario and theme values, theme ordering, and logarithmic axes are unchanged. The comparison and theme summary CSVs match the archived files exactly. Re-rendering with the installed Matplotlib stack may produce different PNG bytes and pixels, which is a rendering difference rather than a statistical difference.

## Exclusions

The public copy excludes cached bytecode and the 1-, 2-, 3-, and 10-scenario probes or partial runs. It also excludes earlier two- and three-model comparison/revision directories, which are superseded by the retained four-model outputs. The two xAI per-run CSV exports are omitted because they contain derived scenario metrics available from the retained JSON files with analyze_results.py. No model output or statistical value was regenerated to create the archive.

The original source directory was left untouched when this public copy was prepared.
