# Legacy

Pre-repair artifacts kept for the audit trail; nothing here feeds the current
reports or the paper.

- `report_arc_judge_prefix.{pdf,tex}`: ARC report snapshot built BEFORE the
  dropped-stream repair - its accuracy/completion numbers are the contaminated
  pre-fix values. The current report is `../reports/arc/results/report_arc.pdf`.
- `rerun_manifest_arc.json`, `rerun_manifest_gpqa.json`: which question idx per
  config were lost to mid-generation connection drops and re-generated with
  identical prompts (57 on ARC, 9 on GPQA).
