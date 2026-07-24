# The Silent Tax: How Typos Reshape Reasoning in Thinking LLMs

**NLP Course 2025/2026 — Project Proposal**

| Name | ID | Email |
|------|----|-------|
| Shiran Hamami | 207487026 | shiranhamami@mail.tau.ac.il |
| Dolev Abudi | 323096099 | dolevabudi@mail.tau.ac.il |
| Ron Shtricker | 209524552 | ronshtricker@mail.tau.ac.il |
| Ido Azoulay | 212710958 | idoazoulay1@mail.tau.ac.il |

## Background and Motivation

Real users make typing mistakes all the time. They swap letters, drop a letter, or hit a nearby key. We know that typos lower the accuracy of normal language models, mostly because the tokenizer breaks a misspelled word into strange pieces it does not recognize. We chose this because people now trust the new reasoning models for math and science, and a typo that silently changes the answer is a hidden and important failure. As frequent users of these models, we noticed our own typos sometimes lead to surprising failures, which motivated us to investigate how systematic this problem really is.

## Research Question

How do typographical errors affect reasoning models across multiple dimensions: accuracy, reasoning length, and internal error-handling behavior? Specifically, do real-word typos lead to silent, confident failures while non-word typos trigger visible self-correction, and can lightweight mitigations recover the lost performance?

## What Is Already Known

Typos are known to lower the accuracy of LLMs (Gan et al., 2024; Zhao et al., 2025), and that a major cause is the tokenizer splitting misspelled words into odd sub-tokens the model does not recognize (Chai et al., 2024). However, this prior work reports only the final accuracy of standard models. It does not study the newer reasoning models, and it does not examine the chain-of-thought or how the model reasons internally. What is still missing is an understanding of how typos affect the reasoning process itself, and how this in turn relates to the final output.

## What We Will Do (Method)

1. **Make typo versions (automatically).** We will create a script that automatically generates typos using 4 techniques: adjacent-key substitution (math → nath), swapping two letters (form → fomr), deletion of a letter (triangle → triagle), and insertion of a letter (solve → solvve). In this way we will produce 2 types of typo: a **non-word typo** (the result is not a real word, so it breaks the tokenizer, e.g. triangle → trianlge) and a **real-word typo** (the result is a valid but different word, e.g. sum → sun, ten → tan), which we will filter so a human can still understand the question. Typos will go on words only. Numbers and math will stay correct.

2. **Run the model on clean and typo questions.** We will run a reasoning model on each question in two forms, the clean version and the typo version, and compare them. For each question we will run it several times. We will record whether the answer flipped (right to wrong or wrong to right) per typo type, and how much longer the model reasoned (typo tokens divided by clean tokens). We will also look at two behaviors inside the reasoning trace. The first is **self-doubt**, meaning how much the model second-guesses itself while reasoning, shown by words like "wait", "hmm", "actually", and "let me reconsider". The second is **repair behavior**, meaning how the model handled the typo, for example silently reading through it and solving correctly, explicitly noticing and fixing it, or misreading it as a different word. We expect non-word typos to trigger visible reactions in the reasoning (costly but sometimes protective), and real-word typos to pass unnoticed and cause confident wrong answers (cheap but dangerous).

3. **Test simple fixes.** We will test three fixes: asking the model to first rewrite the question without typos then solve it, running an external spell-checker before the model sees the text, and just telling the model "the text may contain typos". We check how much accuracy each fix brings back.

## How We Will Measure Success

For each question we compare the typo version to the clean version and measure four things:

- **Accuracy and flips:** the drop in accuracy and the flip rate (right to wrong and wrong to right), reported per typo technique and type.
- **Reasoning length:** the ratio of typo tokens to clean tokens, reported per typo technique and type.
- **Self-doubt:** (two ways) the frequency of doubt markers ("wait", "actually", "let me reconsider") and a rating from a separate LLM judge with a fixed prompt.
- **Repair behavior:** we classify each trace into behavioral categories using an LLM judge with strict instructions.

The baseline is the clean input (the best case). We also confirm on a small sample that a human can still answer the typo questions. We run each question several times and report averages, so the results are not based on luck.

## Requirements

All experiments are inference only, no training. We will use the TAU Slurm cluster to run **DeepSeek-R1-Distill-Qwen-7B**, which is open-weight and free on HuggingFace with no API or personal account needed. Our datasets are **MATH-500** (with difficulty labels L1 to L5), the **GSM8K** test set, and **GPQA Diamond** (graduate-level science questions), all publicly available on HuggingFace.

## Key References

- Gan et al. (2024). *Reasoning Robustness of LLMs to Adversarial Typographical Errors.* EMNLP 2024.
- Chai et al. (2024). *Tokenization Falling Short: On Subword Robustness in LLMs.* Findings of EMNLP 2024.
- Zhao et al. (2025). *Evaluating Robustness of LLMs Against Multilingual Typographical Errors.* arXiv:2510.09536.
