# AI assistance and evaluation provenance

OpenAI Codex helped me complete this project.

## Evaluation protocol and author clarification

The author clarified on 20 October 2026 that models were selected using validation
performance. Test scores were inspected after model selection to compare the
selected model with previous models. According to the author, those test
comparisons did not determine model selection.

Weights and normalization statistics were fitted on the supplied training bank.
Validation controlled early stopping and configuration selection. The maintained
training entry points load training and validation references only. The reported
test result is reproduced from the frozen checkpoint using the official scorer;
the prediction function does not access reference answers.

The comparison summaries and search-cost record identify validation as the
model-selection split. The author's account and the metadata change history
are documented in `results/evaluation_provenance.json`. Original experiment
records are preserved in `results/history/original_records.zip`.

## Reuse

No external training dataset, additional pretrained weights or generation API
was used. The official TabM package was initialized randomly. The original
MIT-licensed MoB algorithm is retained with its license. LEVER provided a
methodological reference for a validation ablation, not a reproduced system.
Full attribution is in [references.md](references.md).

The official data, numeric parser and scoring tools remain unchanged.
