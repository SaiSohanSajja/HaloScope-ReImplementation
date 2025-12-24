# HaloScope-style Hallucination Detection (Reimplementation)

End-to-end reimplementation of a HaloScope-inspired pipeline for detecting hallucinated LLM answers
using embedding geometry + weak supervision.

## Pipeline
1. Download TyDiQA (English)
2. Generate answers using FLAN-T5-small
3. Extract encoder embeddings
4. SVD subspace membership scoring (HaloScope core idea)
5. Pseudo-label unlabeled set from membership scores
6. Train MLP classifier
7. Evaluate with soft-match correctness labels

## Results (Soft-match labels)
Validation: Accuracy = 0.61, AUROC = 0.6627  
Test: Accuracy = 0.58, AUROC = 0.5778

## Notes
- Labels are heuristic (soft-match), not human annotations.
- Best interpreted as a ranking signal (AUROC), not a perfect classifier.
