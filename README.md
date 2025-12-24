# HaloScope-style Hallucination Detection (Reimplementation) 
# - Prof. Sharon Li's Thesis on HaloScope (University of Wisconsin - Madison) 

This repository contains an end-to-end reimplementation of a HaloScope-inspired
pipeline for detecting hallucinated LLM answers using embedding geometry and
weak supervision.

The project investigates whether internal representation geometry of large
language models contains signals correlated with hallucination behavior, even
in the absence of human-labeled supervision.

---

## Pipeline

1. Download TyDiQA (English subset)
2. Generate answers using FLAN-T5-small
3. Extract encoder hidden-state embeddings
4. Perform SVD-based subspace membership scoring (HaloScope core idea)
5. Create pseudo-labels for unlabeled data from membership scores
6. Train a lightweight MLP classifier
7. Evaluate using soft-match correctness labels

---

## Results (Soft-Match Labels)

| Split | Accuracy | AUROC |
|------|---------|-------|
| Validation | 0.61 | 0.6627 |
| Test | 0.58 | 0.5778 |

These results demonstrate non-random hallucination ranking performance under
weak supervision. As expected for geometry-based approaches without human
annotation, the signal is modest but consistent, and best interpreted through
AUROC rather than raw accuracy.

---

## Relation to Prof. Sharon Li’s Research

This project is closely aligned with the research direction of **Prof. Sharon Li**
(University of Wisconsin–Madison), whose work focuses on:

- Reliability and trustworthiness of machine learning systems  
- Hallucination detection and uncertainty estimation in generative models  
- Representation-level signals for identifying failure modes in LLMs  
- Weakly supervised and unsupervised approaches to model evaluation  

In particular, this reimplementation mirrors key themes from Prof. Li’s work by:

- Studying hallucination behavior **without relying on human-labeled data**
- Using **representation geometry** rather than surface-level heuristics
- Framing hallucination detection as a **ranking problem**, evaluated via AUROC
- Achieving performance levels consistent with prior weakly supervised methods
  reported in the literature (AUROC ≈ 0.55–0.70)

The obtained results (Validation AUROC ≈ 0.66, Test AUROC ≈ 0.58) fall within the
expected range for weakly supervised hallucination detection pipelines using
small-to-medium scale language models, reinforcing published findings that
embedding-space structure contains useful but inherently noisy hallucination
signals.

---

## Notes & Limitations

- Correctness labels are heuristic (soft-match), not human-verified
- Uses a relatively small model (FLAN-T5-small)
- Results are best interpreted as **relative hallucination likelihood**, not
  definitive classification

---

## Key Takeaways

- LLM hidden representations encode weak but meaningful hallucination signals
- Subspace-based methods can outperform random baselines without supervision
- Careful training is required to avoid degenerate solutions (e.g., classifier collapse)

---

## Author

**Sai Sohan Sajja**  
Independent research reimplementation project
