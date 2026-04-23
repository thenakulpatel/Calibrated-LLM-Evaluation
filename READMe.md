# Calibrated LLM Evaluation with Uncertainty Quantification

A research-oriented framework for **LLM-based evaluation with calibrated uncertainty**, combining **BERT embeddings, multi-run LLM scoring, neural interval prediction (TubeNet), and conformal prediction**.

---

##  Overview

Large Language Models (LLMs) are widely used as evaluators (“LLM-as-a-Judge”) for tasks like summarization. However, they typically produce **single scalar scores**, which fail to capture **uncertainty**.

This project introduces a **hybrid evaluation framework** that predicts **confidence intervals instead of point scores**, enabling **reliable and calibrated evaluation**.

---

## Key Idea

Instead of asking:

> *“What is the score?”*

We ask:

> *“What range is the score likely to lie in, with high confidence?”*

---

## System Pipeline

```
Article + Summary
        ↓
BERT Embeddings (768-dim)
        ↓
LLM Scoring (multi-run)
        ↓
Uncertainty Features (q05, q95, spread)
        ↓
TubeNet (interval prediction)
        ↓
Conformal Calibration
        ↓
Final Prediction Interval (≈90% coverage)
```

---

## Methodology

### Feature Engineering

* **BERT embeddings** (SentenceTransformers)
* **LLM-based features** (5 runs per sample)

  * 5th percentile (q05)
  * 95th percentile (q95)
  * Midpoint
  * Spread

### Model: TubeNet

* Neural network predicting:

  * Lower bound (μ₁)
  * Upper bound (μ₂)
* Uses:

  ```
  μ₁ = mid - half
  μ₂ = mid + half
  ```

### Loss Function: Tube Loss

Balances:

* Coverage
* Interval width

### 🔹 Conformal Prediction

Final interval:

```
[μ₁ - q̂ , μ₂ + q̂]
```

Ensures **~90% coverage guarantee**

---

## Results (Key Observations)

* ✅ **~90% calibrated coverage achieved**
* 📉 **Up to ~80% reduction in interval width vs BERT-only baseline**
* 📉 **~2× lower error vs raw LLM scores**
* ⚖️ Trade-off observed:

  * Narrow intervals → lower coverage
  * Wider intervals → higher reliability

---

##  Core Insights

* LLM outputs provide **strong semantic signal**
* However, **uncertainty is NOT driven by LLM variability**
* Final uncertainty is dominated by:

  * Model generalization
  * Conformal calibration

> 🔥 **Key Finding:**
> *LLMs improve prediction accuracy, but uncertainty is governed by the model and calibration.*

---

##  Challenges

* LLM inconsistency across runs
* Cache bias (when using stored scores)
* Feature scaling sensitivity
* Overfitting (q̂ → 0 in some cases)
* Coverage vs width trade-off

---

## Dataset

* SummEval-style dataset
* Dimensions:

  * Coherence
  * Consistency
  * Fluency
  * Relevance

---

##  Tech Stack

* **Python**
* **PyTorch**
* **SentenceTransformers (BERT embeddings)**
* **Groq API (LLM inference: Gemma, Qwen)**
* **NumPy / Scikit-learn**
* **Matplotlib (visualization)**

---

## Outputs

* Prediction intervals per dimension
* Coverage vs width plots
* MAE comparison (LLM vs model)
* Diagnostic visualizations

---

##  Key Contributions

* Introduced **uncertainty-aware LLM evaluation**
* Combined **LLM features + neural interval prediction**
* Applied **conformal prediction for calibrated guarantees**
* Demonstrated that:

  > **Uncertainty ≠ LLM variability**

---

##  Future Work

* Multi-LLM disagreement modeling
* Better calibration strategies
* Adaptive interval learning
* Larger datasets for robustness

---

##  Final Takeaway

> Instead of trusting a single LLM score,
> we should trust a **calibrated range with guarantees**.


