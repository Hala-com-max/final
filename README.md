# Character-Level Adversarial Robustness Evaluation using TextAttack in A4S-Eval

## Overview

This project extends the A4S-Eval framework with character-level adversarial robustness evaluation using the TextAttack library and a DeepWordBug-inspired attack strategy.

The implementation evaluates how small typo-level perturbations affect NLP model predictions while preserving human readability.

The project focuses on:

* adversarial robustness evaluation
* character-level perturbation attacks
* typo-level adversarial examples
* black-box NLP attacks
* robustness metrics for trustworthy AI systems

---

# Project Objectives

The goal of this project is to:

* integrate adversarial text attacks into A4S-Eval
* evaluate NLP robustness under character perturbations
* measure attack success rate and accuracy degradation
* analyze the effect of Levenshtein edit distance constraints
* test robustness using realistic typo-level attacks

---

# Implemented Attack Strategy

The project uses a DeepWordBug-inspired character-level attack.

Supported perturbations:

* character insertion
* character deletion
* character substitution
* character swap

Examples:

| Original | Perturbed |
| -------- | --------- |
| Recent   | Rceent    |
| Intel    | Iuntel    |
| Brokers  | BPokers   |
| Reuters  | Reutes    |

These perturbations remain understandable to humans while successfully misleading the classifier.

---

# Levenshtein Distance

The implementation uses Levenshtein distance to constrain perturbation realism.

Allowed operations:

* insertion
* deletion
* substitution
---

# Experimental Setup

## Dataset

Hugging Face AG News dataset.

Classification labels:

```python
AG_NEWS_LABELS = {
    "0": "WORLD",
    "1": "SPORTS",
    "2": "BUSINESS",
    "3": "SCIENCE_TECHNOLOGY",
}
```

---

## Model

Local inference using Ollama.

Example model:

```bash
ollama pull llama3.2:1b
```

---

## Evaluation Parameters

Experiments were conducted using:

* sample sizes:

  * 100
  * 200

* max_edit_distance:

  * 1
  * 10
  * 100

Metrics measured:

* attack success rate
* original accuracy
* accuracy under attack
* perturbed word percentage
* average query count

---

# Final Experimental Results

## Distance = 1 (200 Samples)

| Metric                |  Value |
| --------------------- | -----: |
| Successful attacks    |     15 |
| Failed attacks        |     17 |
| Skipped attacks       |    168 |
| Original accuracy     |  16.0% |
| Accuracy under attack |   8.5% |
| Attack success rate   | 46.88% |
| Avg perturbed words   |  2.69% |

---

## Distance = 100 (200 Samples)

| Metric                | Value |
| --------------------- | ----: |
| Successful attacks    |    17 |
| Failed attacks        |    17 |
| Skipped attacks       |   166 |
| Original accuracy     | 17.0% |
| Accuracy under attack |  8.5% |
| Attack success rate   | 50.0% |
| Avg perturbed words   |  2.3% |

---

# Key Findings

The experiments demonstrate that:

* minimal typo-level perturbations significantly reduce model robustness
* increasing edit distance does not dramatically improve attack effectiveness
* small orthographic modifications are sufficient to fool the classifier
* NLP systems remain highly sensitive to character-level noise

---

# Engineering Contributions

This project includes:

* TextAttack integration into A4S-Eval
* custom adversarial robustness metric
* Ollama inference wrapper
* CSV metric export
* Hugging Face dataset integration
* Levenshtein distance constraints
* automated pytest evaluation
* real adversarial attack pipeline

---

# Installation

## Clone Repository

```bash
git clone https://github.com/Hala-com-max/final
```

## Navigate into Project

```bash
cd final
cd a4s-eval
```

## Create Virtual Environment

```bash
python -m venv uv
```

## Activate Environment

### Linux / macOS

```bash
source uv/bin/activate
```

### Windows

```bash
uv\Scripts\activate
```

---

# Install Dependencies

```bash
pip install -r requirements.txt
```

Install additional libraries:

```bash
pip install textattack
pip install python-Levenshtein
```

---

# Run Tests

```bash
uv run pytest -s
```

# Repository

GitHub Repository:

https://github.com/Hala-com-max/final

---

# Future Work

Possible future improvements:

* semantic adversarial attacks
* synonym substitution attacks
* paraphrase-based attacks
* multilingual robustness evaluation
* adversarial defense training
* transformer-based semantic perturbations

---

# Author

Halefom Mulu

AI & Cybersecurity
