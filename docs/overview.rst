Project Overview
================

Fake News Detection classifies political statements into one of six truthfulness
categories using the `LIAR2 dataset <https://www.cs.ucsb.edu/~william/data/liar_plus_dataset.zip>`_
sourced from PolitiFact.

Architecture
------------

The project uses **Tinker LoRA fine-tuning** applied to
``nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`` to produce a prompt-based classifier.
A local BERT/RoBERTa fallback is also supported for offline inference.

Experiment versions (``v1``–``v4``) are tracked in :mod:`config.TINKER_EXPERIMENTS`.

Directory Layout
----------------

.. code-block:: text

   FakeNewsDetection/
   ├── src/
   │   ├── config.py          # Constants, labels, and experiment configs
   │   ├── data_processor.py  # Data loading, cleaning, and encoding
   │   ├── models.py          # TinkerClassifier and ClassBalancer
   │   ├── training_utils.py  # Trainers, splitter, and metrics
   │   └── predict.py         # CLI prediction script
   ├── notebooks/
   │   ├── 01_data_loading_cleaning_encoding.ipynb
   │   ├── 02_model_training.ipynb
   │   └── 03_training_execution.ipynb
   ├── data/                  # LIAR2 train / valid / test TSVs
   └── artifacts/             # Saved weights, metrics, and plots

Quick Start
-----------

**Predict a single statement (Tinker model):**

.. code-block:: bash

   python src/predict.py --text "The unemployment rate is 3 percent"

**Predict with per-label probabilities:**

.. code-block:: bash

   python src/predict.py --text "The unemployment rate is 3 percent" --proba

**Predict from a file (one statement per line):**

.. code-block:: bash

   python src/predict.py --file data/statements.txt

**Use a local BERT checkpoint:**

.. code-block:: bash

   python src/predict.py --mode local --model artifacts/bert_model.pt --type bert --text "..."

Truthfulness Labels
-------------------

+---+----------------+-------------------------------------+
| # | Label          | Meaning                             |
+===+================+=====================================+
| 0 | barely-true    | Mostly inaccurate, misleading       |
+---+----------------+-------------------------------------+
| 1 | false          | Factually incorrect                 |
+---+----------------+-------------------------------------+
| 2 | half-true      | Partially accurate                  |
+---+----------------+-------------------------------------+
| 3 | mostly-true    | Largely accurate with minor caveats |
+---+----------------+-------------------------------------+
| 4 | pants-fire     | Egregiously false                   |
+---+----------------+-------------------------------------+
| 5 | true           | Fully accurate                      |
+---+----------------+-------------------------------------+
