Fake News Detection -- Project Documentation
=============================================

A 6-class fake news classification system on the **LIAR2 dataset** using
LoRA fine-tuning of NVIDIA Nemotron-3-Nano-30B via the Tinker API.

.. list-table:: Key Results
   :header-rows: 1
   :widths: 15 15 10 10 10 10

   * - Version
     - LoRA Rank
     - LR
     - Epochs
     - Accuracy
     - Macro F1
   * - V1
     - 16
     - 1e-4
     - 3
     - 71.9%
     - 0.720
   * - **V3 (best)**
     - 8
     - 2e-5
     - 3
     - **71.6%**
     - **0.717**

.. note::

   V1 is severely overfit (train loss collapsed to 0.015).
   V3 shows mild overfitting starting at epoch 3.

.. toctree::
   :maxdepth: 2
   :caption: Pipeline Walkthrough

   pipeline/01_data_loading
   pipeline/02_preprocessing
   pipeline/03_label_encoding
   pipeline/04_feature_engineering
   pipeline/05_class_balancing
   pipeline/06_data_splitting
   pipeline/07_model_architecture
   pipeline/08_training_results
   pipeline/09_evaluation
   pipeline/10_improvements

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/data_processor
   api/models
   api/training_utils
   api/config

.. toctree::
   :maxdepth: 1
   :caption: Resources

   resources
