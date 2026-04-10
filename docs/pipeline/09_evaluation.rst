9. Evaluation Deep Dive
=======================

Comparison with Published Baselines
------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 50 15 25

   * - Method
     - Accuracy
     - Notes
   * - Majority class baseline
     - 20.6%
     - LIAR official test
   * - Logistic Regression + BoW
     - 25.5%
     - LIAR official test
   * - CNN (Wang 2017)
     - 27.0%
     - LIAR official test
   * - BERT-base fine-tuned
     - ~42%
     - LIAR official test (approx)
   * - **This Project V3**
     - **71.6%**
     - Re-split of LIAR2 train.tsv

.. warning:: Comparison Caveats

   1. **Different splits** -- published results use official test set;
      this project evaluates on a random re-split of train.tsv.
   2. The 30B model has a massive advantage over BERT-base.
   3. Oversampling before splitting may inflate test accuracy.
   4. LIAR2 != original LIAR (slightly different data).

Why 71.6% May Be Near the Ceiling
----------------------------------

6-class truthfulness classification is **extremely difficult**:

1. **Human agreement**: Even trained fact-checkers disagree on boundaries
   between adjacent labels. Inter-annotator agreement is ~60-70%.

2. **Subjective boundaries**: "mostly-true" vs "half-true" is often a
   judgment call, not factual.

3. **Short text**: Average 18 words per statement -- very limited signal.

Reaching 80%+ would likely require:

- Using metadata (speaker credibility history, context)
- Simplifying to 3 classes (true/mixed/false)
- Evidence retrieval (checking claims against external sources)

.. admonition:: Alternatives -- Better Evaluation

   **1. Evaluate on official test.tsv**:

   .. code-block:: python

      test_df = pd.read_csv('data/test.tsv', sep='\t', ...)
      # Predict and compare with published benchmarks

   **2. LIME explainability** (understand individual predictions):

   .. code-block:: python

      from lime.lime_text import LimeTextExplainer
      explainer = LimeTextExplainer(class_names=labels)
      exp = explainer.explain_instance(text, predict_fn, num_features=10)
      exp.as_pyplot_figure()

   **3. SHAP** (global feature importance):

   .. code-block:: python

      import shap
      explainer = shap.Explainer(model, tokenizer)
      shap_values = explainer(texts[:100])
      shap.plots.text(shap_values[0])

   **4. Calibration curves** (are confidence scores reliable?):

   .. code-block:: python

      from sklearn.calibration import calibration_curve
      prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)

   Resources:
   - `LIME tutorial (YouTube) <https://www.youtube.com/watch?v=d6j6bofhj2M>`_
   - `SHAP documentation <https://shap.readthedocs.io/>`_
