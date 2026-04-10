7. Model Architecture
=====================

Base Model
----------

.. list-table::
   :widths: 25 50

   * - **Name**
     - NVIDIA Nemotron-3-Nano-30B-A3B-BF16
   * - **Architecture**
     - Mixture-of-Experts (MoE) transformer
   * - **Total Parameters**
     - ~30 billion
   * - **Active Parameters**
     - ~3 billion per token (A3B = Active 3 Billion)
   * - **Precision**
     - BF16 (bfloat16)
   * - **Hosting**
     - Tinker cloud infrastructure (remote GPU)

LoRA Fine-Tuning
-----------------

Instead of updating all 30B parameters, **LoRA** (Low-Rank Adaptation) adds
small trainable matrices to each layer:

.. code-block:: text

   Original weight: W             [d x d]   (frozen)
   LoRA matrices:   A             [d x r]   (trainable)
                    B             [r x d]   (trainable)
   Updated weight:  W' = W + A*B

   r = LoRA rank (much smaller than d)

   Example with rank=8, hidden_size=768:
     Original params per layer:   768 x 768 = 589,824
     LoRA params per layer:       768 x 8 + 8 x 768 = 12,288
     Reduction: 98% fewer trainable parameters!

Paper: `Hu et al. 2021 "LoRA" <https://arxiv.org/abs/2106.09685>`_

Experiment Configurations
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 8 8 10 8 8 50

   * - Version
     - Rank
     - LR
     - Batch
     - Epochs
     - Note
   * - v1
     - 16
     - 1e-4
     - 8
     - 3
     - Baseline -- **SEVERE overfit**
   * - v2
     - 16
     - 2e-5
     - 8
     - 6
     - Fix overfit via LR only
   * - **v3**
     - **8**
     - **2e-5**
     - **8**
     - **3**
     - Isolate rank effect (16->8)
   * - v4
     - 8
     - 2e-5
     - 4
     - 6
     - Smaller batch for noisier gradients

Prompt Template
---------------

From ``src/config.py``:

.. code-block:: text

   Classify the truthfulness of the following political statement.

   Statement: {text}

   Choose exactly one label: barely-true, false, half-true, mostly-true, pants-fire, true

   Label:

The model generates the label after ``Label:``. Completion tokens get
``weight=1`` (trained), prompt tokens get ``weight=0``.
At inference, ``temperature=0.0`` for deterministic output.

Tinker Datum Construction
-------------------------

Each training example is converted to a Tinker ``Datum`` object:

.. code-block:: python

   from data_processor import DataProcessor

   datum = DataProcessor.prepare_tinker_datum(
       text="The president claims unemployment is at 4%.",
       label="half-true",
       tokenizer=tokenizer,
       prompt_template=PROMPT_TEMPLATE
   )

Intermediate Output -- Token Construction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   1. Format prompt:     PROMPT_TEMPLATE.format(text=statement)
   2. Format completion: " {label}" (space + label string)
   3. Tokenize with BPE tokenizer
   4. Assign weights:    [0, 0, ..., 0, 1, 1, 1]
                          ^prompt^       ^completion^
   5. Shift by 1 for next-token prediction:
      input_tokens  = all_tokens[:-1]
      target_tokens = all_tokens[1:]
      weights       = weights[1:]

   Typical token counts:
     Prompt:     ~80-120 tokens
     Completion: ~2-4 tokens (e.g., " barely-true")
     Total:      ~82-124 tokens per example

.. admonition:: Alternative -- Better Model Choices

   **1. DeBERTa-v3-large** (recommended for classification):

   Encoder-only, purpose-built for classification. 304M params (vs 30B).

   .. code-block:: python

      from transformers import AutoModelForSequenceClassification, Trainer
      model = AutoModelForSequenceClassification.from_pretrained(
          'microsoft/deberta-v3-large', num_labels=6)

   HuggingFace: https://huggingface.co/microsoft/deberta-v3-large

   **2. SetFit** (few-shot, no GPU needed):

   .. code-block:: python

      from setfit import SetFitModel, SetFitTrainer
      model = SetFitModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

   GitHub: https://github.com/huggingface/setfit

   **3. BERT-base / RoBERTa-base**: 110-125M params, expected 65-71% accuracy.

   Tutorial: `HuggingFace Text Classification <https://huggingface.co/docs/transformers/tasks/sequence_classification>`_
