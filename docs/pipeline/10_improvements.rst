10. Improvement Roadmap
=======================

Quick Wins (Low Effort, High Impact)
-------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 10 50 15

   * - #
     - Action
     - Effort
   * - 1
     - **Use official test split** -- load ``test.tsv`` for evaluation
     - 30 min
   * - 2
     - **Add early stopping** -- stop when val_loss increases for 2 epochs
     - 15 min
   * - 3
     - **Fix oversampling order** -- split first, then oversample training only
     - 30 min
   * - 4
     - **Fix label display bug** -- use ``le.classes_[idx]`` in notebook 01
     - 5 min
   * - 5
     - **Save per-class metrics** -- add ``classification_report()`` to results
     - 15 min

Medium Effort Improvements
--------------------------

**1. Include metadata in prompts** (2 hours)

.. code-block:: text

   Current:  "Statement: {text}"
   Enhanced: "Speaker: {speaker} (Party: {party}). Context: {context}. Statement: {text}"

**2. Learning rate scheduler** (1 hour)

Add cosine annealing with warmup to ``TinkerTrainer``.

**3. Optuna hyperparameter search** (4 hours)

Search: ``rank in {4,8,16,32}``, ``lr in {1e-5, 5e-5, 1e-4}``,
``batch in {4,8,16}``.

**4. Consolidate notebooks** (2 hours)

Notebook 02 does oversampling/splitting that notebook 03 re-does independently.

New Directions
--------------

**1. DeBERTa-v3-large** (1 day)

.. code-block:: python

   from transformers import (AutoModelForSequenceClassification,
                            AutoTokenizer, TrainingArguments, Trainer)

   model = AutoModelForSequenceClassification.from_pretrained(
       'microsoft/deberta-v3-large', num_labels=6)
   tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-large')

   args = TrainingArguments(
       output_dir='./results',
       num_train_epochs=5,
       per_device_train_batch_size=16,
       learning_rate=2e-5,
       evaluation_strategy='epoch',
       load_best_model_at_end=True,
   )

   trainer = Trainer(model=model, args=args,
                     train_dataset=train_ds, eval_dataset=val_ds)
   trainer.train()

**2. SetFit for few-shot** (2 hours)

.. code-block:: python

   from setfit import SetFitModel, SetFitTrainer
   model = SetFitModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
   trainer = SetFitTrainer(model=model,
                          train_dataset=train_ds,
                          eval_dataset=val_ds)
   trainer.train()

**3. Simplify to 3 classes** (30 min)

.. code-block:: python

   label_map = {
       'true': 'TRUE', 'mostly-true': 'TRUE',
       'half-true': 'MIXED',
       'barely-true': 'FALSE', 'false': 'FALSE', 'pants-fire': 'FALSE',
   }
   df['label_3class'] = df['label'].map(label_map)

Expected accuracy: **80%+**.

**4. Evidence retrieval** (1-2 weeks)

Retrieve relevant evidence from a knowledge base for each claim.
Paper: `Evidence-based Fact Checking <https://arxiv.org/abs/2104.05834>`_

**5. Multi-task learning** (1 week)

Train model to both classify AND generate a justification.
The LIAR2 ``justification`` column could serve as target.
