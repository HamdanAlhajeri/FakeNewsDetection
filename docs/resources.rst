Resources & References
======================

Tutorial Videos
---------------

**NLP & Transformers**

- `HuggingFace NLP Course <https://huggingface.co/learn/nlp-course>`_ (free, comprehensive)
- `HuggingFace Transformers Playlist <https://www.youtube.com/playlist?list=PLo2EIpI_JMQvWfQndUesu0nPBAtZ9gP1o>`_ (official YouTube)
- `Fine-tuning LLMs - Sebastian Raschka <https://www.youtube.com/watch?v=eC6Hd1hFvos>`_

**LoRA & PEFT**

- `LoRA Explained - Umar Jamil <https://www.youtube.com/watch?v=PXWYUTMt-AU>`_ (visual, detailed)
- `QLoRA & PEFT Tutorial - Trelis Research <https://www.youtube.com/watch?v=J_3hDqSvpmg>`_

**Fake News Detection**

- `Fake News Classifier - Krish Naik <https://www.youtube.com/watch?v=zetNWSmKSfY>`_

**Evaluation & ML Fundamentals**

- `Cross-Validation - StatQuest <https://www.youtube.com/watch?v=fSytzGwwBVw>`_
- `Confusion Matrix - StatQuest <https://www.youtube.com/watch?v=Kdsp6soqA7o>`_
- `F1 Score - StatQuest <https://www.youtube.com/watch?v=jJ7ff7Gcq34>`_

**Explainability**

- `LIME Explained <https://www.youtube.com/watch?v=d6j6bofhj2M>`_
- `SHAP Explained <https://www.youtube.com/watch?v=VB9rkYgJAKI>`_


Key Papers
----------

**Dataset**

- Wang 2017, `"Liar, Liar Pants on Fire" <https://aclanthology.org/P17-2067/>`_ -- original LIAR dataset
- Alhindi et al. 2018, `"Where is your Evidence" <https://aclanthology.org/W18-5513/>`_ -- LIAR2 with justifications

**Models**

- Hu et al. 2021, `"LoRA" <https://arxiv.org/abs/2106.09685>`_ -- Low-Rank Adaptation
- Dettmers et al. 2023, `"QLoRA" <https://arxiv.org/abs/2305.14314>`_ -- Quantized LoRA
- He et al. 2021, `"DeBERTa" <https://arxiv.org/abs/2006.03654>`_ -- Disentangled attention
- Devlin et al. 2019, `"BERT" <https://arxiv.org/abs/1810.04805>`_ -- Bidirectional transformers
- Liu et al. 2019, `"RoBERTa" <https://arxiv.org/abs/1907.11692>`_ -- Robust BERT pretraining

**Techniques**

- Lin et al. 2017, `"Focal Loss" <https://arxiv.org/abs/1708.02002>`_ -- class imbalance
- Ribeiro et al. 2016, `"LIME" <https://arxiv.org/abs/1602.04938>`_ -- model explainability


Online Courses
--------------

- `HuggingFace NLP Course <https://huggingface.co/learn/nlp-course>`_ (Chapters 1-8, free)
- `Fast.ai Practical Deep Learning <https://course.fast.ai/>`_ (Lessons 1-9, free)
- `Stanford CS224N <https://web.stanford.edu/class/cs224n/>`_ -- NLP with Deep Learning


Tools & Libraries
-----------------

.. list-table::
   :header-rows: 1
   :widths: 20 35 25

   * - Library
     - Purpose
     - Install
   * - transformers
     - HuggingFace models (BERT, DeBERTa)
     - ``pip install transformers``
   * - torch
     - Deep learning framework
     - ``pip install torch``
   * - tinker
     - Tinker SDK for LLM training
     - ``pip install tinker``
   * - scikit-learn
     - ML utilities, metrics, splits
     - ``pip install scikit-learn``
   * - imbalanced-learn
     - Class balancing (SMOTE, etc.)
     - ``pip install imbalanced-learn``
   * - optuna
     - Hyperparameter optimization
     - ``pip install optuna``
   * - lime
     - Local model explainability
     - ``pip install lime``
   * - shap
     - Global feature importance
     - ``pip install shap``
   * - nlpaug
     - Text data augmentation
     - ``pip install nlpaug``
   * - setfit
     - Few-shot fine-tuning
     - ``pip install setfit``
   * - spacy
     - Advanced NLP preprocessing
     - ``pip install spacy``
   * - ydata-profiling
     - Automated EDA reports
     - ``pip install ydata-profiling``
   * - peft
     - HuggingFace LoRA/PEFT
     - ``pip install peft``
   * - datasets
     - HuggingFace dataset library
     - ``pip install datasets``
