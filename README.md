# disinformation_classification_pietervanbrakel

![PyPI version](https://img.shields.io/pypi/v/disinformation_classification_pietervanbrakel.svg)
[![Documentation Status](https://readthedocs.org/projects/disinformation_classification_pietervanbrakel/badge/?version=latest)](https://disinformation_classification_pietervanbrakel.readthedocs.io/en/latest/?version=latest)

This project classifies news articles with binary predictions: true or false.

* PyPI package: https://pypi.org/project/disinformation_classification_pietervanbrakel/
* Free software: MIT License
* Documentation: https://disinformation_classification_pietervanbrakel.readthedocs.io.

## Features

* TODO

## Credits

This package was created with [Cookiecutter](https://github.com/audreyfeldroy/cookiecutter) and the [audreyfeldroy/cookiecutter-pypackage](https://github.com/audreyfeldroy/cookiecutter-pypackage) project template.

---

# Models Directory Overview

This folder contains all models and related artefacts for this project.  
The structure is designed to keep experiments, production models, and pretrained resources organized.

## Subfolders

### `/production`
- **Purpose:** Stable, trained models that are ready for use in production or inference.  
- **Examples:** `svm_model.pkl`, `count_vectorizer.pkl`, `metadata.json`.  
- **Notes:** These are the models you can safely push to GitHub.

### `/experiment`
- **Purpose:** Models created during experiments (e.g., trying different hyperparameters, preprocessing).  
- **Examples:** `svm_model_trial1.pkl`, `tfidf_vectorizer_trial1.pkl`.  
- **Notes:** Usually **not pushed** to GitHub. Good for local testing only.

### `/tmp`
- **Purpose:** Temporary models or intermediate results during development.  
- **Notes:** Typically ignored in version control (`.gitignore`).

### `/meta`
- **Purpose:** Metadata about models, including:
  - Training parameters (`kernel`, `C`, etc.)
  - Dataset version
  - Training date
- **Examples:** `model_versions.json`, `svm_model_meta.json`.

### `/pretrained`
- **Purpose:** Pretrained models or embeddings downloaded externally.  
- **Examples:** Hugging Face Transformers (`bert-base-uncased`), SpaCy (`en_core_web_sm`), GloVe embeddings (`glove.6B.100d.txt`).  
- **Notes:**
  - **Do not push large pretrained models to GitHub.**  
  - Include instructions in your main README on how to download or install them.  
  - Use `.gitignore` to avoid accidentally committing large files.

---

## Recommended Usage

1. Always store your final trained models in `/production`.  
2. Keep experiments isolated in `/experiment`.  
3. Save metadata for reproducibility in `/meta`.  
4. Use `/pretrained` for any external models; document download instructions clearly.  
5. Keep temporary or intermediate models in `/tmp` and ignore them in version control.

