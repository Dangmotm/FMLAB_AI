# FMLAB AI Coursework Portfolio

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?logo=jupyter&logoColor=white)
![NLP](https://img.shields.io/badge/Focus-ML%20%7C%20DL%20%7C%20NLP-2E7D32)

This repository collects my machine learning, deep learning, and NLP coursework from FMLAB. It is organized as a learning portfolio: classical ML foundations, PyTorch deep learning practice, and selected Stanford CS224N-style NLP assignments.

> Note: this README focuses on `Cs224n`, `Exercise`, and `Homework`, which are the coursework sections intended for portfolio review.

## Highlights

- Implemented classical ML pipelines for preprocessing, classification, model selection, and evaluation.
- Built deep learning notebooks covering MLPs, CNNs, ResNet-style models, DenseNet, RNNs, GRUs, LSTMs, GANs, and data augmentation.
- Completed NLP assignments on word vectors, dependency parsing, neural machine translation, attention, GPT-style pretraining, fine-tuning, and RoPE positional embeddings.
- Practiced experiment tracking with TensorBoard-style runs, checkpoints, predictions, and evaluation outputs.

## Repository Structure

| Path | Contents | Main Skills |
| --- | --- | --- |
| [`Cs224n/`](./Cs224n) | Four NLP assignments inspired by Stanford CS224N | word vectors, dependency parsing, NMT, attention, GPT, RoPE |
| [`Exercise/`](./Exercise) | 18 in-class / practice exercise folders | ML algorithms, PyTorch training, CNNs, RNNs, GANs, graph embeddings |
| [`Homework/`](./Homework) | 18 homework folders | preprocessing, visualization, ML models, CNNs, sequence models, graph learning |

## CS224N Work

| Assignment | Focus | Representative Files |
| --- | --- | --- |
| A1 | Word vectors and embedding exploration | [`exploring_word_vectors.ipynb`](./Cs224n/a1/student/exploring_word_vectors.ipynb) |
| A2 | Neural transition-based dependency parsing | [`parser_model.py`](./Cs224n/a2/student-1/parser_model.py), [`parser_transitions.py`](./Cs224n/a2/student-1/parser_transitions.py) |
| A3 | Neural machine translation with attention | [`nmt_model.py`](./Cs224n/a3/student/nmt_model.py), [`model_embeddings.py`](./Cs224n/a3/student/model_embeddings.py) |
| A4 | GPT-style pretraining/fine-tuning and RoPE | [`attention.py`](./Cs224n/a4/student/src/attention.py), [`run.py`](./Cs224n/a4/student/src/run.py), [`trainer.py`](./Cs224n/a4/student/src/trainer.py) |

## Exercise Topics

The `Exercise` directory contains practice notebooks and scripts across:

- Data preprocessing and exploratory analysis
- KNN, Naive Bayes, SVM, decision trees, random forests, and model selection
- Linear regression and neural network fundamentals
- PyTorch training loops, CNNs, ResNet, and hyperparameter tuning
- RNN, LSTM, image captioning, GANs, graph embeddings, and lab-style implementations

## Homework Topics

The `Homework` directory mirrors the course progression with independent assignments:

- Vietnamese news preprocessing and text classification
- Real estate data visualization and tabular ML workflows
- Decision tree, random forest, Naive Bayes, model selection, and image compression
- Neural networks, MLPs, LeNet, DenseNet, data augmentation, and ensembles
- GRU / bidirectional RNN practice, sequence modeling, DCGAN, and graph learning

## Quick Start

Most work is notebook-based. Create an environment with common ML/DL packages, then open the notebooks from the relevant assignment folder.

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas matplotlib scikit-learn jupyter torch torchvision tqdm tensorboard
jupyter notebook
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install numpy pandas matplotlib scikit-learn jupyter torch torchvision tqdm tensorboard
jupyter notebook
```

Some CS224N assignments include their own environment files or requirements:

- [`Cs224n/a1/student/env.yml`](./Cs224n/a1/student/env.yml)
- [`Cs224n/a2/student-1/local_env.yml`](./Cs224n/a2/student-1/local_env.yml)
- [`Cs224n/a3/student/requirements.txt`](./Cs224n/a3/student/requirements.txt)

## Running Selected Scripts

Dependency parser sanity checks:

```bash
cd Cs224n/a2/student-1
python parser_model.py --embedding
python parser_model.py --forward
```

Neural machine translation sanity checks:

```bash
cd Cs224n/a3/student
python sanity_check.py 1d
python sanity_check.py 1e
python sanity_check.py 1f
```

GPT-style assignment entry point:

```bash
cd Cs224n/a4/student
python src/run.py pretrain vanilla wiki.txt --writing_params_path vanilla.model.params
```

## Portfolio Notes

- The main value of this repository is in the implemented notebooks and model code, not in generated artifacts.
- Large datasets, checkpoints, TensorBoard logs, and prediction files should be treated as reproducible outputs.
- `.gitignore` is configured to avoid adding new generated training artifacts such as checkpoints, TensorBoard event files, and model parameter dumps.

## Author

**Luu Hai Dang**  
Machine Learning / Deep Learning Coursework Portfolio
