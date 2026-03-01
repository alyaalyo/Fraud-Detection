# NIC-project
Evolutionary optimization of neural network architectures for financial fraud detection using the IEEE-CIS dataset.

## Abstract
This project investigates the application of evolutionary algorithms for automatic optimization of neural network architectures in financial fraud detection. Using the IEEE-CIS Fraud Detection dataset, we evolve Multi-Layer Perceptron (MLP) and Convolutional Neural Network (CNN) architectures through a Genetic Algorithm (GA). The objective is to improve fraud classification performance on highly imbalanced transaction data by automating architecture search instead of manual tuning. The study integrates nature-inspired computing, machine learning baselines, and deep learning models within a unified experimental framework.

## Keywords
Nature Inspired Computing, Genetic Algorithm, Neural Architecture Search, Fraud Detection, Deep Learning

## Project Idea 
Fraud detection in online transactions is a highly imbalanced binary classification problem with significant financial impact. Designing effective neural network architectures for such data typically requires manual experimentation and hyperparameter tuning.

In this project, we propose an evolutionary optimization framework that automatically searches for optimal neural network architectures using a Genetic Algorithm (GA). Instead of comparing classical machine learning models, we focus on comparing:

- Fixed baseline neural architectures (MLP and CNN)
- Architectures evolved by the Genetic Algorithm

The objective is to evaluate whether evolutionary optimization can discover neural architectures that outperform manually designed models in terms of F1-score and ROC-AUC.

## Methodology

Data Processing  
The IEEE-CIS dataset will be cleaned, encoded, and scaled. Missing values will be handled using imputation strategies. The dataset will be split into training and validation subsets. Class imbalance will be addressed using class weighting or resampling techniques.

Baseline Neural Architectures  
We implement fixed neural architectures for comparison:

- Multi-Layer Perceptron (MLP)
- Convolutional Neural Network (CNN) adapted for tabular data

These baseline models will be manually configured and evaluated using F1-score and ROC-AUC.

Genetic Algorithm for Neural Architecture Search  
Each individual in the population encodes a neural network configuration, including:

- Number of layers
- Number of neurons per layer
- Activation functions
- Dropout rates

The Genetic Algorithm includes:

- Random population initialization
- Tournament selection
- Crossover and mutation operators
- Elitism strategy

The fitness function trains each neural architecture for a limited number of epochs and evaluates validation performance. Evolution proceeds over multiple generations to improve architecture quality.

## Dataset
IEEE-CIS Fraud Detection dataset: `https://www.kaggle.com/competitions/ieee-fraud-detection/data`

The dataset contains anonymized transactional and identity features from real-world e-commerce transactions. It includes numerical and categorical variables and represents a highly imbalanced fraud detection problem.

## Timeline and Individual Contributions

### Roles  
Student A (Artem Mikhailin): Data preprocessing, exploratory data analysis, and implementation of baseline neural architectures (MLP/CNN).  
Student B (Karina Shaikhutdinova): Design and implementation of the Genetic Algorithm, including architecture encoding and evolutionary operators.  
Student C (Alena Petrenko): Integration of GA with neural network training, experimental evaluation, logging, and visualization of results.

### Roadmap

Week 1: Infrastructure and Setup  
Data exploration and preprocessing; architecture encoding; repository setup and basic NN training pipeline.

Week 2: Core GA Components  
Baseline MLP implementation; mutation and crossover operators; integration of GA with training pipeline.

Week 3: Working Prototype (Checkpoint)  
Elitism and population management; small-scale evolutionary run; reproducible GitHub submission.

Week 4: Baseline Optimization  
Hyperparameter tuning for fixed architectures; GA performance optimization; experiment logging and visualization.

Week 5: Main Evolutionary Experiment  
Full GA run (population 20, 15 generations); retraining best architecture; comparison with baseline models.

Week 6: Report Drafting  
Writing Introduction, Dataset, Methodology, Experiments, and Analysis sections; preparing figures.

Week 7: Finalization and Demo Preparation  
Code refactoring, documentation, final PDF submission, presentation rehearsal, and working demo.

## References

- IEEE-CIS Fraud Detection Dataset, Kaggle, 2019. `https://www.kaggle.com/competitions/ieee-fraud-detection`
- D. E. Goldberg, Genetic Algorithms in Search, Optimization and Machine Learning. Addison-Wesley, 1989.
- I. Goodfellow, Y. Bengio, and A. Courville, Deep Learning. MIT Press, 2016.
- A. Petrenko et al., "NIC-project: Evolutionary optimization of neural networks for fraud detection," GitHub repository, 2026. `https://github.com/alyaalyo/NIC-project`

## Week 1 Baseline Pipeline

### Baseline Pipeline
We implemented reproducible PyTorch training pipeline for the IEEE-CIS Fraud Detection dataset. It is designed to serve as a baseline and a future GA fitness function.

## Quick Start

1. Place Kaggle files locally (not downloaded by the code):
   - `data/train_transaction.csv`
   - `data/train_identity.csv` (optional)

2. Create environment and install dependencies:

```bash
pip install -r requirements.txt
```

3. Run baseline training:

```bash
python scripts/train_baseline.py --config configs/baseline_mlp.yaml
```

Artifacts are saved under `runs/<run_id>/`:
- `config.yaml`
- `metrics.json`
- `model.pt` (best by val ROC-AUC)
- `train.log`
