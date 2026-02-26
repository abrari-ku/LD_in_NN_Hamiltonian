# LD_in_NN_Hamiltonian

This repository provides a minimal, self-contained prototype reproducing the numerical experiments reported in the paper:

**“Phase Space Integrity in Neural Network Models of Hamiltonian Dynamics: A Lagrangian Descriptor Approach.”**

The implementation enables reproduction of the phase space analysis and Lagrangian Descriptor computations presented in the study.

The neural network architectures themselves are **not developed from scratch** in this work. Instead, they are based on publicly available implementations and incorporated with minor modifications to ensure compatibility with the analysis framework developed in this study.

---

## Language Structure

This repository contains implementations in **two programming languages**:

- **Python** – used for the Generalized Hamiltonian Neural Network (GHNN), SympNet, and HénonNet models.
- **Julia** – used for the Reservoir Computing (RC) models.

Please ensure that both Python and Julia environments are properly configured if you intend to reproduce all experiments.

---

## Original GHNN-Based Implementations (Python)

The following architectures used in this work:

- Generalized Hamiltonian Neural Network (GHNN)  
- SympNet  
- HénonNet  

are based on the open-source repository:

- **GHNN** – Philipp Horn (2025)  
  https://github.com/AELITTEN/GHNN

Only minor modifications were introduced. In particular, backward prediction capability was added to facilitate the computation of Lagrangian Descriptors.

All credit for the original neural network architectures belongs to the original authors.

---

## Original Reservoir Computing (RC) Implementation (Julia)

The reservoir computing (RC) models used in this repository are based on:

- **ReservoirComputing.jl** – Francesco Martinuzzi et al. (2022)  
  https://github.com/SciML/ReservoirComputing.jl

The original Julia implementation was adapted where necessary for integration with the Hamiltonian phase space analysis framework.

---

## Repository Structure

The repository is organized into four main directories:
---
LD_in_NN_Hamiltonian/
│
├── GHNN/ # Python implementation of GHNN, SympNet, and HénonNet
├── RC/ # Julia implementation of Reservoir Computing models
├── Postprocessing/ # Lagrangian Descriptor computation and analysis tools
├── Dataset/ # Training and evaluation datasets used in the study
└── README.md

### Directory Description

- **GHNN/**  
  Contains the Python implementation of the Generalized Hamiltonian Neural Network (GHNN), SympNet, and HénonNet architectures, adapted from the original open-source repository.

- **RC/**  
  Contains the Julia implementation of the Reservoir Computing (RC) models based on `ReservoirComputing.jl`.

- **Postprocessing/**  
  Contains scripts for:
  - Computing Lagrangian Descriptors  
  - Generating phase space visualizations  
  - Producing figures reported in the paper  

- **Dataset/**  
  Contains the training and testing datasets used in the numerical experiments.  
  All datasets are fixed to ensure reproducibility.
