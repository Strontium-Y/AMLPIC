## AMLPIC: Cross-Domain Few-Shot Hyperspectral Image Classification

This repository contains the official implementation of the paper:  
*"Adaptive Meta-Learning with Physics-Informed Constraints for Few-Shot Cross-Domain Hyperspectral Image Classification"* (AMLPIC)

---

## Overview

This project proposes a meta-learning framework that integrates physics-informed regularization, adaptive weighting, and adversarial domain alignment for cross-domain few-shot hyperspectral image (HSI) classification. The framework embeds a physics-informed loss derived from spectral absorption equations into MAML inner-loop gradient updates, guiding the adaptation process toward physically plausible spectral variations.

---

## Project Structure

- ├── `SSFTTnet_200_1_DA.py` – SSFTT Transformer backbone  
- ├── `mamba_branch.py` – Mamba branch for spectral modeling  
- ├── `agents.py` – Multi-layer gradient agents  
- ├──  `new_SSFTT_MAML_DA_RL_AdaptivePINN_hybrid.py` – Main model definition
- 
- ├── `data_utils.py` – Source domain data loader (Indian Pines)  
- ├── `data_utils_Salinas.py` – Target domain data loader (Salinas)  
- ├── `com_data_utils.py` – Unified dataset loader for all datasets  
- ├── `task_sampler.py` – N-way K-shot episode sampler
- 
- ├── `com_train_hybrid.py` – Main training script (10 epochs, 100 episodes/epoch)  
- ├── `com_test_adaptivePINN.py` – Testing script with 5 random seeds  
- ├── `t-SNE_Botswana.py` – Feature visualization script
- 
- ├── `run_sensitivity_T_salinas.py` – Sensitivity analysis for inner-loop steps  
- ├── `run_sensitivity_salinas_val2.py` – Sensitivity analysis for other hyperparams  
- ├── `README.md`
