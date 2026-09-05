## AMLPIC: Cross-Domain Few-Shot Hyperspectral Image Classification

This repository contains the official implementation of the paper:  
*"Adaptive Meta-Learning with Physics-Informed Constraints for Few-Shot Cross-Domain Hyperspectral Image Classification"* (AMLPIC)

---

## Overview

This project proposes a meta-learning framework that integrates physics-informed regularization, adaptive weighting, and adversarial domain alignment for cross-domain few-shot hyperspectral image (HSI) classification. The framework embeds a physics-informed loss derived from spectral absorption equations into MAML inner-loop gradient updates, guiding the adaptation process toward physically plausible spectral variations.

---

## Project Structure

- ├── `SSFTTnet_200_1_DA.py`
- ├── `mamba_branch.py`   
- ├── `agents.py`   
- ├──  `new_SSFTT_MAML_DA_RL_AdaptivePINN_hybrid.py` 
- 
- ├── `data_utils.py`   
- ├── `data_utils_Salinas.py`   
- ├── `com_data_utils.py`  
- ├── `task_sampler.py` 
- 
- ├── `com_train_hybrid.py`   
- ├── `com_test_adaptivePINN.py` 
- ├── `t-SNE_Botswana.py` 
- 
- ├── `run_sensitivity_T_salinas.py`  
- ├── `run_sensitivity_salinas_val2.py`   
- ├── `README.md`
