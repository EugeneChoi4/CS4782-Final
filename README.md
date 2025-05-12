# Re-Implementation of the paper "A Time Series is Worth 64 Words: Long-Term Forecasting With Transformers"

## Introduction
- **Purpose:** This repo re-implements supervised and self-supervised PatchTST, a state-of-the-art Transformer for long-term multivariate forecasting.  
- **Paper:** "A Time Series Is Worth 64 Words: Long-Term Forecasting with Transformers" by Nie et al., which introduces **patching** and **channel-independence** to lower attention costs and improve accuracy.

## Chosen Result
- **Targeted Reproduction:** Table 3's supervised PatchTST results on the Electricity Transformer dataset that showed improvements over other time-series transformers such as the Informer and Autoformer. Table 4 shows that the self-supervised PatchTST model offers more accuracy improvements, and Table 5 demonstrates that, with transfer learning, PatchTST can get close to the supervised model’s accuracy while converging in fewer training epochs.
- **Significance:** Validates that patching + channel-independence shows improved long-horizon accuracy.

## GitHub Contents
- **README.md:**: Project overview, instructions, and results 
- **code/**: Implementation of PatchTST (training, evaluation, and utilities)
- **data/**: Datasets for ETT (Electricity Transformer) and Electricity datasets
- **results/**: Generated figures & tables from our experiments 
- **poster/**: PDF of the in-class presentation poster
- **report/**: PDF of the final project report
- **LICENSE:** Project license  
- **.gitignore:** Files and directories excluded from Git  



