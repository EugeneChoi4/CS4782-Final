# Re-Implementation of the paper "A Time Series is Worth 64 Words: Long-Term Forecasting With Transformers"

## Introduction
- **Purpose:** This repo re-implements supervised and self-supervised PatchTST, a state-of-the-art Transformer for long-term multivariate forecasting.  
- **Paper:** "A Time Series Is Worth 64 Words: Long-Term Forecasting with Transformers" by Nie et al., which introduces **patching** and **channel-independence** to lower attention costs and improve accuracy.

## Chosen Result
- **Targeted Reproduction:**
  - **Table 3:** Supervised PatchTST on Electricity outperforms Informer, Pryaformer, and Autoformer  
  - **Table 4:** Self-supervised PatchTST further boosts accuracy compared to supervised PatchTST 
  - **Table 5:** Transfer-learned PatchTST matches supervised performance in far fewer epochs  
- **Significance:** Validates that patching + channel-independence shows improved long-horizon accuracy.

## GitHub Contents
- **README.md**: Project overview, instructions, and results 
- **code/**: Implementation of PatchTST (training, evaluation, and utilities)
  - `PatchTST_self_supervised.py`: The self-supervised PatchTST model
  - `PatchTST.py`: The supervised PatchTST model
  - `run_model.ipynb`: Visualize forecast for supervised model
  - `fine_tuning.ipynb`: Run the self-supervised model. Combine with supervised model for fine-tuning and transfer learning result across time horizons.
  - `num_patches.ipynb`: Visualizing relationship between number of patches and loss
  - `patch_size.ipynb`: Visualizing relationship between patch size and loss
- **data/**: Datasets for ETT (Electricity Transformer) and Electricity datasets
- **results/**: Generated figures & tables from our experiments 
- **poster/**: PDF of the in-class presentation poster
- **report/**: PDF of the final project report
- **LICENSE:** Project license  
- **.gitignore**: Files and directories excluded from Git  

## Re-implementation Details
- Implemented both supervised and self-supervised versions of PatchTST, following the original paper’s architecture and training procedures.
- Used ETTh1 and ETTm1 datasets with forecast horizons of 96, 192, and 336 steps; data was normalized using StandardScaler.
- Split multivariate time series into univariate channels, processed independently through a shared Transformer encoder.
- Applied patching (segmenting into fixed-length windows) and RevIN (Reversible Instance Normalization) to improve learning of local and global temporal patterns.
- Evaluated using MSE and MAE metrics to compare against original PatchTST results.
- Encountered limitations due to reduced training epochs and hardware constraints; used a standard Transformer instead of the paper’s variant with BatchNorm.
- Explored transfer learning by pretraining on a larger dataset and fine-tuning on a smaller one, confirming improved convergence with fewer epochs.

## Reproduction Steps
### Setup
1. (Optional) Set up and activate a python virtual environment
```
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# OR
venv\Scripts\activate  
```
2. Install python packages
```
pip install -r requirements.txt
```

### Running the model and reproducing results
1. Execute all the code blocks in `run_model.ipynb`
  - Train a supervised PatchTST model on the ETTm1 dataset
  - Visualize 96 time horizon forecast on the ETTm1 dataset
2. Execute all the code blocks in `fine_tuning.ipynb`
  - Collect results for self-supervised, supervised and transfer learning performance across different time horizons.
  - Visualize results (replace data with what was collected from above run)
3. Execute all the code blocks in `patch_size.ipynb`
  - Collect results for comparison of patch size vs. loss
4. Execute all the code blocks in `num_patches.ipynb`
  - Compare results for comparison of number of patches vs. loss
    - 42 patches
      - input length: 512, patch length: 12, stride: 12
    - 64 patches
      - input length: 336, patch length: 16, stride: 8

## Results/Insights
Our re-implementation of PatchTST reproduces the core performance trends reported in the original paper across ETTh1 and ETTm1 datasets at 96, 192, and 336 forecast horizons. While our MSE and MAE scores were slightly higher—likely due to fewer training epochs and architectural differences in the encoder—we found that patching and channel independence significantly improved performance and training speed compared to traditional Transformer models. Notably, we observed little difference between supervised and fine-tuned self-supervised training. In our transfer learning experiments using the Electricity dataset, the pretrained model achieved comparable performance with fewer epochs, validating the paper’s findings on training efficiency. We also confirmed that loss generally increases with longer forecast horizons, while variations in patch size, stride, and lookback window had minimal impact on performance. This repository offers a clean, modular implementation of PatchTST with support for supervised, self-supervised, and transfer learning workflows.


## Conclusion
Our re-implementation of PatchTST reinforced the effectiveness of patching and channel-wise independence in improving both performance and training efficiency for time series forecasting. We learned that self-supervised pretraining can offer similar performance to fully supervised training, especially when fine-tuned on smaller datasets. Additionally, transfer learning proved valuable in accelerating convergence with fewer epochs. Minor implementation differences—such as the absence of BatchNorm layers—can meaningfully affect performance, highlighting the importance of architectural details often omitted from papers. Overall, we gained a deeper understanding of Transformer-based time series models and the practical considerations involved in reproducing deep learning research.

## References
[1] Guokun Lai, Wei-Cheng Chang, Yiming Yang, and Hanxiao Liu. Modeling long- and short-
term temporal patterns with deep neural networks, 2018.

[2] Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong, and Jayant Kalagnanam. A time series is
worth 64 words: Long-term forecasting with transformers, 2023.

[3] Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong, and Wancai
Zhang. Informer: Beyond efficient transformer for long sequence time-series forecasting. In
The Thirty-Fifth AAAI Conference on Artificial Intelligence, AAAI 2021, Virtual Conference,
volume 35, pages 11106–11115. AAAI Press, 2021.

## Acknowledgements
This project was completed as part of the coursework for CS 4782: Deep Learning at Cornell University. We would like to thank Professor Kilian Weinberger, Assistant Professor Jennifer Sun and the course staff for providing guidance and support throughout the semester. The structure and feedback received during this course were instrumental in shaping the direction and rigor of our re-implementation and analysis.

