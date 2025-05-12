import matplotlib.pyplot as plt

# Data
horizons = [24, 48, 96, 192, 336, 720]
mse_supervised = [0.3435, 0.3652, 0.4373, 0.4818, 0.5463, 0.6883]
mse_selfsup = [0.3289491703067351, 0.3718768596925117, 0.4247183399099224, 
               0.4865285615319187, 0.5654917169700969, 0.7143809271269831]

# Plot
plt.figure()
plt.plot(horizons, mse_supervised, marker='o', label='Supervised')
plt.plot(horizons, mse_selfsup, marker='o', label='Self-Supervised Pretrain + Finetune')
plt.xlabel('Forecast Horizon')
plt.ylabel('MSE')
plt.title('ETTh1 Forecasting: MSE vs Forecast Horizon')
plt.legend()
plt.grid(True)
plt.xticks(horizons)
plt.tight_layout()
plt.show()
