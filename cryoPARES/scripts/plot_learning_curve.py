import os.path
import sys

import pandas as pd
import matplotlib.pyplot as plt

fname = os.path.expanduser(sys.argv[1]) #'metrics.csv')

df_clean = pd.read_csv(fname)
df_val = df_clean[['step', 'val_geo_degs', 'geo_degs_epoch', 'val_loss']]
df_val = df_val.dropna()
print(df_clean.columns)

plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
# plt.plot(df_clean['step'], df_clean['geo_degs_step'], 'b-', label='geo_degs_step')
plt.plot(df_clean['step'], df_clean['geo_degs_epoch'], 'g.-', label='geo_degs_epoch')
plt.plot(df_val['step'], df_val['val_geo_degs'], 'y-', label='val_geo_degs')
plt.ylabel('Geo Degrees')
plt.legend()
plt.grid(True)

if "loss" in df_clean.columns:
    plt.subplot(2, 1, 2)
    plt.plot(df_clean['step'], df_clean['loss'], 'r-', label='loss')
    plt.plot(df_val['step'], df_val['val_loss'], 'b-', label='val_loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()

