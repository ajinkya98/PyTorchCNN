import pandas as pd
import os
import torch

device = ("cuda" if torch.cuda.is_available() else "cpu")

train_df = pd.DataFrame(columns=["img_path","label"])
train_df["img_path"] = os.listdir("train/")
for idx, i in enumerate(os.listdir("train/")):
    if "cat" in i:
        train_df["label"][idx] = 0
    if "dog" in i:
        train_df["label"][idx] = 1

train_df.to_csv (r'train_csv.csv', index = False, header=True)