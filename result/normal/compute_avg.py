"""
Tính trung bình các metrics từ Missing Rate 0.0 đến 0.9 cho từng model
trong file report_grouped.csv
"""

import pandas as pd
import os

# Đường dẫn file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, "report_grouped.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "report_avg.csv")

# Đọc dữ liệu
df = pd.read_csv(INPUT_FILE)

# Các cột metric (Mean)
metric_mean_cols = [
    "Acc_2 Mean", "F1_score Mean", "MAE Mean",
    "Corr Mean", "Acc_7 Mean", "Loss Mean"
]

# Lọc chỉ các dòng Missing Rate từ 0.0 đến 0.9
df_filtered = df[df["Missing Rate"].between(0.0, 0.9)]

# Tính trung bình theo từng model
avg_df = (
    df_filtered
    .groupby("Model")[metric_mean_cols]
    .mean()
    .round(4)
    .reset_index()
)

avg_df.insert(1, "Missing Rate", "0.0 - 0.9 (avg)")

# Lưu kết quả
avg_df.to_csv(OUTPUT_FILE, index=False)

print(f"Đã lưu kết quả vào: {OUTPUT_FILE}\n")
print(avg_df.to_string(index=False))
