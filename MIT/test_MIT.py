import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import numpy as np
import time
from torch.utils.data import DataLoader
from dataloader import DGA_Dataset
from MIT_model import MIT
from tqdm import tqdm
from sklearn.metrics import roc_curve
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utility.path import path_train_data, path_test_data, path_artifacts, path_images, path_results
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def cutting_df(df, count, rank) :
    df = df[~(
        (df['day_count'] <= count) &
        (df['avg_rank'] >= rank)
    )].reset_index(drop=True)
    return df

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--count', type=int, default=1, help='Unique count threshold')
    parser.add_argument('--rank', type=int, default=1000000, help='Rank threshold')
    args = parser.parse_args()

    count = args.count
    rank = args.rank

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MIT().to(device)

    batch_size = 100
    target_cols = ['domain', 'label']

    dga_val_df = pd.read_parquet(path_train_data.joinpath('T24_dga_1day_val.parquet'), columns=target_cols)

    cols_to_keep = target_cols + ['day_count','avg_rank'] if 'day_count' not in target_cols or 'avg_rank' not in target_cols else target_cols
    benign_df = pd.read_parquet(path_train_data.joinpath('T24_benign_1day.parquet'), columns=cols_to_keep)

    benign_df = cutting_df(benign_df, count, rank)
    # benign_df = benign_df[target_cols]

    _, benign_val_df = train_test_split(benign_df, test_size=0.1, random_state=42)

    val_df = pd.concat([benign_val_df, dga_val_df]).reset_index(drop=True)

    val_dataset = DGA_Dataset(val_df)
    val_loader  = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model.load_state_dict(torch.load(path_artifacts.joinpath(f'MIT_T24_{count}_{rank}.pt')))
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # Treshold 찾기
    val_output = []
    val_labels = []

    with torch.no_grad() :
        for X_val, y_val in tqdm(val_loader, desc=f"[Threshold]", leave=False):
            X_val, y_val = X_val.to(device), y_val.to(device)
            outputs = model(X_val)
            val_output.append(outputs.squeeze().cpu())
            val_labels.append(y_val.cpu())
    val_output = torch.cat(val_output)
    val_labels = torch.cat(val_labels)
    fpr, tpr, thresholds = roc_curve(val_labels, val_output)
    theta = thresholds[(tpr-fpr).argmax()]
    print(f'Threshold: {theta:.4f}\n')

    # 테스트
    dga_test_df = pd.read_parquet(path_test_data.joinpath('T25-26_dga_1day.parquet'),columns=target_cols)

    benign_test_df = pd.read_parquet(path_test_data.joinpath('T25-26_benign_1day.parquet'))
    benign_test_df = cutting_df(benign_test_df, count, rank)
    benign_test_df = benign_test_df[target_cols]

    test_df = pd.concat([benign_test_df, dga_test_df]).reset_index(drop=True)

    # test_df['original_domain'] = test_df['domain']
    # test_df['domain'] = test_df['domain'].apply(process_domain)
    test_dataset = DGA_Dataset(test_df)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    all_preds = []
    all_labels = []
    start_time = time.time()

    model.eval()
    with torch.no_grad():
        for X_test, y in tqdm(test_loader, desc="[Test]", leave=False):
            X_test, y = X_test.to(device), y.to(device)
            outputs = model(X_test)
            all_preds.append(outputs.squeeze().cpu())
            all_labels.append(y.squeeze().cpu())

    test_time = time.time() - start_time
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    pred_labels = (all_preds > theta).int().numpy()
    true_labels = all_labels.int().numpy()

    accuracy = (pred_labels == true_labels).sum().item() / len(true_labels)
    precision = precision_score(true_labels, pred_labels, zero_division=0)
    recall = recall_score(true_labels, pred_labels, zero_division=0)
    f1 = f1_score(true_labels, pred_labels, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(true_labels, pred_labels).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    test_df['prediction'] = pred_labels
    test_df['confidence'] = all_preds.numpy()

    # test_df['domain'] = test_df['original_domain']
    # test_df = test_df.drop(columns=['original_domain'])

    fp_df = test_df[(test_df['label'] == 0) & (test_df['prediction'] == 1)].copy()
    fp_df['error_type'] = 'FP'

    fn_df = test_df[(test_df['label'] == 1) & (test_df['prediction'] == 0)].copy()
    fn_df['error_type'] = 'FN'

    errors_df = pd.concat([fp_df, fn_df], ignore_index=True)
    errors_save_path = path_results.joinpath(f'./Errors_List_{count}_{rank}_MIT.csv')
    errors_df.to_csv(errors_save_path, index=False)
    print(errors_df)

    metrics_data = {
        'Count_Threshold': [count],
        'Rank_Threshold': [rank],
        'Accuracy': [accuracy],
        'Precision': [precision],
        'Recall (TPR)': [recall],
        'F1_Score': [f1],
        'FPR': [fpr],
        'FNR': [fnr],
        'TP': [tp],
        'TN': [tn],
        'FP': [fp],
        'FN': [fn]
    }
    metrics_df = pd.DataFrame(metrics_data)
    metrics_save_path = path_results.joinpath(f'./Metrics_Summary_{count}_{rank}_MIT.csv')
    metrics_df.to_csv(metrics_save_path, index=False)
        
    print(f"\n[Test Summary] Test Time: {test_time:.2f}s | Total Data: {len(test_df)}")
    print(f"Acc: {accuracy:.4f} | F1: {f1:.4f} | Prec: {precision:.4f} | Rec: {recall:.4f}")
    print(f"Total Errors: {len(errors_df)} (FP: {fp}, FN: {fn})")
    print(f"FPR: {fpr:.4f} | FNR: {fnr:.4f}")