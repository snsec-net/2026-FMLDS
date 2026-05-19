import pandas as pd
import numpy as np
import seaborn as sns
import torch
import time
from dataset_processor_HMT import DatasetProcessor, BigramDictionaryBuilder
from torch.utils.data import DataLoader
from HMT_model import HybridModifiedTransformer
from tqdm import tqdm
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utility.path import path_train_data, path_test_data, path_artifacts, path_images, path_results
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def cutting_df(df, count, rank) :
    df = df[
        (df['day_count'] >= count) & 
        (df['avg_rank'] <= rank)
    ].reset_index(drop=True)
    return df

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--count', type=int, default=1, help='Unique count threshold')
    parser.add_argument('--rank', type=int, default=1000000, help='Rank threshold')
    args = parser.parse_args()

    count = args.count
    rank = args.rank

    num_classes = 1
    label = 'label'
    n_t_layers = 6
    batch_size = 100
    target_cols = ['domain', 'label']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridModifiedTransformer(num_classes=num_classes,n_transformer_layers=n_t_layers).to(device)

    dga_train_df = pd.read_parquet(path_train_data.joinpath('T24_dga_1day_train.parquet'), columns=target_cols)
    dga_val_df = pd.read_parquet(path_train_data.joinpath('T24_dga_1day_val.parquet'), columns=target_cols)

    benign_df = pd.read_parquet(path_train_data.joinpath('T24_benign_1day.parquet'))

    benign_df = cutting_df(benign_df, count, rank)
    benign_df = benign_df[target_cols]

    benign_train_df, benign_val_df = train_test_split(benign_df, test_size=0.1, random_state=42)

    train_df = pd.concat([benign_train_df, dga_train_df]).reset_index(drop=True)
    val_df = pd.concat([benign_val_df, dga_val_df]).reset_index(drop=True)
    all_df = pd.concat([train_df, val_df])

    dic_builder = BigramDictionaryBuilder(all_df)
    bigram_dict = dic_builder.build_dictionary()
    print('dictionary built')

    val_dataset   = DatasetProcessor(val_df, label_col=label, bigram_to_index=bigram_dict)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model.load_state_dict(torch.load(path_artifacts.joinpath(f'HMT_T24_{count}_{rank}.pt')))
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # Treshold 찾기 (binary classification)
    val_output = []
    val_labels = []

    with torch.no_grad():
        for (char_X, bigram_X, y) in tqdm(val_loader, desc=f"[Threshold]", leave=False):
            char_X, bigram_X = char_X.to(device), bigram_X.to(device)
            y = y.to(device)
            outputs = model(char_X, bigram_X)
            probabilities = torch.sigmoid(outputs.squeeze(dim=1)).cpu()
            val_output.append(probabilities)
            val_labels.append(y.cpu())

    val_output = torch.cat(val_output)
    val_labels = torch.cat(val_labels)
    fpr, tpr, thresholds = roc_curve(val_labels, val_output)
    theta = thresholds[(tpr-fpr).argmax()]
    print(f'Threshold: {theta: .4f}\n')

    # 테스트
    dga_test_df = pd.read_parquet(path_test_data.joinpath('T25-26_dga_1day.parquet'),columns=target_cols)

    benign_test_df = pd.read_parquet(path_test_data.joinpath('T25-26_benign_1day.parquet'))
    benign_test_df = cutting_df(benign_test_df, count, rank)
    benign_test_df = benign_test_df[target_cols]

    test_df = pd.concat([benign_test_df, dga_test_df]).reset_index(drop=True)

    test_dataset  = DatasetProcessor(test_df,label_col=label, bigram_to_index=bigram_dict)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    all_preds = []
    all_labels = []
    start_time = time.time()

    with torch.no_grad():
        for (char_X, bigram_X, y) in tqdm(test_loader, desc="[Test]", leave=False):
            char_X, bigram_X = char_X.to(device), bigram_X.to(device)
            y = y.to(device)
            outputs = model(char_X, bigram_X)
            probabilities = torch.sigmoid(outputs.squeeze(dim=1)).cpu()
            all_preds.append(probabilities)
            all_labels.append(y.squeeze().cpu())

    test_time = time.time() - start_time
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    pred_labels = (all_preds > theta).int().numpy()

    accuracy = (pred_labels == all_labels).sum().item() / len(all_labels)
    precision = precision_score(all_labels, pred_labels, zero_division=0)
    recall = recall_score(all_labels, pred_labels, zero_division=0)
    f1 = f1_score(all_labels, pred_labels, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(all_labels, pred_labels).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    test_df['prediction'] = pred_labels
    test_df['confidence'] = all_preds.numpy()

    fp_df = test_df[(test_df['label'] == 0) & (test_df['prediction'] == 1)].copy()
    fp_df['error_type'] = 'FP'

    fn_df = test_df[(test_df['label'] == 1) & (test_df['prediction'] == 0)].copy()
    fn_df['error_type'] = 'FN'

    errors_df = pd.concat([fp_df, fn_df], ignore_index=True)
    errors_save_path = path_results.joinpath(f'./Errors_List_{count}_{rank}_HMT.csv')
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
    metrics_save_path = path_results.joinpath(f'./Metrics_Summary_{count}_{rank}_HMT.csv')
    metrics_df.to_csv(metrics_save_path, index=False)
        
    print(f"\n[Test Summary] Test Time: {test_time:.2f}s | Total Data: {len(test_df)}")
    print(f"Acc: {accuracy:.4f} | F1: {f1:.4f} | Prec: {precision:.4f} | Rec: {recall:.4f}")
    print(f"Total Errors: {len(errors_df)} (FP: {fp}, FN: {fn})")
    print(f"FPR: {fpr:.4f} | FNR: {fnr:.4f}")