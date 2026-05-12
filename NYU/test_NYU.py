import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import numpy as np
import time
from dataset_processor_NYU import DatasetProcessor
from torch.utils.data import DataLoader
from NYU_model import NYU_DGA
from tqdm import tqdm
from sklearn.metrics import roc_curve
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utility.path import path_train_data, path_test_data, path_artifacts, path_images, path_results

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NYU_DGA().to(device)

    batch_size = 100
    target_cols = ['domain', 'label']

    val_files = [path_train_data.joinpath('T24_benign_val.parquet'), path_train_data.joinpath('T24_dga_sampled_val.parquet')]
    val_df = pd.concat([pd.read_parquet(f, columns=target_cols) for f in val_files]).reset_index(drop=True)
    val_dataset   = DatasetProcessor(val_df)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model.load_state_dict(torch.load(path_artifacts.joinpath('NYU_T24.pt')))
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
    test_months = ['tranco_20250115','tranco_20250212','tranco_20250314','tranco_20250412','tranco_20250518', 'tranco_20250613',
    'tranco_20250718','tranco_20250815','tranco_20250915','tranco_20251013','tranco_20251111','tranco_20251214']
    
    n_splits = 25  # 데이터를 n등분

    boxplot_data = [] # 박스 플롯을 그리기 위해 데이터를 모을 리스트

    for m_test in test_months:
        b_test_path = path_test_data.joinpath(f'{m_test}.csv')

        test_df = pd.read_csv(
            b_test_path, 
            names=['rank','domain'],  # 파일의 실제 형태에 맞춤
            header=None
        )

        test_df = test_df.sort_values(by='rank', ascending=True).reset_index(drop=True)

        # test_df = test_df.drop(columns=['rank'])
        test_df['label'] = 0

        test_dataset = DatasetProcessor(test_df)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

        all_preds = []
        start_time = time.time()
        
        # 모델 추론
        model.eval()
        with torch.no_grad():
            for X_test, y in tqdm(test_loader, desc=f"[{m_test} Test]", leave=False):
                X_test = X_test.to(device)
                outputs = model(X_test)
                all_preds.append(outputs.squeeze().cpu())
                
        test_time = time.time() - start_time
        all_preds = torch.cat(all_preds)
        
        pred_labels = (all_preds > theta).int().numpy()

        test_df['prediction'] = pred_labels
        test_df['confidence'] = all_preds.numpy()
        
        # n등분하여 분할
        chunks = np.array_split(pred_labels, n_splits)

        chunk_indices = np.concatenate([np.full(len(chunk), i + 1) for i, chunk in enumerate(chunks)])
        test_df['chunk_index'] = chunk_indices

        fp_month_df = test_df[test_df['prediction'] == 1].copy()
        month_save_path = path_results.joinpath(f'./FP_List_{m_test}_NYU.csv') # 저장 경로
        fp_month_df.to_csv(month_save_path, index=False)
        
        # 각 청크(Chunk)별로 FP 및 FPR 계산
        for i, chunk in enumerate(chunks):
            chunk_size = len(chunk)
            if chunk_size == 0:
                continue
                
            # 1이면 모두 FP
            fp = chunk.sum()
            fpr = fp / chunk_size
            
            boxplot_data.append({
                'Month': m_test,
                'Chunk_Index': i + 1,
                'Chunk_Size': chunk_size,
                'FP_Count': fp,
                'FPR': fpr
            })
            
        # 전체 월에 대한 단순 요약 출력
        total_fp = pred_labels.sum()
        total_fpr = total_fp / len(pred_labels)
        print(f"[Month: {m_test}] Test Time: {test_time:.2f}s | Total Data: {len(pred_labels)} | Total FP: {total_fp} | Overall FPR: {total_fpr:.4f}")

    results_df = pd.DataFrame(boxplot_data)

    print("\n=== Chunk별 FPR 결과 요약 ===")
    print(results_df.head())

    # 박스플랏
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(12, 6))

    sns.boxplot(
        x='Chunk_Index', y='FPR', 
        data=results_df, 
        palette='viridis'
    )

    sns.stripplot(
        x='Chunk_Index', y='FPR', 
        data=results_df, 
        hue='Month',      # 월별로 다른 색상 부여
        palette='Set1',
        size=6, 
        alpha=0.8, 
        jitter=True, 
        dodge=True        # 겹치지 않게 살짝 흩뿌림
    )

    plt.title('FPR Distribution per Rank Chunk Across All Months', fontsize=15, fontweight='bold')
    plt.xlabel('Rank Chunk Index', fontsize=12)
    plt.ylabel('False Positive Rate (FPR)', fontsize=12)

    plt.legend(title='Month', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    plt.savefig(path_images.joinpath(f'boxplot_chunk_{n_splits}_NYU.png'), bbox_inches='tight')
    plt.show()


    # 라인플랏 (평균)
    plt.figure(figsize=(10, 5))

    sns.lineplot(
        x='Chunk_Index', y='FPR', 
        data=results_df, 
        color='royalblue',
        linewidth=2,
        marker='o'
    )

    plt.title('Average FPR Trend with 95% Confidence Interval', fontsize=14)
    plt.xlabel('Rank Chunk Index', fontsize=12)
    plt.ylabel('FPR', fontsize=12)
    plt.xticks(range(1, n_splits + 1))
    plt.tight_layout()
    plt.savefig(path_images.joinpath(f'lineplot_avg_chunk_{n_splits}_NYU.png'), bbox_inches='tight')
    plt.show()