import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset_processor_NYU import DatasetProcessor
from NYU_model import NYU_DGA
from tqdm import tqdm
import time
import wandb
from sklearn.metrics import roc_curve, precision_score, recall_score, f1_score
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utility.path import path_train_data, path_artifacts
import argparse
from sklearn.model_selection import train_test_split

def cutting_df(df, count, rank) :
    df = df[~(
        (df['day_count'] <= count) &
        (df['avg_rank'] >= rank)
    )].reset_index(drop=True)
    return df

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--count', type=int, default=1, help='Unique count threshold')
    parser.add_argument('--rank', type=int, default=1000000, help='Rank threshold')
    args = parser.parse_args()

    count = args.count
    rank = args.rank

    best_filename = f'NYU_T24_{count}_{rank}'
    num_epochs = 50
    batch_size = 100
    learning_rate = 1e-4

    wandb.init(project='2026FMLDS', name=best_filename,
            config={
            "learning_rate": learning_rate,
            "epochs": num_epochs,
            "batch_size": batch_size,
            "model_architecture": "NYU",
            "benign": "24 1day",
            "dga": "24 1day",
            "count": count,
            "rank": rank,
        }, mode='online')

    target_cols = ['domain', 'label']

    dga_train_df = pd.read_parquet(path_train_data.joinpath('T24_dga_1day_train.parquet'), columns=target_cols)
    dga_val_df = pd.read_parquet(path_train_data.joinpath('T24_dga_1day_val.parquet'), columns=target_cols)

    benign_df = pd.read_parquet(path_train_data.joinpath('T24_benign_1day.parquet'))

    benign_df = cutting_df(benign_df, count, rank)
    benign_df = benign_df[target_cols]

    benign_train_df, benign_val_df = train_test_split(benign_df, test_size=0.1, random_state=42)

    print(f"Benign train: {len(benign_train_df)}, Val: {len(benign_val_df)}")
    print(f"DGA train: {len(dga_train_df)}, Val: {len(dga_val_df)}")

    train_df = pd.concat([benign_train_df, dga_train_df]).reset_index(drop=True)
    val_df = pd.concat([benign_val_df, dga_val_df]).reset_index(drop=True)

    train_dataset = DatasetProcessor(train_df)
    val_dataset   = DatasetProcessor(val_df)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 모델 세팅
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NYU_DGA().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) 

    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(num_epochs):
        start_time = time.time()

        model.train()
        train_loss = 0.0
        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0) # 배치별 loss 총합의 합
        train_loss /= len(train_loader.dataset) # 모든 데이터에 대한 평균 loss
        
        # 검증
        model.eval()
        val_loss = 0.0
        val_output = []
        val_labels = []

        with torch.no_grad():
            for X_val, y_val in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]", leave=False):
                X_val, y_val = X_val.to(device), y_val.to(device)
                outputs = model(X_val)
                loss = criterion(outputs.squeeze(), y_val)
                val_loss += loss.item() * X_val.size(0)
                val_output.append(outputs.detach().cpu().view(-1))
                val_labels.append(y_val.detach().cpu().view(-1))
        val_loss /= len(val_loader.dataset)

        # 모델 저장
        if val_loss < best_val_loss :
            best_val_loss = val_loss
            best_epoch = epoch + 1
            save_path = path_artifacts.joinpath(f'{best_filename}.pt')
            torch.save(model.state_dict(), save_path)

        val_output = torch.cat(val_output)
        val_labels = torch.cat(val_labels)

        fpr, tpr, thresholds = roc_curve(val_labels, val_output)
        theta = thresholds[(tpr-fpr).argmax()]

        pred_labels = (val_output > theta).int()
        true_labels = val_labels.int()

        accuracy = (pred_labels == true_labels).sum().item() / len(true_labels)
        precision = precision_score(true_labels.numpy(), pred_labels.numpy(), zero_division=0)
        recall = recall_score(true_labels.numpy(), pred_labels.numpy(), zero_division=0)
        f1 = f1_score(true_labels.numpy(), pred_labels.numpy(), zero_division=0)

        end_time = time.time()
        epoch_time = end_time - start_time
        
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss, 
            "val_loss": val_loss,
            "val_accuracy": accuracy,
            "val_precision": precision,
            "val_recall": recall,
            "val_f1": f1,
        })

        print(f"Epoch {epoch+1} [Time: {epoch_time:.2f}s]: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val_ACC: {accuracy:.4f}, Val_precision: {precision:.4f}, Val_recall: {recall:.4f}, Val_F1: {f1:.4f}, Optimal Threshold: {theta:.4f}")

    wandb.summary["best_epoch"] = best_epoch
    artifact = wandb.Artifact(name=best_filename, type="model")
    artifact.add_file(save_path)
    wandb.log_artifact(artifact)
    wandb.finish()