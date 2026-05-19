import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import DGA_Dataset
from torch.utils.data import DataLoader
from MIT_model import MIT
from sklearn.metrics import roc_curve, f1_score, precision_score, recall_score
import wandb
import time
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--count', type=int, default=1, help='Unique count threshold')
    parser.add_argument('--rank', type=int, default=1000000, help='Rank threshold')
    args = parser.parse_args()

    count = args.count
    rank = args.rank

    best_filename=f'MIT_T24_{count}_{rank}'
    batch_size = 100
    epochs = 50
    learning_rate = 1e-4

    # wandb
    wandb.init(project="2026FMLDS", name=best_filename,
        config={
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
            "model_architecture": "MIT",
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

    # DataLoader 생성
    train_dataset = DGA_Dataset(train_df)
    val_dataset = DGA_Dataset(val_df)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 모델 세팅
    device = torch.device('cuda')
    model = MIT().to(device)
    criterion = nn.BCELoss() 
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) 

    best_val_loss = float('inf')
    best_epoch = 0

    # 학습 및 검증 루프
    for epoch in range(epochs):
        start_time = time.time()

        # 학습 단계
        model.train()
        total_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False):
            inputs, labels = inputs.to(device), labels.to(device).float()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.float().view_as(outputs))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)

        # 검증 단계
        model.eval()
        val_loss = 0.0
        val_outputs = []
        val_labels = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]", leave=False):
                inputs, labels = inputs.to(device), labels.to(device).float()
                outputs = model(inputs)
                val_loss += criterion(outputs, labels.float().view_as(outputs)).item()
                val_outputs.append(outputs.detach().cpu().view(-1))
                val_labels.append(labels.detach().cpu().view(-1))
        avg_val_loss = val_loss / len(val_loader)
                
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            model_path_loss = path_artifacts.joinpath(f'{best_filename}.pt')
            torch.save(model.state_dict(), model_path_loss)

        val_outputs = torch.cat(val_outputs).view(-1)
        val_labels = torch.cat(val_labels).view(-1)
                
        fpr, tpr, threshold = roc_curve(val_labels.numpy(), val_outputs.numpy())
        theta = threshold[(tpr-fpr).argmax()]

        pred_labels = (val_outputs > theta).int()
        true_labels = val_labels.int()

        accuracy = (pred_labels == true_labels).sum().item() / len(true_labels)
        precision = precision_score(true_labels.numpy(), pred_labels.numpy(), zero_division=0)
        recall = recall_score(true_labels.numpy(), pred_labels.numpy(), zero_division=0)
        f1 = f1_score(true_labels.numpy(), pred_labels.numpy(), zero_division=0)

        end_time = time.time()
        epoch_time = end_time - start_time

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss, 
            "val_loss": avg_val_loss,
            "val_accuracy": accuracy,
            "val_precision": precision,
            "val_recall": recall,
            "val_f1": f1,
        })

        print(f"Epoch {epoch+1} [Time: {epoch_time:.2f}s]: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val_ACC: {accuracy:.4f}, Val_precision: {precision:.4f}, Val_recall: {recall:.4f}, Val_F1: {f1:.4f}, Optimal Threshold: {theta:.4f}")

    wandb.summary["best_epoch"] = best_epoch
    artifact = wandb.Artifact(name=best_filename, type="model")
    artifact.add_file(model_path_loss)
    wandb.log_artifact(artifact)
    wandb.finish()