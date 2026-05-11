import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import DGA_Dataset
from torch.utils.data import DataLoader
import datetime
import os
import argparse
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, precision_score, recall_score
import wandb
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utility.path import path_data, path_artifacts

if __name__ == "__main__":

    learning_rate = 1e-4
    batch_size = 100
    epochs = 100 
    best_filename='MIT_T24'

    # wandb
    wandb.init(project="2026FMLDS", name=best_filename,
        config={
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
            "model_architecture": "MIT",
            "benign": "24",
            "dga": "24",
        }, mode='online')


    benign_train_path = path_data.joinpath("T24_benign_train.parquet")
    benign_val_path = path_data.joinpath("T24_benign_val.parquet")
    dga_train_path = path_data.joinpath("T24_dga_sampled_train.parquet")
    dga_val_path = path_data.joinpath("T24_dga_sampled_val.parquet")

    benign_train = pd.read_parquet(benign_train_path)
    benign_val = pd.read_parquet(benign_val_path)
    dga_train = pd.read_parquet(dga_train_path)
    dga_val = pd.read_parquet(dga_val_path)

    train = pd.concat([benign_train, dga_train])[['domain', 'label']].reset_index(drop=True)
    val = pd.concat([benign_val, dga_val])[['domain', 'label']].reset_index(drop=True)

    def process_domain(domain):
        # 소문자 변환 및 뒤에서 75자만 자르기
        domain = str(domain).lower()[-75:]
        
        # ASCII 정수로 변환
        indices = [ord(c) for c in domain]
        
        # 0으로 패딩 (75자리 맞추기)
        pad_len = 75 - len(indices)
        return [0] * pad_len + indices

    train["domain"] = train["domain"].apply(process_domain)
    val["domain"] = val["domain"].apply(process_domain)

    # 각 데이터셋의 크기 확인
    print(f"훈련 데이터셋 크기: {len(train)}")
    print(f"검증 데이터셋 크기: {len(val)}")

    # 모델 초기화
    from MIT_model import MIT
    device = torch.device('cuda')
    model = MIT().to(device)

    # hyperparameters
    criterion = nn.BCELoss() # Binary Cross-Entropy Loss 사용 (paper1)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) # Adam 옵티마이저 사용 (paper1)

    # DataLoader 생성
    train_dataset = DGA_Dataset(train)
    val_dataset = DGA_Dataset(val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True) # 셔플을 켜고 시드를 주면 같은 파라미터에 대해 실험 결과 고정 # train만 True!!!
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True) # False!!

    best_val_loss = float('inf') # initial loss
    best_epoch = 0

    # 학습 및 검증 루프
    for epoch in range(epochs):
        # 학습 단계=======================================================================================
        model.train() # weight&bias를 update할 수 있는 상태로 변환
        total_loss = 0.0 # loss 초기화
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False):
            inputs, labels = inputs.to(device), labels.to(device).float()
            
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels.float().view_as(outputs))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)

        # 검증 단계=======================================================================================
        model.eval()
        val_loss = 0.0
        val_outputs = []
        val_labels = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]", leave=False): # 배치 루프
                inputs, labels = inputs.to(device), labels.to(device).float()

                outputs = model(inputs)
                val_loss += criterion(outputs, labels.float().view_as(outputs)).item()
                val_outputs.append(outputs.squeeze().cpu())
                val_labels.append(labels.squeeze().cpu())
        
        avg_val_loss = val_loss / len(val_loader)
                
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            model_path_loss = path_artifacts.joinpath(f'{best_filename}.pt')
            torch.save(model.state_dict(), model_path_loss)
            print(f"Model saved to {model_path_loss}")

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

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss, 
            "val_loss": avg_val_loss,
            "val_accuracy": accuracy,
            "val_precision": precision,
            "val_recall": recall,
            "val_f1": f1,
        })

        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val_ACC: {accuracy:.4f}, Val_precision: {precision:.4f}, Val_recall: {recall:.4f}, Val_F1: {f1:.4f}, Optimal Threshold: {theta:.4f}")

    wandb.summary["best_epoch"] = best_epoch
    artifact = wandb.Artifact(name=best_filename, type="model")
    artifact.add_file(model_path_loss)
    wandb.log_artifact(artifact)
    wandb.finish()