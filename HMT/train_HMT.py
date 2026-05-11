import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset_processor_HMT import DatasetProcessor, BigramDictionaryBuilder
from HMT_model import HybridModifiedTransformer
from tqdm import tqdm
import time
import datetime
import wandb
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve
import random
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utility.path import path_data, path_artifacts


if __name__ == '__main__':
    def set_seed(seed: int=42) :
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    num_epochs = 100
    num_classes = 1
    label = 'label'
    n_t_layers = 6
    learning_rate = 1e-4
    batch_size = 128
    best_filename = "HMT_T24"

    wandb.init(project='2026FMLDS', name=best_filename,
            config={
            "learning_rate": learning_rate,
            "epochs": num_epochs,
            "batch_size": batch_size,
            "transformer_layer" : n_t_layers,
            "model_architecture": "HMT",
            "benign": "24",
            "dga": "24",
        }, mode='online')

    target_cols = ['domain', 'label']

    train_files = [
        path_data.joinpath('T24_benign_train.parquet'), 
        path_data.joinpath('T24_dga_sampled_train.parquet')
    ]
    val_files = [
        path_data.joinpath('T24_benign_val.parquet'), 
        path_data.joinpath('T24_dga_sampled_val.parquet')
    ]

    train_df = pd.concat([pd.read_parquet(f, columns=target_cols) for f in train_files]).reset_index(drop=True)
    val_df = pd.concat([pd.read_parquet(f, columns=target_cols) for f in val_files]).reset_index(drop=True)

    all_df = pd.concat([train_df, val_df])

    dic_builder = BigramDictionaryBuilder(all_df)
    bigram_dict = dic_builder.build_dictionary()
    print('dictionary built')

    train_dataset = DatasetProcessor(train_df, label_col=label, bigram_to_index=bigram_dict)
    val_dataset  = DatasetProcessor(val_df, label_col=label, bigram_to_index=bigram_dict)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 모델 구성
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridModifiedTransformer(num_classes=num_classes,n_transformer_layers=n_t_layers).to(device)
    criterion_binary = nn.BCEWithLogitsLoss()
    criterion_multi = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) 

    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(num_epochs) :
        start_time = time.time()

        model.train()
        train_loss = 0.0

        train_loop = tqdm(
            train_loader, 
            total=len(train_loader), 
            desc=f"Epoch {epoch+1}/{num_epochs} [TRAIN]", leave=False
        )

        for idx, (char_X, bigram_X, y) in enumerate(train_loop):
            char_X, bigram_X = char_X.to(device), bigram_X.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            outputs = model(char_X, bigram_X)
            if num_classes == 1 :
                loss = criterion_binary(outputs.squeeze(dim=1), y)
            else :
                loss = criterion_multi(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * char_X.size(0)

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        val_output = []
        val_labels = []
        
        val_loop = tqdm(
            val_loader,
            total=len(val_loader),
            desc=f"Epoch {epoch+1}/{num_epochs} [VAL]", leave=False
        )

        with torch.no_grad():
            for (char_X, bigram_X, y) in val_loop:
                char_X, bigram_X = char_X.to(device), bigram_X.to(device)
                y = y.to(device)
                outputs = model(char_X, bigram_X)
                if num_classes == 1 :
                    loss = criterion_binary(outputs.squeeze(dim=1), y)
                    probabilities = torch.sigmoid(outputs.squeeze(dim=1)).cpu()
                else :
                    loss = criterion_multi(outputs.squeeze(dim=1), y)
                    probabilities = torch.softmax(outputs.squeeze(dim=1), dim=1).cpu()
                val_loss += loss.item() * char_X.size(0)
                val_output.append(probabilities)
                val_labels.append(y.cpu())

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            save_path = path_artifacts.joinpath(f'{best_filename}.pt')
            torch.save(model.state_dict(), save_path)

        val_output = torch.cat(val_output)
        val_labels = torch.cat(val_labels)

        log_data = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }

        if num_classes == 1 :
            fpr, tpr, thresholds = roc_curve(val_labels, val_output)
            theta = thresholds[(tpr-fpr).argmax()]
            pred_labels = (val_output > theta).int()
            true_labels = val_labels.int()

            accuracy = (pred_labels == true_labels).sum().item() / len(true_labels)
            precision = precision_score(true_labels.numpy(), pred_labels.numpy(), zero_division=0)
            recall = recall_score(true_labels.numpy(), pred_labels.numpy(), zero_division=0)
            f1 = f1_score(true_labels.numpy(), pred_labels.numpy(), zero_division=0)

            log_data.update({
                "val_accuracy": accuracy,
                "val_precision": precision,
                "val_recall": recall,
                "val_f1": f1,
            })

            print_metrics = (f"Val_ACC: {accuracy:.4f}, Val_precision: {precision:.4f}, Val_recall: {recall:.4f}, Val_f1: {f1:.4f}, "
                            f"Optimal Threshold: {theta:.4f}")
        else :
            pred_labels = torch.argmax(val_output, dim=1).numpy()
            true_labels = val_labels.squeeze().numpy()

            accuracy = (pred_labels == true_labels).sum() / len(true_labels)

            # 매크로 평균 (Macro Average)
            precision_macro = precision_score(true_labels, pred_labels, average='macro', zero_division=0)
            recall_macro = recall_score(true_labels, pred_labels, average='macro', zero_division=0)
            f1_macro = f1_score(true_labels, pred_labels, average='macro', zero_division=0)

            log_data.update({
                "val_accuracy": accuracy,
                "val_precision_macro": precision_macro,
                "val_recall_macro": recall_macro,
                "val_f1_macro": f1_macro
            })

            print_metrics = (f"Val_ACC: {accuracy:.4f}, "
                            f"Macro_P: {precision_macro:.4f}, Macro_R: {recall_macro:.4f}, Macro_F1: {f1_macro:.4f}")

        end_time = time.time()
        epoch_time = end_time - start_time

        wandb.log(log_data)

        print(f"Epoch {epoch+1} [Time: {epoch_time:.2f}s]: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, {print_metrics}")

    wandb.summary["best_epoch"] = best_epoch

    artifact = wandb.Artifact(name=best_filename, type="model")
    artifact.add_file(save_path)
    wandb.log_artifact(artifact)

    wandb.finish()