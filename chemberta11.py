import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import RobertaForSequenceClassification, RobertaTokenizer, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import numpy as np
import os

# Device Configuration (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define paths
bbbp_path = '/scratch/sakshi.rs.cse21.itbhu/dheeraj/New_RL_Paper_2/Datasets/BBBP.csv'
bitter_path = '/scratch/sakshi.rs.cse21.itbhu/dheeraj/New_RL_Paper_2/Datasets/Bitter.csv'
sweet_path = '/scratch/sakshi.rs.cse21.itbhu/dheeraj/New_RL_Paper_2/Datasets/Sweet.csv'
tokenizer_path = '/scratch/sakshi.rs.cse21.itbhu/dheeraj/New_RL_Paper_2/seyonec_ChemBERTa_zinc_base_v1'
model_path = '/scratch/sakshi.rs.cse21.itbhu/dheeraj/New_RL_Paper_2/seyonec_ChemBERTa_zinc_base_v1'
save_model_path = '/scratch/sakshi.rs.cse21.itbhu/dheeraj/New_RL_Paper_2/DIL Saved Model'

# Load and prepare data
def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    inputs = df['SMILES'].tolist()
    labels = df['Label'].tolist()
    labels = [int(label) for label in labels]
    return inputs, labels

bbbp_inputs, bbbp_labels = load_and_prepare_data(bbbp_path)
bitter_inputs, bitter_labels = load_and_prepare_data(bitter_path)
sweet_inputs, sweet_labels = load_and_prepare_data(sweet_path)

# Tokenize data
tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path)
max_len = 50

def tokenize(inputs):
    return tokenizer(inputs, padding='max_length', truncation=True, max_length=max_len, return_tensors='pt')

bbbp_encoded = tokenize(bbbp_inputs)
bitter_encoded = tokenize(bitter_inputs)
sweet_encoded = tokenize(sweet_inputs)

# Split data
def split_data(encoded_inputs, labels):
    train_indices, test_indices, train_labels, test_labels = train_test_split(range(len(labels)), labels, test_size=0.2, random_state=42)
    val_indices, test_indices, val_labels, test_labels = train_test_split(test_indices, test_labels, test_size=0.5, random_state=42)
    return train_indices, val_indices, test_indices, train_labels, val_labels, test_labels

bbbp_train_idx, bbbp_val_idx, bbbp_test_idx, bbbp_train_lbl, bbbp_val_lbl, bbbp_test_lbl = split_data(bbbp_encoded, bbbp_labels)
bitter_train_idx, bitter_val_idx, bitter_test_idx, bitter_train_lbl, bitter_val_lbl, bitter_test_lbl = split_data(bitter_encoded, bitter_labels)
sweet_train_idx, sweet_val_idx, sweet_test_idx, sweet_train_lbl, sweet_val_lbl, sweet_test_lbl = split_data(sweet_encoded, sweet_labels)

# Create DataLoaders
def create_tensor_dataset(encoded_inputs, indices, labels):
    return TensorDataset(encoded_inputs['input_ids'][indices], encoded_inputs['attention_mask'][indices], torch.tensor(labels))

def create_data_loader(tensor_dataset, batch_size=16):
    return DataLoader(tensor_dataset, batch_size=batch_size, shuffle=True)

bbbp_train_loader = create_data_loader(create_tensor_dataset(bbbp_encoded, bbbp_train_idx, bbbp_train_lbl))
bbbp_val_loader = create_data_loader(create_tensor_dataset(bbbp_encoded, bbbp_val_idx, bbbp_val_lbl), batch_size=32)
bbbp_test_loader = create_data_loader(create_tensor_dataset(bbbp_encoded, bbbp_test_idx, bbbp_test_lbl), batch_size=32)

bitter_train_loader = create_data_loader(create_tensor_dataset(bitter_encoded, bitter_train_idx, bitter_train_lbl))
bitter_val_loader = create_data_loader(create_tensor_dataset(bitter_encoded, bitter_val_idx, bitter_val_lbl), batch_size=32)
bitter_test_loader = create_data_loader(create_tensor_dataset(bitter_encoded, bitter_test_idx, bitter_test_lbl), batch_size=32)

sweet_train_loader = create_data_loader(create_tensor_dataset(sweet_encoded, sweet_train_idx, sweet_train_lbl))
sweet_val_loader = create_data_loader(create_tensor_dataset(sweet_encoded, sweet_val_idx, sweet_val_lbl), batch_size=32)
sweet_test_loader = create_data_loader(create_tensor_dataset(sweet_encoded, sweet_test_idx, sweet_test_lbl), batch_size=32)

# ER and ER + Refresh Learning Functions
def er_loss(replay_buffer, model, criterion, lambda_er=0.01):
    if len(replay_buffer) == 0:
        return torch.tensor(0.0, device=device)  # Avoid division by zero by returning 0 when the buffer is empty
    
    total_loss = 0
    for batch in replay_buffer:
        input_ids, attention_mask, labels = batch
        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        total_loss += criterion(output.logits, labels)
    return lambda_er * total_loss / len(replay_buffer)

# Evaluation Function
def evaluate(model, val_loader, return_loss=False):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = output.loss
            total_loss += loss.item()
            preds = torch.argmax(output.logits, dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())
    accuracy = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds)
    
    if return_loss:
        return total_loss / len(val_loader), accuracy, auc, report
    else:
        print(f'Validation Accuracy: {accuracy}, AUC: {auc}')
        print(f'Classification Report:\n{report}')

# ER Training Function
def train_er(model, train_loader, val_loader, replay_buffer, optimizer, scheduler, criterion, epochs=3, lambda_er=0.01):
    best_val_loss = float('inf')
    early_stopping_counter = 0
    early_stopping_patience = 2

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        all_preds, all_labels = [], []
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = output.loss + er_loss(replay_buffer, model, criterion, lambda_er)
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            preds = torch.argmax(output.logits, dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())

        avg_train_loss = total_loss / len(train_loader)
        train_accuracy = accuracy_score(all_labels, all_preds)
        train_auc = roc_auc_score(all_labels, all_preds)
        print(f'Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss}, Accuracy: {train_accuracy}, AUC: {train_auc}')
        print(f'Classification Report for Training:\n{classification_report(all_labels, all_preds)}')

        val_loss, val_accuracy, val_auc, val_report = evaluate(model, val_loader, return_loss=True)
        print(f'Validation Loss: {val_loss}, Accuracy: {val_accuracy}, AUC: {val_auc}')
        print(f'Validation Report:\n{val_report}')

        # Early Stopping Check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            # Save the model here (if desired)
            torch.save(model.state_dict(), os.path.join(save_model_path, 'best_er_model.pth'))
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print("Early stopping triggered!")
                break

# ER + Refresh Learning
def refresh_learning_with_er(dataset_name, train_loader, val_loader, test_loader):
    print(f'--- Training ER on {dataset_name} ---')
    
    # Initialize model
    model = RobertaForSequenceClassification.from_pretrained(model_path, num_labels=2)
    model.to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    total_steps = len(train_loader) * 3  # Assuming 3 epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    replay_buffer = []  # Replay buffer for ER
    
    # Train ER model
    train_er(model, train_loader, val_loader, replay_buffer, optimizer, scheduler, criterion, epochs=3)

    # Testing phase
    print(f'--- Testing ER on {dataset_name} ---')
    evaluate(model, test_loader)

# Output Format Function
def run_all_experiments():
    print("Section 1: ER")
    datasets = [("BBBP", bbbp_train_loader, bbbp_val_loader, bbbp_test_loader),
                ("Bitter", bitter_train_loader, bitter_val_loader, bitter_test_loader),
                ("Sweet", sweet_train_loader, sweet_val_loader, sweet_test_loader)]
    
    for dataset_name, train_loader, val_loader, test_loader in datasets:
        print(f'\n1) Train on {dataset_name} Dataset')
        refresh_learning_with_er(dataset_name, train_loader, val_loader, test_loader)

    print("\nSection 2: ER + Refresh Learning")
    for dataset_name, train_loader, val_loader, test_loader in datasets:
        print(f'\n1) Train on {dataset_name} Dataset (ER + Refresh Learning)')
        refresh_learning_with_er(dataset_name, train_loader, val_loader, test_loader)

# Call the function to execute experiments
run_all_experiments()
