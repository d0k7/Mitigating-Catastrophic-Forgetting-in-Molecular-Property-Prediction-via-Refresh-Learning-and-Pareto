import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import RobertaForSequenceClassification, RobertaTokenizer, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import numpy as np

# Define paths
bbbp_path = '/scratch/sakshi.rs.cse21.itbhu/dheeraj/New_RL_Paper_2/Datasets/BBBP.csv'
bitter_path = '/scratch/sakshi.rs.cse21.itbhu/dheeraj/New_RL_Paper_2/Datasets/Bitter.csv'
sweet_path = '/scratch/sakshi.rs.cse21.itbhu/dheeraj/New_RL_Paper_2/Datasets/Sweet.csv'
tokenizer_path = '/scratch/sakshi.rs.cse21.itbhu/dheeraj/New_RL_Paper_2/seyonec_ChemBERTa_zinc_base_v1'
model_path = '/scratch/sakshi.rs.cse21.itbhu/dheeraj/New_RL_Paper_2/seyonec_ChemBERTa_zinc_base_v1'

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

# Dark Experience Replay Functions
def dark_experience_replay(buffer, model, optimizer, scheduler, noise_factor=0.005):
    model.train()
    total_loss = 0
    for batch in buffer:
        optimizer.zero_grad()
        input_ids, attention_mask, labels = batch
        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = output.loss
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
    
    # Adding noise to the model parameters
    for name, param in model.named_parameters():
        noise = torch.normal(0, noise_factor, size=param.size()).to(param.device)
        param.data += noise
    
    avg_loss = total_loss / len(buffer)
    return avg_loss

def der_train(model, train_loader, val_loader, buffer, optimizer, scheduler, epochs=1):
    best_val_loss = float('inf')
    early_stopping_counter = 0
    early_stopping_patience = 2
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        all_preds, all_labels = [], []
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids, attention_mask, labels = batch
            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = output.loss
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            buffer.append(batch)
            preds = torch.argmax(output.logits, dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())
        
        avg_train_loss = total_loss / len(train_loader)
        train_accuracy = accuracy_score(all_labels, all_preds)
        train_auc = roc_auc_score(all_labels, all_preds)
        print(f'Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss}, Accuracy: {train_accuracy}, AUC: {train_auc}')
        print(f'Classification Report for Training:\n{classification_report(all_labels, all_preds)}')

        val_loss, val_accuracy, val_auc, val_report = evaluate(model, val_loader, return_loss=True)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print("Early stopping")
                break

        # Apply Dark Experience Replay
        der_loss = dark_experience_replay(buffer, model, optimizer, scheduler)
        print(f"Dark Experience Replay Loss: {der_loss}")
    
    return model

# Dark Experience Replay with Refresh Learning
def dark_experience_replay_with_refresh_learning(dataset_name, train_loader, val_loader, test_loader, epochs=3):
    print(f"Starting DER++ + Refresh Learning for dataset: {dataset_name}")
    model = RobertaForSequenceClassification.from_pretrained(model_path, num_labels=2)
    optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)  # Lowered learning rate
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    buffer = []
    best_val_loss = float('inf')
    early_stopping_counter = 0
    early_stopping_patience = 3  # Increased patience for early stopping
    
    for k in range(epochs):
        # Indicate the start of DER++ training
        print(f'--- Starting DER++ Training on {dataset_name}, Epoch {k+1}/{epochs} ---')
        model = der_train(model, train_loader, val_loader, buffer, optimizer, scheduler, epochs=1)
        
        # Indicate the start of DER++ + Refresh Learning (Unlearn steps)
        print(f'--- Starting DER++ + Refresh Learning on {dataset_name}, Epoch {k+1}/{epochs} ---')
        model.train()
        for j in range(3):  # Unlearn steps
            total_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                input_ids, attention_mask, labels = batch
                output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = output.loss
                total_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
            print(f"Unlearn Step {j+1}, Loss: {total_loss / len(train_loader)}")
        
        # Adding noise to the model parameters
        for name, param in model.named_parameters():
            noise = torch.normal(0, 0.005, size=param.size()).to(param.device)
            param.data += noise
        
        # Validation after DER++ and Refresh Learning
        val_loss, val_accuracy, val_auc, val_report = evaluate(model, val_loader, return_loss=True)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print("Early stopping")
                break
        print(f"Validation Loss after DER++ + Refresh Learning: {val_loss}, Accuracy: {val_accuracy}, AUC: {val_auc}")
        print(f"Classification Report for Validation:\n{val_report}")
        
        # Test the model
        test_loss, test_accuracy, test_auc, test_report = evaluate(model, test_loader, return_loss=True)
        print(f"Test Loss: {test_loss}, Accuracy: {test_accuracy}, AUC: {test_auc}")
        print(f"Classification Report for Test:\n{test_report}")

# Evaluation Function
def evaluate(model, loader, return_loss=False):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids, attention_mask, labels = batch
            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = output.loss
            total_loss += loss.item()
            preds = torch.argmax(output.logits, dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())
    accuracy = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds)
    avg_loss = total_loss / len(loader) if return_loss else None
    return (avg_loss, accuracy, auc, report) if return_loss else (accuracy, auc, report)

# Start training and evaluating on datasets
datasets = [
    ('BBBP', bbbp_train_loader, bbbp_val_loader, bbbp_test_loader),
    ('Bitter', bitter_train_loader, bitter_val_loader, bitter_test_loader),
    ('Sweet', sweet_train_loader, sweet_val_loader, sweet_test_loader),
]

for dataset_name, train_loader, val_loader, test_loader in datasets:
    dark_experience_replay_with_refresh_learning(dataset_name, train_loader, val_loader, test_loader)
