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

# CPR and CPR + Refresh Learning Functions
def init_projection_matrix(model, num_classes=2):
    projection_matrix = {}
    for name, param in model.named_parameters():
        if "classifier" in name and param.ndim == 2:  # Assuming 2D parameters
            projection_matrix[name] = torch.eye(param.shape[1]).to(param.device)  # Shape: [input_dim, input_dim]
    return projection_matrix

def cpr_loss(model, projection_matrix, lambda_cpr=0.01):
    loss = 0
    for name, param in model.named_parameters():
        if name in projection_matrix:
            projected_weights = torch.matmul(param, projection_matrix[name].T)  # Use transposed matrix for correct shape
            loss += torch.norm(param - projected_weights, p=2)
    return lambda_cpr * loss

# CPR Training Function
def train_cpr(model, train_loader, val_loader, projection_matrix, optimizer, scheduler, epochs=3, lambda_cpr=0.01):
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
            loss = output.loss + cpr_loss(model, projection_matrix, lambda_cpr)
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
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print("Early stopping")
                break
    return model

# Refresh Learning with CPR
def refresh_learning_with_cpr(dataset_name, train_loader, val_loader, test_loaders, epochs=3, lambda_cpr=0.01):
    print(f"\nSection 2: CPR + Refresh Learning\n{'-'*50}")
    print(f"Starting CPR + Refresh Learning for dataset: {dataset_name}")
    model = RobertaForSequenceClassification.from_pretrained(model_path, num_labels=2)
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    projection_matrix = init_projection_matrix(model)
    best_val_loss = float('inf')
    early_stopping_counter = 0
    early_stopping_patience = 2
    
    for k in range(epochs):
        # Indicate the start of CPR training
        print(f'--- Starting CPR Training on {dataset_name}, Epoch {k+1}/{epochs} ---')
        model = train_cpr(model, train_loader, val_loader, projection_matrix, optimizer, scheduler, epochs=1, lambda_cpr=lambda_cpr)
        
        # Indicate the start of CPR + Refresh Learning (Unlearn steps)
        print(f'--- Starting CPR + Refresh Learning on {dataset_name}, Epoch {k+1}/{epochs} ---')
        model.train()
        for j in range(3):  # Unlearn steps
            total_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                input_ids, attention_mask, labels = batch
                output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = output.loss + cpr_loss(model, projection_matrix, lambda_cpr)
                total_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
            avg_train_loss = total_loss / len(train_loader)
            print(f'Unlearn Step {j + 1}/3, Loss: {avg_train_loss}')
        
        val_loss, val_accuracy, val_auc, val_report = evaluate(model, val_loader, return_loss=True)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print("Early stopping")
                break

        print(f'\nValidation Loss: {val_loss}, Accuracy: {val_accuracy}, AUC: {val_auc}')
        print(f'Classification Report for Validation:\n{val_report}')

        # Evaluate and print test results
        for test_loader_name, test_loader in test_loaders.items():
            test_loss, test_accuracy, test_auc, test_report = evaluate(model, test_loader, return_loss=True)
            print(f'\nTest Results for {test_loader_name}:')
            print(f'Test Loss: {test_loss}, Accuracy: {test_accuracy}, AUC: {test_auc}')
            print(f'Classification Report for Test:\n{test_report}')

    save_model(model, dataset_name + '_CPR+RL')

# Evaluation Function
def evaluate(model, data_loader, return_loss=False):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in data_loader:
            input_ids, attention_mask, labels = batch
            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = output.loss
            if return_loss:
                total_loss += loss.item()
            preds = torch.argmax(output.logits, dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())
    accuracy = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds)
    if return_loss:
        return total_loss / len(data_loader), accuracy, auc, report
    else:
        return accuracy, auc, report

# Save model function
def save_model(model, model_name):
    model.save_pretrained(f'/scratch/sakshi.rs.cse21.itbhu/dheeraj/New_RL_Paper_2/TIL Saved Model/{model_name}')

# Train and test using CPR
def train_and_test_cpr(dataset_name, train_loader, val_loader, test_loaders, epochs=3, lambda_cpr=0.01):
    print(f"\nSection 1: CPR\n{'-'*50}")
    print(f"Starting CPR training for dataset: {dataset_name}")
    model = RobertaForSequenceClassification.from_pretrained(model_path, num_labels=2)
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    projection_matrix = init_projection_matrix(model)
    
    for k in range(epochs):
        print(f'--- Starting CPR Training on {dataset_name}, Epoch {k+1}/{epochs} ---')
        model = train_cpr(model, train_loader, val_loader, projection_matrix, optimizer, scheduler, epochs=1, lambda_cpr=lambda_cpr)
        
        # Evaluate and print test results
        for test_loader_name, test_loader in test_loaders.items():
            test_loss, test_accuracy, test_auc, test_report = evaluate(model, test_loader, return_loss=True)
            print(f'\nTest Results for {test_loader_name}:')
            print(f'Test Loss: {test_loss}, Accuracy: {test_accuracy}, AUC: {test_auc}')
            print(f'Classification Report for Test:\n{test_report}')

    save_model(model, dataset_name + '_CPR')

# Run for CPR and CPR + Refresh Learning
def run_experiments():
    print("Running experiments...")
    datasets = {
        "BBBP": (bbbp_train_loader, bbbp_val_loader, {"BBBP": bbbp_test_loader}),
        "Bitter": (bitter_train_loader, bitter_val_loader, {"Bitter": bitter_test_loader}),
        "Sweet": (sweet_train_loader, sweet_val_loader, {"Sweet": sweet_test_loader}),
    }

    for dataset_name, (train_loader, val_loader, test_loaders) in datasets.items():
        # Section 1: CPR
        train_and_test_cpr(dataset_name, train_loader, val_loader, test_loaders, epochs=3, lambda_cpr=0.01)
        
        # Section 2: CPR + Refresh Learning
        refresh_learning_with_cpr(dataset_name, train_loader, val_loader, test_loaders, epochs=3, lambda_cpr=0.01)

if __name__ == "__main__":
    run_experiments()
