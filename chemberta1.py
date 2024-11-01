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

# Encode each dataset
bbbp_encoded = tokenize(bbbp_inputs)
bitter_encoded = tokenize(bitter_inputs)
sweet_encoded = tokenize(sweet_inputs)

# Split data
def split_data(encoded_inputs, labels):
    train_indices, test_indices, train_labels, test_labels = train_test_split(range(len(labels)), labels, test_size=0.2, random_state=42)
    val_indices, test_indices, val_labels, test_labels = train_test_split(test_indices, test_labels, test_size=0.5, random_state=42)
    return train_indices, val_indices, test_indices, train_labels, val_labels, test_labels

# Split datasets
bbbp_train_idx, bbbp_val_idx, bbbp_test_idx, bbbp_train_lbl, bbbp_val_lbl, bbbp_test_lbl = split_data(bbbp_encoded, bbbp_labels)
bitter_train_idx, bitter_val_idx, bitter_test_idx, bitter_train_lbl, bitter_val_lbl, bitter_test_lbl = split_data(bitter_encoded, bitter_labels)
sweet_train_idx, sweet_val_idx, sweet_test_idx, sweet_train_lbl, sweet_val_lbl, sweet_test_lbl = split_data(sweet_encoded, sweet_labels)

# Create DataLoaders
def create_tensor_dataset(encoded_inputs, indices, labels):
    return TensorDataset(encoded_inputs['input_ids'][indices], encoded_inputs['attention_mask'][indices], torch.tensor(labels))

def create_data_loader(tensor_dataset, batch_size=16):
    return DataLoader(tensor_dataset, batch_size=batch_size, shuffle=True)

# Create DataLoaders for each dataset
bbbp_train_loader = create_data_loader(create_tensor_dataset(bbbp_encoded, bbbp_train_idx, bbbp_train_lbl))
bbbp_val_loader = create_data_loader(create_tensor_dataset(bbbp_encoded, bbbp_val_idx, bbbp_val_lbl), batch_size=32)
bbbp_test_loader = create_data_loader(create_tensor_dataset(bbbp_encoded, bbbp_test_idx, bbbp_test_lbl), batch_size=32)

bitter_train_loader = create_data_loader(create_tensor_dataset(bitter_encoded, bitter_train_idx, bitter_train_lbl))
bitter_val_loader = create_data_loader(create_tensor_dataset(bitter_encoded, bitter_val_idx, bitter_val_lbl), batch_size=32)
bitter_test_loader = create_data_loader(create_tensor_dataset(bitter_encoded, bitter_test_idx, bitter_test_lbl), batch_size=32)

sweet_train_loader = create_data_loader(create_tensor_dataset(sweet_encoded, sweet_train_idx, sweet_train_lbl))
sweet_val_loader = create_data_loader(create_tensor_dataset(sweet_encoded, sweet_val_idx, sweet_val_lbl), batch_size=32)
sweet_test_loader = create_data_loader(create_tensor_dataset(sweet_encoded, sweet_test_idx, sweet_test_lbl), batch_size=32)

# EWC and Refresh Learning Functions
def init_fim(model):
    fim = {}
    for name, param in model.named_parameters():
        fim[name] = param.data.clone().zero_()
    return fim

def update_fim(model, fim, fisher_coeff):
    for name, param in model.named_parameters():
        fim[name].add_(fisher_coeff * param.grad.data ** 2)

def oewc_loss(prev_model, current_model, fim, fisher_coeff):
    loss = 0
    for name, param in current_model.named_parameters():
        if name in fim:
            loss += torch.sum(fim[name] * (param - prev_model.state_dict()[name]) ** 2)
    return (fisher_coeff / 2) * loss

# Refresh Learning functions
def refresh_learning(model, prev_model, fim, fisher_coeff):
    loss = oewc_loss(prev_model, model, fim, fisher_coeff)
    # Add more Refresh Learning logic if necessary
    return loss

# Load the ChemBERTa model
def initialize_model():
    return RobertaForSequenceClassification.from_pretrained(model_path, num_labels=2)

def create_optimizer_and_scheduler(model, total_steps):
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    return optimizer, scheduler

# Define training parameters
epochs = 3  # Change this to 25 for longer training
fisher_coeff = 1e-6
early_stopping_patience = 2  # Set the patience for early stopping

# Train the model with oEWC
def train_oewc(model, train_loader, val_loader, fim=None, fisher_coeff=fisher_coeff, epochs=3, prev_model=None):
    global best_val_loss, early_stopping_counter
    optimizer, scheduler = create_optimizer_and_scheduler(model, len(train_loader) * epochs)
    best_val_loss = float('inf')
    early_stopping_counter = 0
    
    # Initialize fim if it's None
    if fim is None:
        fim = init_fim(model)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        all_preds, all_labels = [], []
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids, attention_mask, labels = batch
            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = output.loss
            ewc_term = 0
            if prev_model is not None:
                ewc_term = oewc_loss(prev_model, model, fim, fisher_coeff)
            total_loss += (loss + ewc_term).item()
            (loss + ewc_term).backward()
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
        
        update_fim(model, fim, fisher_coeff)
        val_loss, val_accuracy, val_auc, val_report = evaluate(model, val_loader, return_loss=True)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print("Early stopping")
                break
        prev_model = model
    return prev_model

# Evaluation function
def evaluate(model, data_loader, return_loss=False):
    model.eval()
    preds, true_labels = [], []
    total_loss = 0
    for batch in data_loader:
        input_ids, attention_mask, labels = batch
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits
        loss = outputs.loss
        total_loss += loss.item()
        preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
        true_labels.extend(labels.cpu().tolist())
    accuracy = accuracy_score(true_labels, preds)
    classification_report_str = classification_report(true_labels, preds)
    auc = roc_auc_score(true_labels, preds)
    if return_loss:
        return total_loss / len(data_loader), accuracy, auc, classification_report_str
    return accuracy, auc, classification_report_str

# Training and testing for OEWC and OEWC + Refresh Learning

def run_training(model, fim, prev_model=None, method="OEWC"):
    print(f"\nSection 1: {method}")
    
    print("\n1) Train on BBBP Dataset")
    prev_model = train_oewc(model, bbbp_train_loader, bbbp_val_loader, fim=fim, prev_model=prev_model)
    print("Test on BBBP Dataset")
    test_accuracy, test_auc, test_report = evaluate(model, bbbp_test_loader)
    print(f"Test Accuracy: {test_accuracy}, AUC: {test_auc}\nClassification Report:\n{test_report}")
    
    print("\n2) Train on Bitter Dataset")
    prev_model = train_oewc(model, bitter_train_loader, bitter_val_loader, fim=fim, prev_model=prev_model)
    print("Test on Bitter Dataset")
    test_accuracy, test_auc, test_report = evaluate(model, bitter_test_loader)
    print(f"Test Accuracy: {test_accuracy}, AUC: {test_auc}\nClassification Report:\n{test_report}")
    
    print("\n3) Train on Sweet Dataset")
    prev_model = train_oewc(model, sweet_train_loader, sweet_val_loader, fim=fim, prev_model=prev_model)
    print("Test on Sweet Dataset")
    test_accuracy, test_auc, test_report = evaluate(model, sweet_test_loader)
    print(f"Test Accuracy: {test_accuracy}, AUC: {test_auc}\nClassification Report:\n{test_report}")
    
    return prev_model

def run_oewc():
    fim = None
    model = initialize_model()
    prev_model = run_training(model, fim, method="OEWC")
    fim = init_fim(model)
    
    model = initialize_model()  # Reinitialize model for OEWC + Refresh Learning
    run_training(model, fim, prev_model=prev_model, method="OEWC + Refresh Learning")

# Run the OEWC and OEWC + Refresh Learning
run_oewc()
