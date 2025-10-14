import pandas as pd
import time
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support, confusion_matrix, roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize

# Step 1: Load and Prepare the Dataset
df = pd.read_csv('final_dataset.csv')

# Drop specific columns you don't want to use
columns_to_remove = ['ip.src_host', 'ip.dst_host', 'arp.src.proto_ipv4', 'tcp.payload', 'http.file_data']
df.drop(columns=columns_to_remove, inplace=True)

# Rename 'Attack_label' to 'Label'
df.rename(columns={'Attack_label': 'Label'}, inplace=True)

# Combine relevant features (excluding Label and Attack_type) into a single string
df['text'] = df.apply(
    lambda row: ' '.join([f"{col}: {row[col]}" for col in df.columns if col not in ['Label', 'Attack_type']]),
    axis=1
)

# Convert 'Label' to integer
df['label'] = df['Label'].astype(int)

# Step 2: Split the Dataset into Training and Testing Sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

# Step 3: Tokenization using BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the data for both training and testing sets
train_encodings = tokenizer(train_df['text'].tolist(), padding=True, truncation=True, max_length=512, return_tensors='pt')
test_encodings = tokenizer(test_df['text'].tolist(), padding=True, truncation=True, max_length=512, return_tensors='pt')

train_labels = train_df['label'].tolist()
test_labels = test_df['label'].tolist()

# Custom Dataset Class for PyTorch
class IoTDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Create the training and testing datasets
train_dataset = IoTDataset(train_encodings, train_labels)
test_dataset = IoTDataset(test_encodings, test_labels)

# Step 4: Device Identification (using GPU if available)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Step 5: Model Setup (BERT for sequence classification)
num_classes = len(df['label'].unique())  # Number of unique classes
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_classes)
model.to(device)

# Step 6: Define Custom Metrics for Multiclass
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    # Overall accuracy
    accuracy = accuracy_score(labels, preds)
    
    # Weighted precision, recall, and F1-score
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    
    # Class-specific metrics
    report = classification_report(labels, preds, output_dict=True)
    
    print("\nClassification Report:")
    print(classification_report(labels, preds))
    
    # Save class-specific metrics to CSV
    class_report_df = pd.DataFrame(report).transpose()
    class_report_df.to_csv("anomaly_performance/classification_report.csv", index=True)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'class_report': report
    }

# Step 7: Training Arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    save_total_limit=2,
    fp16=torch.cuda.is_available()
)

# Step 8: Trainer Setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# Step 9: Training with Time Measurement
start_time = time.time()
trainer.train()
training_time = time.time() - start_time
print(f"Training Time: {training_time:.2f} seconds")

# Step 10: Evaluation with Inference Time Measurement
start_time = time.time()
eval_result = trainer.evaluate()
inference_time = time.time() - start_time
print(f"Inference Time: {inference_time:.2f} seconds")

# Save overall evaluation metrics and timing information to CSV
eval_result['training_time'] = training_time
eval_result['inference_time'] = inference_time
eval_metrics_df = pd.DataFrame([eval_result])
eval_metrics_df.to_csv("anomaly_performance/evaluation_metrics_with_time.csv", index=False)

print(f"Evaluation results: {eval_result}")

# Step 11: Visualization

# 1. Performance Metrics Over Epochs
train_logs = [log for log in trainer.state.log_history if 'eval_accuracy' in log]
epochs = [log['epoch'] for log in train_logs]
accuracies = [log['eval_accuracy'] for log in train_logs]
f1_scores = [log['eval_f1'] for log in train_logs]
precision_scores = [log['eval_precision'] for log in train_logs]
recall_scores = [log['eval_recall'] for log in train_logs]

plt.figure(figsize=(10, 6))
plt.plot(epochs, accuracies, label="Accuracy", marker='o', linestyle='-', linewidth=2, color='blue')
plt.plot(epochs, f1_scores, label="F1 Score", marker='s', linestyle='--', linewidth=2, color='orange')
plt.plot(epochs, precision_scores, label="Precision", marker='^', linestyle='-.', linewidth=2, color='green')
plt.plot(epochs, recall_scores, label="Recall", marker='d', linestyle=':', linewidth=2, color='red')
plt.ylim(0.995, 1.001)
plt.xlabel("Epochs")
plt.ylabel("Performance")
plt.title("Model Performance Over Epochs")
plt.legend()
plt.grid()
plt.savefig("anomaly_performance/Performance_Metrics_Improved.pdf", format="pdf", dpi=300)
plt.close()

# 2. Confusion Matrix
predictions = trainer.predict(test_dataset).predictions.argmax(-1)
cm = confusion_matrix(test_labels, predictions)

# Map numeric labels to Attack_type
label_to_attack_type = df[['label', 'Attack_type']].drop_duplicates().sort_values('label')
label_to_attack_type = label_to_attack_type.set_index('label')['Attack_type'].to_dict()

# Abbreviations for long labels
abbreviations = {
    "Vulnerability_scanner": "Vul_Scan",
    "SQL_injection": "SQL_Inj",
    "Port_Scanning": "Port_Scan",
    "DDoS_HTTP": "DDoS_HTTP",
    "DDoS_TCP": "DDoS_TCP",
    "DDoS_UDP": "DDoS_UDP",
    "DDoS_ICMP": "DDoS_ICMP",
}
attack_type_labels = [
    abbreviations[label_to_attack_type[label]] if label_to_attack_type[label] in abbreviations else label_to_attack_type[label]
    for label in sorted(label_to_attack_type.keys())
]

# Normalize confusion matrix to percentages with two decimal places
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
cm_normalized = np.nan_to_num(cm_normalized)

plt.figure(figsize=(14, 12))
sns.heatmap(
    cm_normalized,
    annot=True,
    fmt=".2f",
    cmap="viridis",
    xticklabels=attack_type_labels,
    yticklabels=attack_type_labels
)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix with Attack Types (Values in Percentages)")
plt.tight_layout() 
plt.savefig("anomaly_performance/Confusion_Matrix_With_Percentages.pdf", format="pdf", dpi=300)
plt.close()

# 3. ROC Curve
labels_binarized = label_binarize(test_labels, classes=list(range(num_classes)))
predictions_probs = trainer.predict(test_dataset).predictions
colors = sns.color_palette("tab10", num_classes)  # Highly distinguishable colors

plt.figure(figsize=(12, 8))
auc_data = []  # To store original AUC values for CSV
for i in range(num_classes):
    fpr, tpr, _ = roc_curve(labels_binarized[:, i], predictions_probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{label_to_attack_type[i]} (AUC = {roc_auc:.2f})", color=colors[i])
    # Save original AUC values for the CSV file
    auc_data.append({
        'Class': label_to_attack_type[i],
        'AUC': roc_auc
    })

# Save Original AUC Values to CSV
auc_df = pd.DataFrame(auc_data)
auc_df.to_csv("anomaly_performance/AUC_Original_Values_Per_Class.csv", index=False)
print("Original AUC values saved to 'anomaly_performance/AUC_Original_Values_Per_Class.csv'")

# Finalize and Save ROC Plot
plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve ")
plt.legend(loc="lower right", fontsize='small', ncol=2)
plt.grid()
plt.savefig("anomaly_performance/ROC_Curve_With_Attack_Types.pdf", format="pdf", dpi=300)
plt.close()

# Step 12: Save the Trained Model and Tokenizer
model.save_pretrained('./saved_model')
tokenizer.save_pretrained('./saved_model')
