# ===============================
# Full House Price Prediction Script with All Visualizations
# ===============================

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

# -------------------------------
# 1️⃣ Load Data
# -------------------------------
train = pd.read_csv("/Users/marietafl/Documents/housing/train.csv")
train_id = train['Id']
train = train.drop('Id', axis=1)

# -------------------------------
# 2️⃣ Preprocess Data
# -------------------------------
numeric_cols = train.select_dtypes(include=["int64","float64"]).columns
numeric_cols = numeric_cols.drop("SalePrice")
categorical_cols = train.select_dtypes(include=["object"]).columns

train[numeric_cols] = train[numeric_cols].fillna(train[numeric_cols].mean())
train[categorical_cols] = train[categorical_cols].fillna("None")
train = pd.get_dummies(train)

# Extract target and features
y = train["SalePrice"].values
X = train.drop("SalePrice", axis=1).values

# Scale features and target
scaler_X = StandardScaler()
X = scaler_X.fit_transform(X)
scaler_y = StandardScaler()
y = scaler_y.fit_transform(y.reshape(-1,1))

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1,1)

# -------------------------------
# 3️⃣ Train/Validation Split
# -------------------------------
val_ratio = 0.2
val_size = int(len(X_tensor) * val_ratio)
train_size = len(X_tensor) - val_size

train_dataset, val_dataset = random_split(TensorDataset(X_tensor, y_tensor), [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# -------------------------------
# 4️⃣ Define Model
# -------------------------------
class HousePriceModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

model = HousePriceModel(X_tensor.shape[1])

# -------------------------------
# 5️⃣ Training Setup
# -------------------------------
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# -------------------------------
# 6️⃣ Train Model
# -------------------------------
epochs = 200
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}")

# -------------------------------
# 7️⃣ Predict on Validation Data
# -------------------------------
model.eval()
val_preds_list = []
val_actual_list = []

with torch.no_grad():
    for X_val, y_val in val_loader:
        preds = model(X_val)
        val_preds_list.append(preds)
        val_actual_list.append(y_val)

val_preds = torch.cat(val_preds_list).numpy()
val_actual = torch.cat(val_actual_list).numpy()

# Reverse scaling
val_preds_prices = scaler_y.inverse_transform(val_preds)
val_actual_prices = scaler_y.inverse_transform(val_actual)

# -------------------------------
# 8️⃣ Low/Medium/High Classes for Confusion Matrix
# -------------------------------
bins = [0, 150000, 300000, np.inf]
labels = [0,1,2]
val_actual_class = np.digitize(val_actual_prices, bins) - 1
val_pred_class   = np.digitize(val_preds_prices, bins) - 1
class_names = ["Low","Medium","High"]

# Standard confusion matrix
cm = confusion_matrix(val_actual_class, val_pred_class, labels=[0,1,2])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix (Validation Set)")
plt.show()

# Normalized confusion matrix
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(6,5))
plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Oranges)
plt.title("Normalized Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
plt.ylabel('Actual')
plt.xlabel('Predicted')
for i in range(len(class_names)):
    for j in range(len(class_names)):
        plt.text(j, i, f"{cm_normalized[i,j]:.2f}", ha="center", va="center", color="black")
plt.show()

# Misclassification bar plot
misclassified = cm.sum(axis=1) - np.diag(cm)
plt.figure(figsize=(6,4))
plt.bar(class_names, misclassified)
plt.ylabel("Number of Misclassifications")
plt.title("Misclassified Samples per Class")
plt.show()

# Classification report
report = classification_report(val_actual_class, val_pred_class, target_names=class_names)
print("Classification Report:\n", report)

# -------------------------------
# 9️⃣ Predicted vs Actual Price Matrix (Binned)
# -------------------------------
price_bins = [0, 100000, 150000, 200000, 250000, 300000, 400000, 600000, np.inf]
bin_labels = ["0-100k","100-150k","150-200k","200-250k","250-300k","300-400k","400-600k","600k+"]

actual_bins = np.digitize(val_actual_prices.flatten(), price_bins) - 1
pred_bins   = np.digitize(val_preds_prices.flatten(), price_bins) - 1

matrix_size = len(price_bins) - 1
price_matrix = np.zeros((matrix_size, matrix_size), dtype=int)

for a,p in zip(actual_bins, pred_bins):
    price_matrix[a,p] += 1

plt.figure(figsize=(10,6))
plt.imshow(price_matrix, cmap="Blues")
plt.colorbar(label="Number of houses")
plt.xticks(ticks=np.arange(matrix_size), labels=bin_labels, rotation=45)
plt.yticks(ticks=np.arange(matrix_size), labels=bin_labels)
plt.xlabel("Predicted Price Range")
plt.ylabel("Actual Price Range")
plt.title("Predicted vs Actual House Price Matrix")
for i in range(matrix_size):
    for j in range(matrix_size):
        plt.text(j,i, price_matrix[i,j], ha="center", va="center", color="black")
plt.tight_layout()
plt.show()

# -------------------------------
# 🔹 Scatter Plot: Actual vs Predicted Prices
# -------------------------------
plt.figure(figsize=(8,6))
plt.scatter(val_actual_prices, val_preds_prices, alpha=0.5, color='green')
plt.plot([val_actual_prices.min(), val_actual_prices.max()],
         [val_actual_prices.min(), val_actual_prices.max()],
         color='red', linestyle='--', linewidth=2)  # y = x line
plt.xlabel("Actual SalePrice")
plt.ylabel("Predicted SalePrice")
plt.title("Predicted vs Actual Sale Prices (Validation Set)")
plt.tight_layout()
plt.show()
