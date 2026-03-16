# =====================================
# House Price Prediction with PyTorch
# Train: train.csv
# Validation: test.csv
# =====================================

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load datasets
# -----------------------------
train = pd.read_csv("/Users/marietafl/Documents/housing/train.csv")
test = pd.read_csv("/Users/marietafl/Documents/housing/test.csv")

train_id = train["Id"]
test_id = test["Id"]

train = train.drop("Id", axis=1)
test = test.drop("Id", axis=1)

# -----------------------------
# 2. Separate target
# -----------------------------
y_train = train["SalePrice"]
train = train.drop("SalePrice", axis=1)

# -----------------------------
# 3. Combine datasets for preprocessing
# -----------------------------
full_data = pd.concat([train, test], axis=0)

# -----------------------------
# 4. Handle missing values
# -----------------------------
numeric_cols = full_data.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = full_data.select_dtypes(include=["object"]).columns

full_data[numeric_cols] = full_data[numeric_cols].fillna(full_data[numeric_cols].mean())
full_data[categorical_cols] = full_data[categorical_cols].fillna("None")

# -----------------------------
# 5. Convert categorical variables
# -----------------------------
full_data = pd.get_dummies(full_data)

# -----------------------------
# 6. Split data back
# -----------------------------
X_train = full_data.iloc[:len(train), :]
X_test = full_data.iloc[len(train):, :]

# -----------------------------
# 7. Feature scaling
# -----------------------------
scaler_X = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train = scaler_y.fit_transform(y_train.values.reshape(-1,1))

# -----------------------------
# 8. Convert to tensors
# -----------------------------
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# -----------------------------
# 9. Define neural network
# -----------------------------
class HousePriceModel(nn.Module):

    def __init__(self, input_size):
        super().__init__()

        self.fc1 = nn.Linear(input_size,128)
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,1)

    def forward(self,x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


model = HousePriceModel(X_train.shape[1])

# -----------------------------
# 10. Training setup
# -----------------------------
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 200

# -----------------------------
# 11. Train model
# -----------------------------
for epoch in range(epochs):

    total_loss = 0

    for X_batch, y_batch in train_loader:

        optimizer.zero_grad()

        predictions = model(X_batch)

        loss = criterion(predictions, y_batch)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

# -----------------------------
# 12. Predict test data
# -----------------------------
model.eval()

with torch.no_grad():

    test_predictions = model(X_test_tensor)

test_predictions = test_predictions.numpy()

# convert back to real prices
test_predictions = scaler_y.inverse_transform(test_predictions)

# -----------------------------
# 13. Create submission file
# -----------------------------
submission = pd.DataFrame({
    "Id": test_id,
    "SalePrice": test_predictions.flatten()
})

submission.to_csv("submission.csv", index=False)

print("Predictions saved to submission.csv")

# -----------------------------
# 14. Optional visualization
# -----------------------------
# if SalePrice exists in test set

if "SalePrice" in test.columns:

    actual_prices = test["SalePrice"].values
    predicted_prices = test_predictions.flatten()

    plt.figure(figsize=(8,6))

    plt.scatter(actual_prices, predicted_prices, alpha=0.5)

    plt.plot([actual_prices.min(), actual_prices.max()],
             [actual_prices.min(), actual_prices.max()],
             color="red", linestyle="--")

    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual vs Predicted House Prices")

    plt.show()


compare the prices from the submission.csv file with those from another sample_submission.csv file for the same ids
