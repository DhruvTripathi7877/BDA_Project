from pyspark.sql import SparkSession
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.linalg import Vectors
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Initialize Spark session
spark = SparkSession.builder.appName("SparkCNNIntegration").getOrCreate()

# Load the dataset
df = spark.read.csv('./data/final_dataset_train_all.csv', header=True, inferSchema=True)
df = df.na.drop()

# Convert target column to one-hot encoding and split features and labels
from pyspark.sql.functions import udf, col
from pyspark.sql.types import ArrayType, IntegerType

def one_hot_encode(label, num_classes=7):
    encoding = [0] * num_classes
    encoding[int(label)] = 1
    return encoding

one_hot_udf = udf(lambda x: one_hot_encode(x), ArrayType(IntegerType()))

data = df.withColumn("label", one_hot_udf(col("target"))).drop("target")

# Convert features to a Spark Vector
feature_cols = [c for c in data.columns if c != "label"]
data = data.rdd.map(lambda row: (Vectors.dense(row[:-1]), row[-1])).toDF(["features", "label"])

# Scale features
scaler = MinMaxScaler(inputCol="features", outputCol="scaledFeatures")
scaler_model = scaler.fit(data)
data = scaler_model.transform(data).select("scaledFeatures", "label")

# Split into training and testing sets
train_data, test_data = data.randomSplit([0.9, 0.1], seed=42)

# Convert Spark DataFrame to Pandas for PyTorch
train_pd = train_data.toPandas()
test_pd = test_data.toPandas()

X_train = np.array([x.toArray() for x in train_pd["scaledFeatures"]])
y_train = np.array(train_pd["label"].tolist())
X_test = np.array([x.toArray() for x in test_pd["scaledFeatures"]])
y_test = np.array(test_pd["label"].tolist())

# PyTorch Dataset
class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return len(self.X)

# Create datasets and loaders
datasets = {
    "train": EEGDataset(X_train, y_train),
    "test": EEGDataset(X_test, y_test)
}

train_loader = DataLoader(datasets['train'], shuffle=True, batch_size=64)
test_loader = DataLoader(datasets['test'], shuffle=True, batch_size=64)

# Define CNN model
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=6, stride=1)
        self.conv2 = nn.Conv1d(64, 8, kernel_size=6, stride=1)
        self.linear1 = nn.Linear(9040, 1024)
        self.linear2 = nn.Linear(1024, 7)
        self.leakyrelu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=0.25)

    def forward(self, X):
        X = X.unsqueeze(1)  # Add channel dimension
        X = self.leakyrelu(self.conv1(X))
        X = self.leakyrelu(self.conv2(X))
        X = X.view(-1, 9040)  # Flatten
        X = self.leakyrelu(self.linear1(X))
        X = self.dropout(X)
        X = self.linear2(X)
        return X

# Training and evaluation setup
def train_model(model, criterion, optimizer, train_loader, test_loader, epochs=100):
    for epoch in range(epochs):
        model.train()
        train_loss, correct = 0, 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, torch.argmax(y_batch, dim=1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            correct += (outputs.argmax(1) == y_batch.argmax(1)).sum().item()

        train_accuracy = correct / len(train_loader.dataset) * 100

        model.eval()
        with torch.no_grad():
            correct = 0
            for X_batch, y_batch in test_loader:
                outputs = model(X_batch)
                correct += (outputs.argmax(1) == y_batch.argmax(1)).sum().item()

            test_accuracy = correct / len(test_loader.dataset) * 100

        print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%")

# Initialize model, loss, optimizer
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
train_model(model, criterion, optimizer, train_loader, test_loader)

# Save model
torch.save(model.state_dict(), "cnn_model.pth")
