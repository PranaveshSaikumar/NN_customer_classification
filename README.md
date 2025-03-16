# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

![image](https://github.com/user-attachments/assets/32fb636a-6880-4740-832f-5342f70058f1)


## DESIGN STEPS
## STEP 1:
Import the required libraries and load the dataset.
## STEP 2:
Encode categorical values and normalize numerical data.
## STEP 3:
Divide the dataset into training and testing sets.
## STEP 4:
Create a multi-layer neural network with activation functions.
## STEP 5:
Use an optimizer and loss function to train the model on the dataset.
## STEP 6:
Test the model and generate a confusion matrix.
## STEP 7:
Use the trained model to classify a new sample.
## STEP 8:
Show the confusion matrix, classification report, and predictions.


## PROGRAM

### Name: Pranavesh Saikumar
### Register Number: 212223040149

```
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size,32)
        self.fc2 = nn.Linear(32,16)
        self.fc3 = nn.Linear(16,8)
        self.fc4 = nn.Linear(8,4)

    def forward(self,x):
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      x = F.relu(self.fc3(x))
      x = self.fc4(x)
      return x
```
```
# Initialize the Model, Loss Function, and Optimizer
model = PeopleClassifier(input_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

```
```
def train_model(model, train_loader, criterion, optimizer, epochs):
  for epochs in range(epochs):
    model.train()
    for X_batch, y_batch in train_loader:
      optimizer.zero_grad()
      output = model(X_train)
      loss = criterion(output, y_train)
      loss.backward()
      optimizer.step()

  if (epoch+1) % 10 == 0:
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
```

## Dataset Information

![image](https://github.com/user-attachments/assets/86efd67f-fe61-4a9a-b35d-c9cf113408d5)


## OUTPUT
### Confusion Matrix

![image](https://github.com/user-attachments/assets/be0cfe88-26a4-4574-9896-5f7ce3bc0ffd)

### Classification Report

![image](https://github.com/user-attachments/assets/9fe188af-6efb-4932-a748-c55423f2f702)

### New Sample Data Prediction

![image](https://github.com/user-attachments/assets/90bb23f7-3fa0-4b0d-9bbe-a5fd17ff6668)

## RESULT
Thus, a neural network classification model for the given dataset as been created successfully.
