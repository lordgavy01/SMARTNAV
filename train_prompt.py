import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import tensorflow_hub as hub
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import torch.nn.functional as F

# Load the USE
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
module_url='Models/USE'
model = hub.load(module_url)

# Generate your data
dataframe = pd.read_csv('./Datasets/train_prompt.csv') ## load data
sentences = dataframe['sentence'].tolist()
labels = dataframe['label'].tolist()
class_embeddings=model(list(set(labels))).numpy()

le = LabelEncoder()
labels = le.fit_transform(labels)
y = to_categorical(labels)

# Generate embeddings for all sentences
embeddings = model(sentences).numpy()


X_train, X_val, y_train, y_val = train_test_split(embeddings, y, test_size=0.2, random_state=42)

# Creating a PyTorch Dataset
class ClassifierDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)

train_dataset = ClassifierDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
val_dataset = ClassifierDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())

# Create a DataLoader
batch_size = 16
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size)

def get_accuracy(logit, target, batch_size):
    ''' Obtain accuracy for training round '''
    corrects = (torch.max(logit, 1)[1].view(target.size()).data == torch.max(target, 1)[1].data).sum()
    accuracy = 100.0 * corrects/batch_size
    return accuracy.item()

# Define a simple linear classifier
class SimpleClassifier(nn.Module):
    def __init__(self, num_classes):
        super(SimpleClassifier, self).__init__()
        self.layer = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, inputs):
        return self.layer(inputs)

class SimpleClassifierWithSimilarity(nn.Module):
    def __init__(self, num_classes, embedding_dim, class_emb1, class_emb2, class_emb3):
        super(SimpleClassifierWithSimilarity, self).__init__()
        self.class_emb1 = torch.nn.Parameter(torch.from_numpy(class_emb1.reshape(1,-1))) # define class embedding1
        self.class_emb1.requires_grad = False # freeze class embedding1
        self.class_emb2 = torch.nn.Parameter(torch.from_numpy(class_emb2.reshape(1,-1))) # define class embedding2
        self.class_emb2.requires_grad = False # freeze class embedding2
        self.class_emb3 = torch.nn.Parameter(torch.from_numpy(class_emb3.reshape(1,-1))) # define class embedding3
        self.class_emb3.requires_grad = False # freeze class embedding3
        self.layer1 = nn.Linear(embedding_dim*4, 512)
        self.layer2 = nn.Linear(512, num_classes)

    def forward(self, inputs):
        similarity1 = F.cosine_similarity(inputs, self.class_emb1, dim=-1).reshape(-1,1)
        similarity2 = F.cosine_similarity(inputs, self.class_emb2, dim=-1).reshape(-1,1)
        similarity3 = F.cosine_similarity(inputs, self.class_emb3, dim=-1).reshape(-1,1)
        x = torch.cat([inputs, similarity1, similarity2, similarity3], dim=1)
        print(x.dim())
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return x



classifier = SimpleClassifier(len(le.classes_))

# Create classifier
classifier = SimpleClassifierWithSimilarity(len(le.classes_), 512, class_embeddings[0], class_embeddings[1], class_embeddings[2])

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(classifier.parameters())

# Training loop
train_acc, val_acc = [], []
num_epochs = 10
for epoch in tqdm(range(num_epochs)):
    train_acc, val_acc = 0.0, 0.0

    classifier.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = classifier(inputs)
        loss = criterion(outputs, torch.max(labels, 1)[1])
        loss.backward()
        optimizer.step()
        train_loss += loss.detach().item()
        train_acc += get_accuracy(outputs, labels, batch_size)
    print("Epoch:  %d | Loss: %.4f | Train Accuracy: %.2f%%" %(epoch, train_loss / i, train_acc/i))
    classifier.eval()
    for inputs, labels in val_loader:
        outputs = classifier(inputs)
        loss = criterion(outputs, torch.max(labels, 1)[1])
        val_loss += loss.detach().item()
        val_acc += get_accuracy(outputs, labels, batch_size)
    print("Epoch:  %d | Loss: %.4f | Validation Accuracy: %.2f%%" %(epoch, val_loss / i, val_acc/i))
   
    train_acc.append(train_loss/len(train_loader))
    val_acc.append(val_loss/len(val_loader))

# Plot learning curve
plt.figure()
plt.plot(train_losses, label='Train loss')
plt.plot(val_losses, label='Validation loss')
plt.legend()
plt.title('Learning curve')
plt.show()