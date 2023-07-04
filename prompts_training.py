import torch
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import tensorflow_hub as hub
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import pandas as pd
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

# Load the USE
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"

model = hub.load(module_url)

# Generate your data
dataframe = pd.read_csv('train_omega_prompt.csv') ## load data
sentences = dataframe['sentence'].tolist()
labels = dataframe['label'].tolist()
class_embeddings=model(list(set(labels))).numpy()
le = LabelEncoder()
labels = le.fit_transform(labels)
y = to_categorical(labels)

# Generate embeddings for all sentences
embeddings = model(sentences).numpy()

X_train, X_val, y_train, y_val = train_test_split(embeddings, y, test_size=0.2, random_state=15)

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
batch_size = 64
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size)

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
        self.layer1 = nn.Linear(embedding_dim+3, 512)
        self.layer2 = nn.Linear(512, num_classes)

    def forward(self, inputs):

        similarity1 = F.cosine_similarity(inputs, self.class_emb1, dim=-1).reshape(-1,1)

        similarity2 = F.cosine_similarity(inputs, self.class_emb2, dim=-1).reshape(-1,1)

        similarity3 = F.cosine_similarity(inputs, self.class_emb3, dim=-1).reshape(-1,1)
        x = torch.cat([inputs, similarity1, similarity2, similarity3], dim=1)

        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return x

# classifier = SimpleClassifier(len(le.classes_))

# Create classifier
classifier = SimpleClassifierWithSimilarity(len(le.classes_), 512, class_embeddings[0], class_embeddings[1], class_embeddings[2])

LearningRate=0.009
num_epochs = 15
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(classifier.parameters(), lr=LearningRate ,betas=(0.9, 0.97), eps=1e-07)
total_steps = len(train_loader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=1, num_training_steps=total_steps)


def get_accuracy(logit, target):
    ''' Obtain accuracy for training round '''
    corrects = (logit.argmax(dim=1) == target.argmax(dim=1)).sum().item()
    total = target.size(0)
    return 100.0 * corrects/total
def class_wise_accuracy(model, dataloader):
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            _, labels = torch.max(labels, 1)
            all_preds.extend(predicted.numpy())
            all_labels.extend(labels.numpy())

    conf_matrix = confusion_matrix(all_labels, all_preds)
    class_accuracy = conf_matrix.diagonal()/conf_matrix.sum(1)
    return class_accuracy

train_accs, val_accs = [], []

for epoch in tqdm(range(num_epochs)):
    train_acc, val_acc = 0.0, 0.0
    train_loss, val_loss = 0.0, 0.0
    classifier.train()

    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = classifier(inputs)

        loss = criterion(outputs, labels.argmax(dim=1))
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_loss += loss.detach().item()
        train_acc += get_accuracy(outputs, labels)

    print("Epoch:  %d | Loss: %.4f | Train Accuracy: %.2f%%" %(epoch, train_loss / (i+1), train_acc/(i+1)))

    classifier.eval()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(val_loader):
            outputs = classifier(inputs)
            loss = criterion(outputs, labels.argmax(dim=1))
            val_loss += loss.item()
            val_acc += get_accuracy(outputs, labels)

    print("Epoch:  %d | Loss: %.4f | Validation Accuracy: %.2f%%" %(epoch, val_loss / (i+1), val_acc/(i+1)))

    train_accs.append(train_acc/len(train_loader))
    val_accs.append(val_acc/len(val_loader))

val_class_accuracy = class_wise_accuracy(classifier, val_loader)
class_names = le.inverse_transform([0,1,2])
print('Class Name',class_names)
print('Validation class wise accuracy: ', val_class_accuracy)

# Plot learning curve
plt.figure()
plt.plot(train_accs, label='Train Accuracy')
plt.plot(val_accs, label='Validation Accuracy')
plt.legend()
plt.title('Learning curve Angular Velocity Commands')
plt.show()

def predict_class(encoder,model, sentence):
    # Transform sentence to embedding
    embedding = encoder([sentence]).numpy()

    # Pass through model
    output = model(torch.Tensor(embedding))

    # Convert output probabilities to predicted class
    pred = torch.max(output, dim=1)[1]
    pred = pred.numpy()[0]

    # Transform class label back to original encoding
    pred_label = le.inverse_transform([pred])[0]

    return pred_label

# Example usage:
input_sentence = "try not rotating much"
print(f"The predicted class is '{predict_class(model,classifier, input_sentence)}'")

checkpoint=classifier.state_dict()
torch.save(checkpoint,'omega_layer.pth')

