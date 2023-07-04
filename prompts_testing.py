import torch
import torch.nn as nn
import tensorflow_hub as hub
import pandas as pd
import torch.nn.functional as F
import re
import string

# Load the USE
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"

model = hub.load(module_url)

v_dict={0:'Decrease Velocity',1: 'Increase Velocity',2: 'Remain Same Velocity'}
w_dict={0:'Decrease Omega',1: 'Increase Omega',2: 'Remain Same Omega'}

def pre_process(sentences):
    for i in range(len(sentences)):
        sentence=sentences[i]
        # Remove numbering
        sentence = re.sub(r'^\d+\.+\s', '', sentence)
        # Remove quotes
        sentence = sentence.replace('"', '').replace("'", '')
        # Remove other punctuation
        sentence = sentence.translate(str.maketrans('', '', string.punctuation))
        # Preprocess for USE - make lowercase (optional)
        sentence = sentence.lower()
        sentences[i]=sentence

def predict_classes(encoder, model, class_dict, sentences, threshold=0.5):
    # Transform sentences to embeddings
    embeddings = encoder(sentences).numpy()

    predictions = []
    for embedding in embeddings:
        # Pass through model
        output = model(torch.Tensor(embedding.reshape(1, -1)))
        
        # Convert output probabilities to predicted class
        _, pred = torch.max(output, dim=1)

        # Calculate confidence
        confidence = torch.nn.functional.softmax(output, dim=1)[0][pred].item()

        pred = pred.numpy()[0]
        # Add prediction only if confidence is higher than the threshold
       
        if confidence >= threshold:
            # Transform class label back to original encoding
            pred_label = class_dict[pred]
            predictions.append(pred_label)
        else:
            # Add None or any other indicator for low confidence predictions
            predictions.append(class_dict[2])

    return predictions

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

# classifier = SimpleClassifierWithSimilarity(len(le.classes_), 512, class_embeddings[0], class_embeddings[1], class_embeddings[2])

v_checkpoint=torch.load('velocity_layer.pth')
v_classifier=SimpleClassifierWithSimilarity(3,512,v_checkpoint['class_emb1'].numpy(),v_checkpoint['class_emb2'].numpy(),v_checkpoint['class_emb3'].numpy())
v_classifier.load_state_dict(v_checkpoint)

w_checkpoint=torch.load('omega_layer.pth')
w_classifier=SimpleClassifierWithSimilarity(3,512,w_checkpoint['class_emb1'].numpy(),w_checkpoint['class_emb2'].numpy(),w_checkpoint['class_emb3'].numpy())
w_classifier.load_state_dict(w_checkpoint)

input_sentences = [
"Maintain a safe distance from nearby obstacles to avoid collision.",
"Caution: The area ahead is highly congested. Consider slowing down to navigate safely. ",
"Watch out! There's a large obstacle approaching. Make sure to maintain safe distance. ",
"Good news! The destination is within your line of sight. Proceed towards it, ensuring to avoid any collisions. ",
"Alert: You are in a potential deadlock situation. It's recommended to wait and monitor the other agent's actions to avoid collision.",
"All clear. Proceed towards the destination."]

# pre_process(input_sentences)

v_predictions = predict_classes(model, v_classifier, v_dict, input_sentences, threshold=0.9)
w_predictions = predict_classes(model, w_classifier, w_dict, input_sentences, threshold=0.9)

# print(f"The predicted classes are '{predict_classes(model, v_classifier, v_dict, input_sentences, threshold=0.2)}'")
# print(f"The predicted classes are '{predict_classes(model, w_classifier, w_dict, input_sentences, threshold=0.2)}'")

# Combine the results
results = pd.DataFrame({'sentence': input_sentences, 
                        'v_prediction': v_predictions, 
                        'w_prediction': w_predictions})

# Print the results
results.to_csv('prompt_testing_results.csv')