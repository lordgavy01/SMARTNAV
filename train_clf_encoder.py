import pandas as pd
import numpy as np
from collections import defaultdict
import torch
from torch import nn
from transformers import AdamW,get_linear_schedule_with_warmup
from tqdm import tqdm 
import time
from seqeval.metrics import f1_score, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from random import randint
from sklearn.model_selection import train_test_split
from collections import Counter

from Models.model_clf import CLFModel
from Models.model_new_encoder_clf import OGCLFModel

class LidarDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]).float() for key, val in self.encodings.items() if key != 'Y'}
        item['Y'] = torch.tensor(self.encodings['Y'][idx]).long()  # Convert 'Y' to long data type
        return item
    def __len__(self):
        return len(self.encodings['X1'])

class OGCLFTrainer:
    def __init__(self,filename,model,depths=False):
        self.DF=pd.read_csv(filename).drop_duplicates().sample(frac=1,random_state=15)
        # max_limit = 5000
        # self.DF = self.D.groupby('action_label').head(max_limit).reset_index(drop=True)
        # self.DF=self.D[(self.D['action_label']==20)|(self.D['action_label']==21)]
        # print(self.DF['action_label'].value_counts())
        self.class_mapping={c:i for i,c in enumerate(self.DF["action_label"].value_counts().keys())}
        self.trainDF,self.validDF= train_test_split(self.DF, test_size=0.1, random_state=15,stratify=self.DF["action_label"])
        self.cleanDataset(self.trainDF,'train',depths)
        self.cleanDataset(self.validDF,'val',depths)
        self.model=model
    def cleanDataset(self,dataframe,train_or_val='train',depth_flag=False):

        inputs = defaultdict(list)
        # # Create the Y list
        # class_mapping={
        #     0: 0,
        #     5: 1,
        #     6: 2,
        #     7: 3,
        #     8: 4,
        #     9: 5,
        #     10: 6,
        #     11: 7
        # }
       
        
        dataframe["action_label"]=dataframe["action_label"].map(self.class_mapping)
        dataframe=dataframe[(dataframe['action_label']!=1) & (dataframe['action_label']<=3)]
        print(train_or_val)
        print(dataframe["action_label"].value_counts())

        inputs["Y"] = dataframe["action_label"].tolist()
        # print(dataframe["action_label"].value_counts())
        # Create the X2 list
        inputs["X2"] = dataframe[["distance_from_goal", "angle_from_goal"]].to_numpy().tolist()

        # Create the X1 list
        if depth_flag:
            depth_columns = [f"lidar_depth_{i}" for i in range(50)]
            inputs["X1"] = dataframe[depth_columns].to_numpy().tolist()
        else:
            embedding_columns = [f"embedding_dim_{i}" for i in range(512)]
            inputs["X1"] = dataframe[embedding_columns].to_numpy().tolist()

        self.loadDataset(inputs,train_or_val)

    def loadDataset(self,inputs,train_or_val='train'):
        if train_or_val=='train':
            self.dataset=LidarDataset(inputs)   
            self.loader=torch.utils.data.DataLoader(self.dataset, batch_size=2048, shuffle=True,pin_memory=True)
        else:
            self.val_dataset=LidarDataset(inputs)   
            self.val_loader=torch.utils.data.DataLoader(self.val_dataset, batch_size=2048, shuffle=True,pin_memory=True)
   
    def trainDataset(self,checkpointFilename,LearningRate=1e-3):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # device = torch.device('cpu')
        self.model=self.model.to(device)
        LearningRate=1.5e-3
        Epochs=4

        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
                {
                        'params':[p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                        'weight_decay':0.1
                },
                {
                        'params':[p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                        'weight_decay':0.0
                }
        ]
        # initialize optimizer
        optimizer = AdamW(optimizer_grouped_parameters, lr=LearningRate ,betas=(0.9, 0.97), eps=1e-07,no_deprecation_warning=True)
        class_weights = torch.tensor(self.trainDF['action_label'].value_counts(normalize=True).sort_index().values).float().to(device)
        
        # criterion = nn.CrossEntropyLoss(weight=class_weights)   
        criterion = nn.CrossEntropyLoss()
        total_steps = len(self.loader) * Epochs
        # adding a scheduler to linearly reduce the learning rate throughout the epochs.
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=40, num_training_steps=int(1.5*total_steps))
        self.model.eval()
        all_prob=[]
        all_labels=[]
        # Deactivate autograd for evaluation.
        with torch.no_grad():
            for batch in self.val_loader:
                batch_X1 = batch['X1'].to(device)
                batch_X2 = batch['X2'].to(device)
                batch_Y = batch['Y'].to(device)
                logits = self.model(batch_X1,batch_X2)
                loss = criterion(logits.view(-1, self.model.num_labels), batch_Y.view(-1))
                probabilities = nn.functional.softmax(logits, dim=-1)
                _, out_classes = probabilities.max(dim=1)
                
                
                all_prob.extend(out_classes.cpu().numpy().astype(int))
                all_labels.extend(batch_Y.cpu().numpy().astype(int))
        pred_tags = all_prob
        valid_tags = all_labels     
        
        classwise_accuracy = {c:0.0 for c in self.validDF['action_label'].value_counts().keys()}
        validation_accuracy=0.0
        
        for i,pred_tag in enumerate(pred_tags):
            if pred_tag==valid_tags[i]:
                validation_accuracy+=1.0
                classwise_accuracy[pred_tag]+=1.0
        count_dict=self.validDF['action_label'].value_counts()
        
        for c,v in classwise_accuracy.items():
            classwise_accuracy[c]=100*v/count_dict[c]
        
        validation_accuracy=validation_accuracy/len(pred_tags)
        validation_accuracy=validation_accuracy*100

        print(validation_accuracy)
        print(classwise_accuracy)

        train_loss=[]
        val_accuracies=[]
        for epoch in range(Epochs):
            self.model.train() 
            print("* Training epoch {}:".format(epoch))
            # Starting time for the current epoch
            epochStart = time.time()
            # Running loss for the current epoch
            runningLoss = 0.0
            # setup loop with TQDM and dataloader
            loop = tqdm(self.loader, leave=True)
            for batch_index,batch in enumerate(loop):
                # initialize calculated gradients (from prev step)
                optimizer.zero_grad()
                # pull all tensor batches required for training
                batch_X1 = batch['X1'].to(device)
                batch_X2 = batch['X2'].to(device)
                batch_Y = batch['Y'].to(device)
                
                # process
                logits = self.model(batch_X1,batch_X2)
                loss = criterion(logits.view(-1, self.model.num_labels), batch_Y.view(-1))
                
                probabilities = nn.functional.softmax(logits, dim=-1)
                _, out_classes = probabilities.max(dim=1)                
                
                # extract loss
                runningLoss += loss.item()
                # Update Weights
                loss.backward()
                optimizer.step()
                scheduler.step()
                # print relevant info to progress bar
                loop.set_description(f'Epoch {epoch}')
                loop.set_postfix(RunningLoss=runningLoss/(batch_index+1))
            

            self.model.eval()
            all_prob=[]
            all_labels=[]
            # Deactivate autograd for evaluation.
            with torch.no_grad():
                for batch in self.val_loader:
                    batch_X1 = batch['X1'].to(device)
                    batch_X2 = batch['X2'].to(device)
                    batch_Y = batch['Y'].to(device)
                    logits = self.model(batch_X1,batch_X2)
                    loss = criterion(logits.view(-1, self.model.num_labels), batch_Y.view(-1))
                    probabilities = nn.functional.softmax(logits, dim=-1)
                    _, out_classes = probabilities.max(dim=1)
                    
                    all_prob.extend(out_classes.cpu().numpy().astype(int))
                    all_labels.extend(batch_Y.cpu().numpy().astype(int))
            df=pd.DataFrame({'predict':all_prob,'real':all_labels})
            df.to_csv('val.csv')
            pred_tags = all_prob
            valid_tags = all_labels    
            classwise_accuracy = {c:0.0 for c in self.validDF['action_label'].value_counts().keys()}
            validation_accuracy=0.0
            
            for i,pred_tag in enumerate(pred_tags):
                if pred_tag==valid_tags[i]:
                    validation_accuracy+=1.0
                    classwise_accuracy[pred_tag]+=1.0
            count_dict=self.validDF['action_label'].value_counts()
            
            for c,v in classwise_accuracy.items():
                classwise_accuracy[c]=100*v/count_dict[c]
            
            validation_accuracy=validation_accuracy/len(pred_tags)
            validation_accuracy=validation_accuracy*100

            val_accuracies.append(validation_accuracy)            
            train_loss.append(runningLoss)
            # df=pd.DataFrame({'pred_tags':pred_tags,'valid_tags':valid_tags})
            # df.to_csv('val.csv')
            epochTime = time.time() - epochStart
            epochLoss = runningLoss / len(self.loader)
            
            print("-> Training time: {:.4f}s, loss = {:.4f} ValAccuracy={}".format(epochTime, epochLoss,validation_accuracy)) 
            print('Class Wise Accuracy',classwise_accuracy)
        self.saveCheckpoint(checkpointFilename)
        self.saveLearningCurves(train_loss,checkpointFilename)
        self.saveLearningCurves(val_accuracies,checkpointFilename,label='Validation Accuracy',ylabel='Accuracy')
    
    def saveCheckpoint(self,checkpointFilename):
        checkpoint=self.model.state_dict()
        torch.save(checkpoint,'./Checkpoints/'+checkpointFilename+'.pth')
        print('Checkpoint Saved Successfuly')
    
    def saveLearningCurves(self,train_loss,checkpointFilename,label='Training Loss',ylabel='Loss'):
        
        sns.set(style='darkgrid')
        # Increase the plot size and font size.
        sns.set(font_scale=1.5)
        plt.rcParams["figure.figsize"] = (12,6)

        # Plot the learning curve.
        plt.plot(train_loss, 'b-o', label=label)

        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.legend()
        plt.savefig(f'./LearningCurve/{checkpointFilename}_{label}.png')
        plt.close()

# model=OGCLFModel(512,num_labels=4)
# # model=CLFModel(50,512,512,num_labels=14)
# # trainer=OGCLFTrainer("Datasets/new_large_train.csv", model,True)
# trainer=OGCLFTrainer("Datasets/new_large_train.csv", model)
# trainer.trainDataset('hola')
# # # # trainer.trainDataset('OGCLFModel')

