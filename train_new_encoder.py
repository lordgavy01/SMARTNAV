import pandas as pd
from collections import defaultdict
import torch
from torch import nn
from transformers import AdamW,get_linear_schedule_with_warmup
from tqdm import tqdm 
import time

import seaborn as sns
import matplotlib.pyplot as plt
from random import randint

from Models.model_new_encoder import OGModel

class LidarDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]).float() for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings['X1'])

class OGTrainer:
    def __init__(self,filename,model):
        self.trainDF=pd.read_csv(filename).drop_duplicates().sample(frac=1)
    
        self.cleanDataset()
        self.model=model
    
    def cleanDataset(self):
        inputs=defaultdict(list)
        inputs["X2"] = self.trainDF[["distance_from_goal", "angle_from_goal"]].to_numpy().tolist()
        embedding_columns = [f"embedding_dim_{i}" for i in range(512)]
        inputs["X1"] = self.trainDF[embedding_columns].to_numpy().tolist()
        inputs["Y"] = self.trainDF[['output_linear_velocity', 'output_angular_velocity']].to_numpy().tolist()
        self.loadDataset(inputs)

    def loadDataset(self,inputs):
        self.dataset=LidarDataset(inputs)   
        self.loader=torch.utils.data.DataLoader(self.dataset, batch_size=4096, shuffle=True,pin_memory=True)
    
    def trainDataset(self,checkpointFilename,LearningRate=1e-3):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # device = torch.device('cpu')
        self.model=self.model.to(device)
        LearningRate=1e-3
        Epochs=5

        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
                {
                        'params':[p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                        'weight_decay':0.02
                },
                {
                        'params':[p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                        'weight_decay':0.0
                }
        ]
        # initialize optimizer
        optimizer = AdamW(optimizer_grouped_parameters, lr=LearningRate ,betas=(0.9, 0.96), eps=1e-06)

        criterion=nn.MSELoss()
        total_steps = len(self.loader) * Epochs
        # adding a scheduler to linearly reduce the learning rate throughout the epochs.
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=4*total_steps)


        train_loss=[]
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
                predicted = self.model(batch_X1,batch_X2)
                loss=criterion(predicted,batch_Y)
                # extract loss
                runningLoss += loss.item()
                # Update Weights
                loss.backward()
                optimizer.step()
                scheduler.step()
                # print relevant info to progress bar
                loop.set_description(f'Epoch {epoch}')
                loop.set_postfix(RunningLoss=runningLoss/(batch_index+1))
            train_loss.append(runningLoss)
            epochTime = time.time() - epochStart
            epochLoss = runningLoss / len(self.loader)
        
            print("-> Training time: {:.4f}s, loss = {:.4f} ".format(epochTime, epochLoss)) 
            self.saveCheckpoint(checkpointFilename+f'{epoch}')
            self.saveLearningCurves(train_loss,checkpointFilename+f'{epoch}')
    
    def saveCheckpoint(self,checkpointFilename):
        checkpoint=self.model.state_dict()
        torch.save(checkpoint,'./Checkpoints/'+checkpointFilename+'.pth')
        print('Checkpoint Saved Successfuly')

    def saveLearningCurves(self,train_loss,checkpointFilename):
        sns.set(style='darkgrid')
        # Increase the plot size and font size.
        sns.set(font_scale=1.5)
        plt.rcParams["figure.figsize"] = (12,6)

        # Plot the learning curve.
        plt.plot(train_loss, 'b-o', label="Training Loss")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig('./LearningCurve/'+checkpointFilename+'.png')


# model=OGModel(512)

# trainer=OGTrainer("Datasets/new_large_train.csv", model)
# trainer.trainDataset('GGModel')

