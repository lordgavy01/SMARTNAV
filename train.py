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

from old_model import OldModel

class LidarDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]).float() for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings['X1'])

class MyTrainer:
    def __init__(self,filename,model):
        self.trainDF=pd.read_csv(filename)
        self.cleanDataset()
        self.model=model
    def cleanDataset(self):
        inputs=defaultdict(list)
        for i in range(len(self.trainDF)):
            inputs['Y'].append([self.trainDF.iloc[i]['output_linear_velocity'],self.trainDF.iloc[i]['output_angular_velocity']])
            inputs['X2'].append([self.trainDF.iloc[i]['distance_from_goal'],self.trainDF.iloc[i]['angle_from_goal']])
            depths=[]
            name='lidar_depth_'
            for j in range(1,51):
                nname=name+str(j)
                depths.append(self.trainDF.iloc[i][nname])
            inputs['X1'].append(depths.copy())
        self.loadDataset(inputs)
    def loadDataset(self,inputs):
        self.dataset=LidarDataset(inputs)   
        self.loader=torch.utils.data.DataLoader(self.dataset, batch_size=64, shuffle=True,pin_memory=True)
    
    def trainDataset(self,checkpointFilename):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # device = torch.device('cpu')
        self.model=self.model.to(device)
        LearningRate=6e-4
        Epochs=6

        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
                {
                        'params':[p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                        'weight_decay':0.01
                },
                {
                        'params':[p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                        'weight_decay':0.0
                }
        ]
        # initialize optimizer
        optimizer = AdamW(optimizer_grouped_parameters, lr=LearningRate ,betas=(0.9, 0.98), eps=1e-06)

        criterion=nn.MSELoss()
        total_steps = len(self.loader) * Epochs
        # adding a scheduler to linearly reduce the learning rate throughout the epochs.
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=total_steps)


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
        self.saveCheckpoint(checkpointFilename)
        self.saveLearningCurves(train_loss,checkpointFilename)
    
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


model=OldModel(50,2,512,512,64)

trainer=MyTrainer("Datasets/train.csv", model)
trainer.trainDataset('oldModel')

