from Models.model_new_encoder import *
from Models.old_model import *
from util import *
from train_new_encoder import *
from train_clf_encoder import *
class Policy():
    def __init__(self,model_name='oldModel',clusterPath=None):
        self.model_name=model_name
        if model_name=='OGModel':
            self.model=OGModel(512)
        elif model_name=='oldModel':
            self.model=OldModel(50,2,512,512,64)
        elif model_name=='OGCLFModel':
            self.model=OGCLFModel(512,2)

    # Uses data in apf_data.csv to optimize weights of current NN model object
    # The new learnt weights are written in file Checkpoint.pth
    def learnAndSaveModel(self,dataFilename="apf_data.csv",checkpointFilename="checkpoint.pth",LR=1e-3):
        
        if self.model_name=='oldModel':
            trainer=MyTrainer(dataFilename,self.model)
        elif self.model_name=='OGModel':
            trainer=OGTrainer(dataFilename, self.model)
        if self.model_name=='OGCLFModel':
            trainer=OGCLFTrainer(dataFilename,self.model)
        trainer.trainDataset(checkpointFilename,LR)
    
    def saveModel(self,checkpointFilename):
        torch.save(self.model.state_dict(),checkpointFilename)
        print('Checkpoint Saved Successfuly')
    
    # Loads weights from file Checkpoint.pth to the current NN model object to make
    # it ready to use
    def loadModel(self,checkpointFilename='checkpoint'):
        self.model.load_state_dict(torch.load('./Checkpoints/'+checkpointFilename+'.pth'))
        self.model.eval()

    def act(self,lidarDepths,disAndAngle,pose=None):
        input1=torch.tensor([lidarDepths]).float()
        input2=torch.tensor([disAndAngle]).float()
        if self.model_name in ['oldModel','OGModel']:
            return self.model.forward(input1,input2)
        elif self.model_name=='OGCLFModel':
            _,out_classes=self.model.get_class_probabilities(input1, input2)
            return out_classes.indices.detach().cpu().numpy()
        cluster=0
        input3=torch.tensor([cluster]).float()
        return self.model.forward(input1,input2,input3)
