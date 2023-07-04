from util import *
from planner import *
import torch.nn.functional as F

class AgentState:

    # pose, goal --> x,y,theta(in radians)
    # lidarData --> 2 lists : lidarAngles(in radians), lidarDepths

    def __init__(self,distanceGoal,thetaGoal,lidarData,velocity,pose):
        self.distanceGoal=distanceGoal        
        self.thetaGoal=thetaGoal
        self.lidarData=lidarData
        self.velocity=velocity
        self.pose=pose
    
    def update(self,distanceGoal,thetaGoal,lidarData,velocity,pose):
        self.distanceGoal=distanceGoal        
        self.thetaGoal=thetaGoal
        self.lidarData=lidarData
        self.velocity=velocity
        self.pose=pose

    # action --> (linearVelocity,angularVelocity)
    def selectAction(self,apfParams=None,algorithm="APF",policy=None,embeddings=None):
        if algorithm=="APF":
            return APF(self.distanceGoal,self.thetaGoal,self.lidarData,self.velocity,apfParams=apfParams)
        elif algorithm=="NN":
            _,lidarDepths=self.lidarData
            bestAction=policy.act(lidarDepths,[self.distanceGoal,self.thetaGoal],[self.pose])
            return bestAction[0].tolist()
        elif algorithm=="OG":
            bestAction=policy.act(embeddings,[self.distanceGoal,self.thetaGoal])
            return bestAction[0].tolist()
        elif algorithm=="OGCLF":
            out_classes=policy.act(embeddings,[self.distanceGoal,self.thetaGoal])
            return out_classes[0].tolist()  
                
        return None