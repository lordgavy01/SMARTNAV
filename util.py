from math import *
from config import *
import numpy as np
import pygame
import os
import random
from lidar_converter import *
from generate_embedding import get_clip_embeddings,get_clip_text_embeddings
import pandas as pd 
import random

INF=1e18

os.environ["SDL_VIDEO_CENTERED"] = "1"
np.random.seed(1)
class Background(pygame.sprite.Sprite):
    def __init__(self, image_file, location):
        pygame.sprite.Sprite.__init__(self)  #call Sprite initializer
        self.image = pygame.image.load(image_file)
        self.rect = self.image.get_rect()
        self.rect.left, self.rect.top = location

def getMapBackground(mapImageFilename=MAP_IMAGE_FILENAME):
    mapBackground=Background(mapImageFilename,[0,0])
    return mapBackground

def initMap(mapObstaclesFilename=MAP_OBSTACLES_FILENAME):
    obstacles=[]
    with open(mapObstaclesFilename) as file:
        for line in file:
            line_data=[int(x) for x in line.split()]
            obstacle=[]
            for i in range(0,len(line_data),2):
                obstacle.append((line_data[i],line_data[i+1]))
                obstacles.append(obstacle)
    return obstacles

def euclidean(A,B):
    return sqrt((A[0]-B[0])**2+(A[1]-B[1])**2)

def normalAngle(theta):
  return atan2(sin(theta),cos(theta))

def getLinesegmentsIntersection(line1,line2):
    ((x1,y1),(x2,y2))=line1
    ((x3,y3),(x4,y4))=line2
    denom = (y4-y3)*(x2-x1) - (x4-x3)*(y2-y1)
    if denom == 0: # parallel
        return False
    ua = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / denom
    if ua < 0 or ua > 1: # out of range
        return False
    ub = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / denom
    if ub < 0 or ub > 1: # out of range
        return False
    x = x1 + ua * (x2-x1)
    y = y1 + ua * (y2-y1)
    return (x,y)

def kinematic_equation(oldPose,v,w,dT):
    xOld,yOld,thetaOld=oldPose
    if w==0:
        xNew=xOld+v*dT*cos(thetaOld)
        yNew=yOld+v*dT*sin(thetaOld)
        thetaNew=thetaOld
        return (xNew,yNew,thetaNew)
    dTheta=w*dT
    L=v/w
    ICCx=xOld-L*sin(thetaOld)
    ICCy=yOld+L*cos(thetaOld)
    xNew=(cos(dTheta)*(xOld-ICCx))-(sin(dTheta)*(yOld-ICCy))+ICCx
    yNew=(sin(dTheta)*(xOld-ICCx))+(cos(dTheta)*(yOld-ICCy))+ICCy
    thetaNew=thetaOld+dTheta
    thetaNew=atan2(sin(thetaNew),cos(thetaNew))
    return (xNew,yNew,thetaNew)

def addForces(F1,F2):
    magnitude1,theta1=F1
    magnitude2,theta2=F2
    x_comp=magnitude1*cos(theta1)+magnitude2*cos(theta2)
    y_comp=magnitude1*sin(theta1)+magnitude2*sin(theta2)
    magnitudeRes=sqrt(x_comp**2+y_comp**2)
    thetaRes=normalAngle(atan2(y_comp,x_comp))
    return (magnitudeRes,thetaRes)

def getLinesegmentCircleIntersection(line1,circle):
    p1,p2=line1
    center,radius=circle
    
    Q=pygame.math.Vector2(center[0],center[1])
    r=radius
    P1=pygame.math.Vector2(p1[0],p1[1])
    V=pygame.math.Vector2(p2[0]-p1[0],p2[1]-p1[1])
    a = V.dot(V)
    b = 2 * V.dot(P1 - Q)
    c = P1.dot(P1) + Q.dot(Q) - 2 * P1.dot(Q) - r**2
    disc = b**2 - 4 * a * c
    if disc < 0:
        return False
    sqrt_disc = sqrt(disc)
    t1 = (-b + sqrt_disc) / (2 * a)
    t2 = (-b - sqrt_disc) / (2 * a)
    if not (0 <= t1 <= 1 or 0 <= t2 <= 1):
        return False
    inter_p1=P1+t1*V
    inter_p2=P1+t2*V
    if euclidean(p1,inter_p1)<euclidean(p1,inter_p2):
        return inter_p1
    return inter_p2

def getDistancePointLineSegment(point,line_segment):
    ((x1,y1),(x2,y2))=line_segment
    (x3,y3)=point
    px = x2-x1
    py = y2-y1
    norm = px*px + py*py
    u =  ((x3 - x1) * px + (y3 - y1) * py) / float(norm)
    if u > 1:
        u = 1
    elif u < 0:
        u = 0
    x = x1 + u * px
    y = y1 + u * py
    dx = x - x3
    dy = y - y3
    dist = sqrt(dx*dx + dy*dy)
    return dist

def prepare_index(clusterPath):
    trainDF = pd.read_csv(clusterPath)
    centroids = trainDF.groupby("refined_predicted_community")["image_embeddings"].apply(lambda x: np.mean(np.stack(x.values))).to_dict()

    return centroids


def check_convergence(last_positions, threshold):
    for i in range(len(last_positions)):
        for j in range(i + 1, len(last_positions)):
            if euclidean(last_positions[i],last_positions[j]) > threshold:
                return False
    return True

def get_nearest_action(apf_action, action_space):
    distances = np.linalg.norm(action_space - apf_action, axis=1)
    closest_index = np.argmin(distances)
    closest_action = action_space[closest_index]
    return closest_action, closest_index

def update_subgoals(subgoals_orig):
    subgoals=subgoals_orig.copy()
    eps=random.random()
    if eps<0.4:
        random.shuffle(subgoals)
    else:
        subgoals=subgoals_orig.copy()
        eps=random.random()
        
        subset_size = random.randint(1, max(1,len(subgoals) - 1))
        sublist_indices = random.sample(range(len(subgoals)), subset_size)
        for idx in sublist_indices:
            subgoals[idx] = subgoals[idx][::-1]
        
    return subgoals

    

