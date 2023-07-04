from util import *
from lidar import *
from planner import *
from environment import *
from agent import *
from config import *
import csv
import time
import matplotlib.pyplot as plt

start_time=time.time()

MAPS=["small_map","small_map2","map2","map3","map4","squares_map"]
SUBGOALS=[AGENT_SUBGOALS2,AGENT_SUBGOALS3,AGENT_SUBGOALS4,AGENT_SUBGOALS5,AGENT_SUBGOALS6,AGENT_SUBGOALS7]
APF_PARAMS=[APF_PARAMS_1,APF_PARAMS_2,APF_PARAMS_1,APF_PARAMS_1,APF_PARAMS_3,APF_PARAMS_4]

APF_DATA_ITER=150
APF_DATA_NO_ROTATE_KEEP=0.4
USE_CHECKPOINT=True
GOAL_DISTANCE_THRESHOLD=6
FILE_NUM=7

obstacles=[]
mapBackgrounds=[]
for map in MAPS:
    obstacles.append(initMap(mapObstaclesFilename=f"Maps/{map}_obstacles.txt"))
    mapBackgrounds.append(getMapBackground(mapImageFilename=f"Maps/{map}.png"))

env=Environment()
env.reset(obstacles=obstacles[0],agentRadius=AGENT_RADIUS,agentSubGoals=SUBGOALS[0])
pygame.init()
pygame.display.set_caption(f"DAgger")
screen=pygame.display.set_mode((mapBackgrounds[0].image.get_width(),mapBackgrounds[0].image.get_height()))
screen.blit(mapBackgrounds[0].image, mapBackgrounds[0].rect)
env.renderSubGoals(screen)
pygame.display.update()
time.sleep(2)

apfDataFilename = f"Datasets/apf_data_{FILE_NUM}_{APF_DATA_ITER}_{APF_DATA_NO_ROTATE_KEEP}.csv"
checkpointFilename= f"Checkpoints/checkpoint_{FILE_NUM}_{APF_DATA_ITER}_{APF_DATA_NO_ROTATE_KEEP}.pth"
tempDataFilename = f"Datasets/iter_data_{FILE_NUM}_{APF_DATA_ITER}_{APF_DATA_NO_ROTATE_KEEP}.csv"
tempCheckpointFilename= f"Checkpoints/iter_checkpoint_{FILE_NUM}_{APF_DATA_ITER}_{APF_DATA_NO_ROTATE_KEEP}.pth"

fields=["output_linear_velocity","output_angular_velocity","distance_from_goal","angle_from_goal"]
fields+=[f"lidar_depth_{i}" for i in range(1,1+NUMBER_OF_LIDAR_ANGLES)]
with open(tempDataFilename,'w') as csvfile: 
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)

pathNumTimestamps=[[] for _ in range(len(MAPS))]
pathClearances=[[] for _ in range(len(MAPS))]
pathAvgGoalDistances=[[] for _ in range(len(MAPS))]

ctrAllPassed=0
keepIterating=True
policy=Policy()
NUM_ITERATIONS=50
for i in range(1,1+NUM_ITERATIONS):
    allMapsPassed=True
    mapsPassed=[] 
    print(f"\n***Iteration {i}***")
    if USE_CHECKPOINT:
        if i==1:
            policy.loadModel(checkpointFilename)
        else:
            policy.learnAndSaveModel(tempDataFilename,tempCheckpointFilename)
    else:
        if i==1:
            policy.learnAndSaveModel(apfDataFilename,checkpointFilename)
        else:
            policy.learnAndSaveModel(tempDataFilename,tempCheckpointFilename)
            

    rows=[]
    for j in range(len(MAPS)):
        print(f"\n*Map {j}*")
        env=Environment()
        env.reset(obstacles=obstacles[j],agentRadius=AGENT_RADIUS,agentSubGoals=SUBGOALS[j])
        pygame.display.set_caption(f"DAgger: Iteration {i} Map {j}")
        screen=pygame.display.set_mode((mapBackgrounds[j].image.get_width(),mapBackgrounds[j].image.get_height()))
        screen.blit(mapBackgrounds[j].image, mapBackgrounds[j].rect)
        running=True
        lastProgress=0
        isApfOn=0
        robotColor=(255,0,0)
        collisionFlag=False
        numTimestamps=0
        pathClearance=INF
        sumGoalDistance=0
    
        while running:
            screen.blit(mapBackgrounds[j].image, mapBackgrounds[j].rect)
            for event in pygame.event.get():
                if event.type==pygame.QUIT:
                    running=False
                    keepIterating=False
                    break
                if event.type==pygame.KEYDOWN:
                    key_name=pygame.key.name(event.key)
                    key_name=key_name.upper()
                    print(key_name)  
                    if(key_name=="RETURN"):
                        isApfOn=1-isApfOn
                        if isApfOn==1:
                            robotColor=(0,0,0)
                        else:
                            robotColor=(255,0,0)

            if(isApfOn==0):
                action=env.agentStates[0].selectAction("NN",policy)
            else:
                action=env.agentStates[0].selectAction("APF",apfParams=APF_PARAMS[j])
            apfAction=env.agentStates[0].selectAction("APF",apfParams=APF_PARAMS[j])
            row=[apfAction[0],apfAction[1],
                env.agentStates[0].distanceGoal,
                env.agentStates[0].thetaGoal,
                ]+env.agentStates[0].lidarData[1]
            reward=env.executeAction(action,noise=0.1,goalDistanceThreshold=GOAL_DISTANCE_THRESHOLD)
            env.render(screen,robotColor)
            pygame.display.update()

            numTimestamps+=1
            pathClearance=min(pathClearance,env.getAgentClearances()[0])
            goalDistance=euclidean((env.agentPoses[0][0],env.agentPoses[0][1]),(env.agentGoals[0][0],env.agentGoals[0][1]))
            sumGoalDistance+=goalDistance

            if(abs(row[1])<abs(radians(1))):
                epsilon=random.uniform(0,1)
                if(epsilon<=APF_DATA_NO_ROTATE_KEEP):
                    rows.append(row)
            else:
                rows.append(row)

            if env.getAgentClearances()[0]==-1 or env.getAgentClearances()[0]<=1.5:
                for _ in range(4):
                    rows.append(row)
            elif env.getAgentClearances()[0]<=3:
                for _ in range(2):
                    rows.append(row)

            if(env.getAgentClearances()[0]==-1):
                print(env.getAgentClearances())
                print("Robot Collided!!!")
                collisionFlag=True
                break
            
            if goalDistance<GOAL_DISTANCE_THRESHOLD:
                if (env.agentProgress[0]+1)==(len(env.agentSubGoals[0])-1):
                    print("Robot reached Goal!")
                    running=False
                    break
            
            if(numTimestamps>400):
                print("Time Limit Exceeded")
                collisionFlag=True
                break  

        if not keepIterating:
            break

        if not collisionFlag:
            print()
            print(f"Number of timestamps: {numTimestamps}")
            print(f"Path Clearance: {pathClearance}")
            print(f"Average Goal Distance along Path: {sumGoalDistance/numTimestamps}")
            print(f"Number of learning iterations: {i}")
            pathNumTimestamps[j].append(numTimestamps)
            pathClearances[j].append(pathClearance)
            pathAvgGoalDistances[j].append(sumGoalDistance/numTimestamps)
            mapsPassed.append(1)
        else:
            allMapsPassed=False
            mapsPassed.append(0)
            pathNumTimestamps[j].append(INF)
            pathClearances[j].append(0)
            pathAvgGoalDistances[j].append(sumGoalDistance/numTimestamps)
    
    if not keepIterating:
        break

    print(f"Adding {len(rows)} rows to database.")
    with open(tempDataFilename,'a') as csvfile: 
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(rows)

    if allMapsPassed:
        ctrAllPassed+=1
        curFile="WorkingCheckpoints/iter_checkpoint_{FILE_NUM}_{APF_DATA_ITER}_{APF_DATA_NO_ROTATE_KEEP}_V_{ctrAllPassed}.pth"
        policy.saveModel(curFile)

    print("Execution Time since start:",(time.time()-start_time),"s")
    print("mapsPassed:",mapsPassed)
    print("ctrAllPassed:",ctrAllPassed)

    # if i>=15 and allMapsPassed==True:
    #     break

print()
print(pathNumTimestamps)
print(pathClearances)   
print(pathAvgGoalDistances)    

print("Execution Time:",(time.time()-start_time)/60,"mins")