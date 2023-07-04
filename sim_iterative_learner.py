from util import *
from lidar import *
from environment import *
from agent import *
import csv
import time
from lidar_converter import *
from subgoals import get_subgoals
from policy import Policy

start_time=time.time()
NUM_MAPS=11
MAPS=[f'Map{i}' for i in range(-5,NUM_MAPS+1) if i not in [6,7]]
SUBGOALS=[get_subgoals(maps) for maps in MAPS]

APF_DATA_ITER=150
APF_DATA_NO_ROTATE_KEEP=0.4
USE_CHECKPOINT=False
GOAL_DISTANCE_THRESHOLD=6
FILE_NUM=7


checkpointFilename='Checkpoints/OGCLFModel.pth'
tempDataFilename = f"Datasets/iter_data_temp.csv"
tempCheckpointFilename= f"iter_cp_temp"

obstacles=[]
mapBackgrounds=[]
ALL_APF_PARAMS=[]
for map in MAPS:
    if map in [7,8]:
        continue
    obstacles.append(initMap(mapObstaclesFilename=f"Maps/{map}.txt"))
    mapBackgrounds.append(getMapBackground(mapImageFilename=f"Maps/{map}.png"))
    ALL_APF_PARAMS.append(init_params(map))

env=Environment()
env.reset(obstacles=obstacles[0],agentRadius=AGENT_RADIUS,agentSubGoals=SUBGOALS[0])
pygame.init()
pygame.display.set_caption(f"DAgger")
screen=pygame.display.set_mode((mapBackgrounds[0].image.get_width(),mapBackgrounds[0].image.get_height()))
screen.blit(mapBackgrounds[0].image, mapBackgrounds[0].rect)
env.renderSubGoals(screen)
pygame.display.update()
time.sleep(2)

fields=["output_linear_velocity","output_angular_velocity","distance_from_goal","angle_from_goal"]
fields+=['action_label']
fields+=[f"embedding_dim_{i}" for i in range(512)]

with open(tempDataFilename,'w') as csvfile: 
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)

pathNumTimestamps=[[] for _ in range(len(MAPS))]
pathClearances=[[] for _ in range(len(MAPS))]
pathAvgGoalDistances=[[] for _ in range(len(MAPS))]

ctrAllPassed=0
keepIterating=True
model_name='OGCLFModel'
policy=Policy(model_name)
NUM_ITERATIONS=50
LR=1e-2

v_split=3
w_split=6
V_values = np.linspace(-VMAX, VMAX, v_split)
W_values = np.linspace(-WMAX/2, WMAX/2, w_split)
action_space = np.array(np.meshgrid(V_values, W_values)).T.reshape(-1, 2)


for i in range(1,1+NUM_ITERATIONS):
    allMapsPassed=True
    mapsPassed=[] 
    print(f"\n***Iteration {i}***")
    if i==1 and USE_CHECKPOINT:
        policy.loadModel(tempCheckpointFilename)
    if i!=1:
        policy.learnAndSaveModel(tempDataFilename,tempCheckpointFilename,LR/100*i)
        policy.loadModel(tempCheckpointFilename)
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
        if i==1:
            isApfOn=1
        robotColor=(255,0,0)
        collisionFlag=False
        numTimestamps=0
        pathClearance=INF
        sumGoalDistance=0
        last_positions=[]
        while running:

            screen.blit(mapBackgrounds[j].image, mapBackgrounds[j].rect)
            for event in pygame.event.get():
                if event.type==pygame.QUIT:
                    running=False
                    collisionFlag=True
                    keepIterating=False
                    break
                if event.type==pygame.KEYDOWN:
                    key_name=pygame.key.name(event.key)
                    key_name=key_name.upper()
                    if(key_name=="RETURN"):
                        isApfOn=1-isApfOn
                        if isApfOn:
                            robotColor=(0,255,0)
                        else:
                            robotColor=(255,0,0)

            lidarAngles,lidarDepths=env.agentStates[0].lidarData
            generate_lidar_image(lidarDepths)
            embedding=get_clip_embeddings()
            if isApfOn:
                action=env.agentStates[0].selectAction(algorithm="APF",apfParams=ALL_APF_PARAMS[j])
            else:    
                action=env.agentStates[0].selectAction(algorithm="OGCLF",policy=policy,embeddings=embedding)
                action=action_space[action]

            apfAction=env.agentStates[0].selectAction(algorithm="APF",apfParams=ALL_APF_PARAMS[j])
            apfAction,action_index=get_nearest_action(np.array(apfAction),action_space)
        
            row=[apfAction[0],apfAction[1],
            env.agentStates[0].distanceGoal,
            env.agentStates[0].thetaGoal,action_index
            ]+embedding
            pose=env.agentPoses[0]

            if len(last_positions) >= 20:
                last_positions.pop(0)  # remove the oldest position

            last_positions.append(pose)

            if len(last_positions) == 20 and check_convergence(last_positions, 3):
                rows=rows[:-20]
                collisionFlag=True
                print(f"Converged")
                break


            env.executeAction(action,noise=0.1,goalDistanceThreshold=GOAL_DISTANCE_THRESHOLD,apfParams=ALL_APF_PARAMS[j])
            env.render(screen,robotColor)
            pygame.display.update()

            numTimestamps+=1
            pathClearance=min(pathClearance,env.getAgentClearances()[0])
            goalDistance=euclidean((env.agentPoses[0][0],env.agentPoses[0][1]),(env.agentGoals[0][0],env.agentGoals[0][1]))
            sumGoalDistance+=goalDistance

            if isApfOn:
                rows.append(row)
            else:
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
                if isApfOn:
                    rows.pop()
                print("Robot Collided!!!")
                collisionFlag=True
                break
            
            if goalDistance<GOAL_DISTANCE_THRESHOLD:
                if (env.agentProgress[0]+1)==(len(env.agentSubGoals[0])-1):
                    print("Robot reached Goal!")
                    running=False
                    break
            
            if(numTimestamps>800):
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

    if allMapsPassed and i!=1:
        ctrAllPassed+=1
        curFile=f"WorkingCheckpoints/iter_checkpoint_{FILE_NUM}_{APF_DATA_ITER}_{APF_DATA_NO_ROTATE_KEEP}_V_{ctrAllPassed}.pth"
        policy.saveModel(curFile)

    print("Execution Time since start:",(time.time()-start_time),"s")
    print("mapsPassed:",mapsPassed)
    print("ctrAllPassed:",ctrAllPassed)
    pygame.quit()



print()
print(pathNumTimestamps)
print(pathClearances)   
print(pathAvgGoalDistances)    

print("Execution Time:",(time.time()-start_time)/60,"mins")