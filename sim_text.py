from util import *
from lidar import *
from environment import *
from agent import *
import csv
import time
from policy import Policy
from lidar_converter import *
from subgoals import get_subgoals
import torch
import logging

logging.basicConfig(filename=f"./text.log",format='%(asctime)s %(message)s', filemode='w')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


data=[]
with open('text_result.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Map', 'Timestamps', 'Path Clearance', 'Avg Goal Distance'])

DATAFILENAME='./Datasets/text_train.csv'

fields=["output_linear_velocity","output_angular_velocity","distance_from_goal","angle_from_goal"]
fields+=['prompt']
fields+=[f"embedding_dim_{i}" for i in range(512)]

with open(DATAFILENAME,'w') as csvfile: 
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)
    
# MAPS=['Map10', 'Map5', 'Map-5', 'Map6', 'Map8', 'Map3', 'Map2', 'Map1']
MAPS=['Map10', 'Map11', 'Map5', 'Map-5', 'Map6', 'Map8', 'Map9', 'Map3', 'Map2', 'Map1']

SUBGOALS=[get_subgoals(maps) for maps in MAPS]
print(f'Number of maps to be tested: {len(MAPS)}')

# DATAFILENAME=f'./Results/test_{model_name}.csv'
obstacles=[]
mapBackgrounds=[]
for maps in MAPS:
    obstacles.append(initMap(mapObstaclesFilename=f"Maps/{maps}.txt"))
    mapBackgrounds.append(getMapBackground(mapImageFilename=f"Maps/{maps}.png"))
paused=False
env=Environment()
env.reset(obstacles=obstacles[0],agentRadius=AGENT_RADIUS,agentSubGoals=SUBGOALS[0])
pygame.init()
pygame.display.set_caption(f"Tester")
screen=pygame.display.set_mode((mapBackgrounds[0].image.get_width(),mapBackgrounds[0].image.get_height()))
screen.blit(mapBackgrounds[0].image, mapBackgrounds[0].rect)
env.renderSubGoals(screen)
pygame.display.update()
rows=[]
for j in range(len(MAPS)):
    logger.info(f'Starting Map: {j+1}')
   
    print(f'Starting Map: {j+1}')
    env=Environment()
    env.reset(obstacles=obstacles[j],agentSubGoals=SUBGOALS[j])
    print(f'Loaded Environment')
    screen=pygame.display.set_mode((mapBackgrounds[j].image.get_width(),mapBackgrounds[j].image.get_height()))
    screen.blit(mapBackgrounds[j].image, mapBackgrounds[j].rect)
    running=True
    collisionFlag=False
    numTimestamps=0
    pathClearance=INF
    sumGoalDistance=0
    APF_PARAMS=init_params(MAPS[j])
    last_positions=[]
    temporarily_stop=False
    cooldown=0
    while running:
        screen.blit(mapBackgrounds[j].image, mapBackgrounds[j].rect)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                exit(0)
                break
            elif event.type == pygame.MOUSEBUTTONDOWN:
                paused ^= 1
            
        if paused:
            continue
        
        numTimestamps+=1
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            cooldown = 1
        else:
            cooldown = 0
        apfAction=env.agentStates[0].selectAction(algorithm="APF",apfParams=APF_PARAMS)
        # apfAction,action_index=get_nearest_action(np.array(apfAction),action_space)
        # lidarAngles,lidarDepths=env.agentStates[0].lidarData
        # generate_lidar_image(lidarDepths)
        # embedding=get_clip_embeddings()

        
        
        if cooldown:
            apfAction=(0.0,0.0)
            cooldown-=1

        action=apfAction

        pose=env.agentPoses[0]

        oldPoses,newPoses=env.executeAction(action,apfParams=APF_PARAMS,noise=0.1)
        env.render(screen)
        pygame.display.update()

        
        pathClearance=min(pathClearance,env.getAgentClearances()[0])
        goalDistance=euclidean((env.agentPoses[0][0],env.agentPoses[0][1]),(env.agentGoals[0][0],env.agentGoals[0][1]))
        sumGoalDistance+=goalDistance
        if len(last_positions) >= 20:
            last_positions.pop(0)  # remove the oldest position

        if cooldown!=0:
            last_positions.append(pose)
        prompt=env.generate_prompt(oldPoses,newPoses)
        if len(prompt):
            if cooldown==0:
                cooldown=10
            else:
                cooldown+=2
        # row=[apfAction[0],apfAction[1],
        #         env.agentStates[0].distanceGoal,
        #         env.agentStates[0].thetaGoal,prompt
        #         ]+embedding
        # rows.append(row)
        # if cooldown!=0 and len(last_positions) == 20 and check_convergence(last_positions, 3):
        #     collisionFlag=True
        #     print(f"Converged")
        #     break

        if(numTimestamps>15000):
            logger.info("Time Limit Exceeded")
            collisionFlag=True
            break  
        if(env.getAgentClearances()[0]==-1):
            logger.info("Robot Collided!!!")
            collisionFlag=True
            break
        
        if goalDistance<12:
            if (env.agentProgress[0]+1)==(len(env.agentSubGoals[0])-1):
                logger.info("Robot reached Goal!")
                running=False
                break

    if not collisionFlag:
        logger.info(f'Stats for Map {j}')
        logger.info(f"Number of timestamps: {numTimestamps//10}")
        logger.info(f"Path Clearance: {pathClearance}")
        logger.info(f"Average Goal Distance along Path: {sumGoalDistance/numTimestamps}")
        logger.info('*'*20)
        # Add the data into the data list
        data.append([MAPS[j], numTimestamps, pathClearance, sumGoalDistance/numTimestamps])
    # if len(rows):
    #     print(f"Adding {len(rows)} rows to database.")
    #     with open(DATAFILENAME,'a') as csvfile: 
    #         csvwriter = csv.writer(csvfile)
    #         csvwriter.writerows(rows)
    #     rows=[]
# Create the csv file
with open('text_result.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)