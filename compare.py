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



model_name='OGModel'
# checkpointFilename='GGModel4'

# policy=Policy(model_name)
# policy.loadModel(checkpointFilename)

logging.basicConfig(filename=f"./Results/Testing/{model_name}.log",format='%(asctime)s %(message)s', filemode='w')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

data=[]
with open('result.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Map', 'Timestamps', 'Path Clearance', 'Avg Goal Distance'])

NUM_MAPS=11
# MAPS=[f'Map{i}' for i in range(-5,NUM_MAPS+1)]
MAPS=['Map9']
# MAPS=['Map10', 'Map11', 'Map5', 'Map-5', 'Map6', 'Map8', 'Map9', 'Map3', 'Map2', 'Map1']

SUBGOALS=[get_subgoals(maps) for maps in MAPS]
logger.info(f'Number of maps to be tested: {len(MAPS)}')

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

# v_split=3
# w_split=6
# V_values = np.linspace(-VMAX, VMAX, v_split)
# W_values = np.linspace(-WMAX/2, WMAX/2, w_split)
# action_space = np.array(np.meshgrid(V_values, W_values)).T.reshape(-1, 2)

good_maps=[]
for j in range(len(MAPS)):
    logger.info(f'Starting Map: {MAPS[j]}')
    print(f'Starting Map: {j+1}')
    env=Environment()
    env.reset(obstacles=obstacles[j],agentSubGoals=SUBGOALS[j])
    logger.info(f'Loaded Environment')
    screen=pygame.display.set_mode((mapBackgrounds[j].image.get_width(),mapBackgrounds[j].image.get_height()))
    screen.blit(mapBackgrounds[j].image, mapBackgrounds[j].rect)
    running=True
    collisionFlag=False
    numTimestamps=0
    pathClearance=INF
    sumGoalDistance=0
    APF_PARAMS=init_params(MAPS[j])
    last_positions=[]
    while running:
        screen.blit(mapBackgrounds[j].image, mapBackgrounds[j].rect)
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                running=False
                exit(0)
                break
            if event.type==pygame.MOUSEBUTTONDOWN:
                paused^=1
        if paused:
            continue
        apfAction=env.agentStates[0].selectAction(algorithm="APF",apfParams=APF_PARAMS)
        # apfAction,action_index=get_nearest_action(np.array(apfAction),action_space)
        
        lidarAngles,lidarDepths=env.agentStates[0].lidarData
        generate_lidar_image(lidarDepths)
        # embedding=get_clip_embeddings()
        action=apfAction
        # action=env.agentStates[0].selectAction(algorithm="OG",policy=policy,embeddings=embedding)
        
        # print('*'*20)
        # print('APF:',apfAction)
        # print('Action:',action)
        # print('*'*20)
        # row=[apfAction[0],apfAction[1],
        #     env.agentStates[0].distanceGoal,
        #     env.agentStates[0].thetaGoal,
        #     ]+env.agentStates[0].lidarData[1]
        # rows.append(row)
        pose=env.agentPoses[0]

        oldPoses,newPoses=env.executeAction(action,apfParams=APF_PARAMS,noise=0)
        env.render(screen)
        pygame.display.update()

        numTimestamps+=1
        pathClearance=min(pathClearance,env.getAgentClearances()[0])
        goalDistance=euclidean((env.agentPoses[0][0],env.agentPoses[0][1]),(env.agentGoals[0][0],env.agentGoals[0][1]))
        sumGoalDistance+=goalDistance
        if len(last_positions) >= 20:
            last_positions.pop(0)  # remove the oldest position

        last_positions.append(pose)
        prompt=env.generate_prompt(oldPoses,newPoses)
        # if len(prompt):
        #     print(j)
        #     good_maps.append(j)
        # if len(last_positions) == 20 and check_convergence(last_positions, 3):
        #     collisionFlag=True
        #     logger.info(f"Converged")
        #     break

        if(numTimestamps>800):
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
        logger.info(f'Stats for Map {MAPS[j]}')
        logger.info(f"Number of timestamps: {numTimestamps}")
        logger.info(f"Path Clearance: {pathClearance}")
        logger.info(f"Average Goal Distance along Path: {sumGoalDistance/numTimestamps}")
        logger.info('*'*20)
        data.append([MAPS[j], numTimestamps, pathClearance, sumGoalDistance/numTimestamps])


# Create the csv file
with open('result.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)