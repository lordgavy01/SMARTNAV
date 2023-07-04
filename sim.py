from util import *
from lidar import *
from environment import *
from agent import *
import time
from lidar_converter import *
from subgoals import get_subgoals
from collections import defaultdict

MAP_NAME="Map-4"

APF_PARAMS=init_params(MAP_NAME)
pygame.init()
pygame.display.set_caption("APF")
obstacles=initMap(mapObstaclesFilename=f"Maps/{MAP_NAME}.txt")
mapBackground=getMapBackground(mapImageFilename=f"Maps/{MAP_NAME}.png")
AGENT_SUBGOALS=get_subgoals(MAP_NAME)
env=Environment()
env.reset(obstacles=obstacles,agentSubGoals=AGENT_SUBGOALS)
GOAL_DISTANCE_THRESHOLD=6

running=True
key=0
screen=pygame.display.set_mode((mapBackground.image.get_width(),mapBackground.image.get_height()))
print("Map Dimensions:",(mapBackground.image.get_width(),mapBackground.image.get_height()))
screen.blit(mapBackground.image, mapBackground.rect)
env.renderSubGoals(screen)
pygame.display.update()

paused=False
collisionFlag=False
numTimestamps=0
pathClearance=INF
sumGoalDistance=0
controlKey=0

v_split=3
w_split=6
V_values = np.linspace(-VMAX, VMAX, v_split)
W_values = np.linspace(-WMAX/2, WMAX/2, w_split)
action_space = np.array(np.meshgrid(V_values, W_values)).T.reshape(-1, 2)
class_mapping={15:0,14:1,12:2,17:3,13:4,16:5,6:6,11:7}
WMAX/2
for i,w in enumerate(W_values):
    print(f'{i}:{w}')
wdict=defaultdict(list)
for i in range(action_space.dim[0]):
    wdict[action_space[k][1]].append(k)
print(wdict)


while running:
    screen.blit(mapBackground.image, mapBackground.rect)
    for events in pygame.event.get():
        if events.type==pygame.QUIT:
            running=False
            break
        if events.type==pygame.MOUSEBUTTONDOWN:
            paused^=1
    if paused:
        continue
    controlKey+=1
    if controlKey%1==0:
        action=env.agentStates[0].selectAction(algorithm="APF",apfParams=APF_PARAMS)
        apfAction,action_index=get_nearest_action(np.array(action),action_space)
        print('Actual:',action,'Matched',apfAction)
        before_progress=env.agentProgress[0]
        env.executeAction(apfAction,apfParams=APF_PARAMS)
        after_progress=env.agentProgress[0]
        env.render(screen)
        pose=env.agentPoses[0]
        goal=env.agentGoals[0]
        thetaGoal=normalAngle(atan2(goal[1]-pose[1],goal[0]-pose[0])-pose[2])
        
        pygame.display.update()
        lidarAngles,lidarDepths=env.agentStates[0].lidarData
        generate_lidar_image(lidarDepths,base_angle=env.agentPoses[0][2])

        goalDistance=euclidean((env.agentPoses[0][0],env.agentPoses[0][1]),(env.agentGoals[0][0],env.agentGoals[0][1]))

        if(env.getAgentClearances()[0]==-1):
                print("Robot Collided!!!")
                break
        if after_progress>before_progress:
            goalDistance=euclidean((env.agentPoses[0][0],env.agentPoses[0][1]),(env.agentSubGoals[0][after_progress][0],env.agentSubGoals[0][after_progress][1]))
            print(goalDistance,goalDistance<GOAL_DISTANCE_THRESHOLD)
            print('Robot reached sub-goal',controlKey)
        
        if goalDistance<GOAL_DISTANCE_THRESHOLD:
            if (env.agentProgress[0]+1)==(len(env.agentSubGoals[0])-1):
                print("Robot reached Goal!")
                running=False
                break
        # prompt=env.generate_prompt()
        # print('Prompt',prompt)
        # print('Embedding',len(get_clip_text_embeddings(prompt)))
            

