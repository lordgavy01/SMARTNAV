from util import *
from lidar import *
from environment import *
from agent import *
import csv
import time
from lidar_converter import *
from subgoals import get_subgoals

start_time=time.time()

NUM_MAPS=11
MAPS=[f'Map{i}' for i in range(-5,NUM_MAPS+1) if i not in [6,7]]
SUBGOALS=[get_subgoals(maps) for maps in MAPS]


DATAFILENAME='./Datasets/new_train_old_model.csv'
appendFlag=False
obstacles=[]
mapBackgrounds=[]
for maps in MAPS:
    obstacles.append(initMap(mapObstaclesFilename=f"Maps/{maps}.txt"))
    mapBackgrounds.append(getMapBackground(mapImageFilename=f"Maps/{maps}.png"))

env=Environment()
env.reset(obstacles=obstacles[0],agentRadius=AGENT_RADIUS,agentSubGoals=SUBGOALS[0])
pygame.init()
pygame.display.set_caption(f"DAgger")
screen=pygame.display.set_mode((mapBackgrounds[0].image.get_width(),mapBackgrounds[0].image.get_height()))
screen.blit(mapBackgrounds[0].image, mapBackgrounds[0].rect)
env.renderSubGoals(screen)
pygame.display.update()

if not appendFlag:
    fields=["output_linear_velocity","output_angular_velocity","distance_from_goal","angle_from_goal"]
    fields+=['action_label']
    fields+=[f"lidar_depth_{i}" for i in range(50)]
    # fields+=[f"embedding_dim_{i}" for i in range(512)]

    with open(DATAFILENAME,'w') as csvfile: 
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)

# for j in range(0,NUM_MAPS):
#     env=Environment()
#     env.reset(obstacles=obstacles[j],agentRadius=AGENT_RADIUS,agentSubGoals=SUBGOALS[j])
#     pygame.display.set_caption(f"Map {j+1}")
#     screen=pygame.display.set_mode((mapBackgrounds[j].image.get_width(),mapBackgrounds[j].image.get_height()))
#     screen.blit(mapBackgrounds[j].image, mapBackgrounds[j].rect)
#     env.renderSubGoals(screen)
#     pygame.display.update()
#     time.sleep(10)

# exit(0)

v_split=3
w_split=6
V_values = np.linspace(-VMAX, VMAX, v_split)
W_values = np.linspace(-WMAX/2, WMAX/2, w_split)
action_space = np.array(np.meshgrid(V_values, W_values)).T.reshape(-1, 2)

keepIterating=True
imageIndex=0

rows=[]
for j in range(len(MAPS)):
    print(f"\n*Map {j}*")
    env=Environment()
    env.reset(obstacles=obstacles[j],agentSubGoals=SUBGOALS[j])
    
    screen=pygame.display.set_mode((mapBackgrounds[j].image.get_width(),mapBackgrounds[j].image.get_height()))
    screen.blit(mapBackgrounds[j].image, mapBackgrounds[j].rect)
    running=True
    collisionFlag=False
    numTimestamps=0
    pathClearance=INF
    sumGoalDistance=0
    APF_PARAMS=init_params(MAPS[j])
    while running:
        screen.blit(mapBackgrounds[j].image, mapBackgrounds[j].rect)
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                running=False
                break

        apfAction=env.agentStates[0].selectAction(algorithm="APF",apfParams=APF_PARAMS)
        apfAction,action_index=get_nearest_action(np.array(apfAction),action_space)
        lidarAngles,lidarDepths=env.agentStates[0].lidarData
        # generate_lidar_image(lidarDepths)
        # embedding=get_clip_embeddings()
        row=[apfAction[0],apfAction[1],
            env.agentStates[0].distanceGoal,
            env.agentStates[0].thetaGoal,action_index
            ]+lidarDepths
        rows.append(row)
        imageIndex+=1
        env.executeAction(apfAction,apfParams=APF_PARAMS)
        env.render(screen)
        pygame.display.update()

        numTimestamps+=1
        pathClearance=min(pathClearance,env.getAgentClearances()[0])
        goalDistance=euclidean((env.agentPoses[0][0],env.agentPoses[0][1]),(env.agentGoals[0][0],env.agentGoals[0][1]))
        
        sumGoalDistance+=goalDistance

        if(env.getAgentClearances()[0]==-1):
            print(env.getAgentClearances())
            print("Robot Collided!!!")
            collisionFlag=True
            break
        
        if goalDistance<12:
            if (env.agentProgress[0]+1)==(len(env.agentSubGoals[0])-1):
                print("Robot reached Goal!")
                running=False
                break

    if not collisionFlag:
        print(f'Stats for Map {j}')
        print(f"Number of timestamps: {numTimestamps}")
        print(f"Path Clearance: {pathClearance}")
        print(f"Average Goal Distance along Path: {sumGoalDistance/numTimestamps}")
        print('*'*20)

with open(DATAFILENAME,'a') as csvfile: 
    csvwriter = csv.writer(csvfile)
    csvwriter.writerows(rows)

