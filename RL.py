from Models.model_RL import *
from util import *
from lidar import *
from environment import *
from agent import *
import csv
import time
from lidar_converter import *
from subgoals import get_subgoals
import torch.optim as optim
from torch.distributions import Categorical
from tqdm import tqdm
from transformers import AdamW,get_linear_schedule_with_warmup

np.random.seed(0)

def compute_returns(next_value, rewards, masks, gamma=0.96):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns

def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage):
    batch_size = states.size(0)
    for _ in range(int(ceil(batch_size /mini_batch_size))):
        rand_ids = np.random.randint(0, batch_size, mini_batch_size)
        yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]


def ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, clip_param=0.23,printPPOLoss=False):
    for i in range(ppo_epochs):
        runningLoss=0.0
        count_steps=0
        for state, action, old_log_probs, return_, advantage in ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages):
            input1, input2 = torch.split(state, [512, 2], dim=-1)
            new_log_probs, value = model((input1.squeeze(1), input2.squeeze(1)))
            dist = Categorical(new_log_probs)
            entropy = -torch.sum(dist.probs * torch.log(dist.probs + 1e-8), dim=-1).mean()
            ratio = torch.exp(new_log_probs - old_log_probs.detach())
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = (return_ - value).pow(2).mean()
            entropy_coefficient = 0.1
            loss = 0.5 * critic_loss + actor_loss - entropy_coefficient * entropy
            runningLoss+=loss
            count_steps+=1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        runningLoss/=count_steps
    runningLoss/=ppo_epochs
    print(f'Loss after PPO Update: {runningLoss}')
def HF_function(state, action):
    return 0

def get_reward(state, action):
    reward = HF_function(state, action)
    return reward
def get_agent_state(env):
    _,lidarDepths=env.agentStates[0].lidarData
    generate_lidar_image(lidarDepths)
    embedding=get_clip_embeddings()
    input1=torch.tensor([embedding]).float()
    disAndAngle=[env.agentStates[0].distanceGoal,env.agentStates[0].thetaGoal]
    input2=torch.tensor([disAndAngle]).float()
    return (input1,input2)
    
def env_step(env,action,action_space,APF_PARAMS,GOAL_DISTANCE_THRESHOLD):
    v_w=action_space[action]

    oldgoalDistance=euclidean((env.agentPoses[0][0],env.agentPoses[0][1]),(env.agentGoals[0][0],env.agentGoals[0][1]))
    before_progress=env.agentProgress[0]
    env.executeAction(v_w,APF_PARAMS,GOAL_DISTANCE_THRESHOLD)
    after_progress=env.agentProgress[0]
    goalDistance=euclidean((env.agentPoses[0][0],env.agentPoses[0][1]),(env.agentGoals[0][0],env.agentGoals[0][1]))
    
    done=False
    clearance=env.getAgentClearances()[0]
    next_state_tensor=get_agent_state(env)
    time_penalty=-1.5
    ang_vel_threshold=radians(30)
    
    reward_dict={}
    reward_dict['Penalty Low Clearance']=(0,0)
    reward_dict['Penalty Rotation']=(0,0)
    reward= -0.1*goalDistance + time_penalty
    pose=env.agentPoses[0]
    goal=env.agentGoals[0]
    thetaGoal=normalAngle(atan2(goal[1]-pose[1],goal[0]-pose[0])-pose[2])
    field_of_view_bonus = 2*np.cos(thetaGoal)
    reward += field_of_view_bonus
    
    ang_vel=v_w[1]
    if abs(ang_vel) > ang_vel_threshold:
        reward_dict['Penalty Rotation']=(-5 * (abs(ang_vel) - ang_vel_threshold),-5 * (abs(ang_vel) - ang_vel_threshold)/reward)
        reward += -5 * (abs(ang_vel) - ang_vel_threshold)
    
       
    

    if clearance < 0.2 and clearance!=-1:
        reward+= -25*clearance
        reward_dict['Penalty Low Clearance']=(-25*clearance,-25*clearance/reward)
    reward_dict['Penalty Distance to Goal']=(-0.1*goalDistance,-0.1*goalDistance/reward)
    reward_dict['Penalty Time']=(time_penalty,time_penalty/reward)
    reward_dict['Bonus Goal FOV']=(field_of_view_bonus,field_of_view_bonus/reward)


    if clearance==-1:
        reward= -100
        done=True
        reward_dict={'Penalty Collision':(reward,1.0)}
    

    if after_progress>before_progress: # reached sub-goal
        print('Robot Reached Sub-Goal')
        goalDistance=euclidean((env.agentPoses[0][0],env.agentPoses[0][1]),(env.agentSubGoals[0][after_progress][0],env.agentSubGoals[0][after_progress][1]))
        reward= 1000 - goalDistance + time_penalty
        reward_dict={'Bonus Sub-Goal':(reward,1.0)}

    if goalDistance<GOAL_DISTANCE_THRESHOLD: #reached final goal
        if (env.agentProgress[0]+1)==(len(env.agentSubGoals[0])-1):
            reward = 500 - goalDistance + time_penalty
            done=True
            reward_dict={'Bonus Goal':(reward,1.0)}
    
    new_reward_dict = {key: (value[0], abs(value[1]) * 100) for key, value in reward_dict.items()}
    # print('*'*20)
    # print(new_reward_dict)
    # print('*'*20)
    return next_state_tensor,reward,done

MAP_NAME="Map-4"
APF_PARAMS=init_params(MAP_NAME)
pygame.init()
pygame.display.set_caption("APF")
obstacles=initMap(mapObstaclesFilename=f"Maps/{MAP_NAME}.txt")
mapBackground=getMapBackground(mapImageFilename=f"Maps/{MAP_NAME}.png")
AGENT_SUBGOALS=get_subgoals(MAP_NAME)
GOAL_DISTANCE_THRESHOLD=6

v_split=2
w_split=4
V_values = np.linspace(0, VMAX, v_split)
W_values = np.linspace(-WMAX/2, WMAX/2, w_split)
action_space = np.array(np.meshgrid(V_values, W_values)).T.reshape(-1, 2)
LearningRate=1e-4
shared_model=RLModel(512)
model = ActorCritic(shared_model,action_dim=v_split*w_split,shared_dim=256)

num_steps = 128
mini_batch_size = 128
ppo_epochs = 5
max_episodes = 1000
param_optimizer = list(model.named_parameters())
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
optimizer = AdamW(optimizer_grouped_parameters, lr=LearningRate ,betas=(0.9, 0.97), eps=1e-07)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=5, num_training_steps=max_episodes)


episode_rewards=[]


for episode in tqdm(range(max_episodes), desc="Training Episodes"):
    screen=pygame.display.set_mode((mapBackground.image.get_width(),mapBackground.image.get_height()))
    env=Environment()
    env.reset(obstacles=obstacles,agentSubGoals=AGENT_SUBGOALS)
    env.renderSubGoals(screen)
    pygame.display.update()
    state_tensor=get_agent_state(env)
    log_probs = []
    values = []
    states = []
    actions = []
    rewards = []
    masks = []

    steps = 0

    running = True
    paused = False
    episode_reward = 0
    while running and steps < num_steps:
        screen.blit(mapBackground.image, mapBackground.rect)
        for events in pygame.event.get():
            if events.type == pygame.QUIT:
                running = False
                exit(0)
                break
            if events.type == pygame.MOUSEBUTTONDOWN:
                paused ^= 1

        if not paused:
            steps += 1

        action_probs, state_value = model(state_tensor)

        dist = Categorical(action_probs)
        action = dist.sample()
        
        next_state, reward, done = env_step(env,action.item(),action_space,APF_PARAMS,GOAL_DISTANCE_THRESHOLD)
        episode_reward += reward
        rewards.append(reward)
        masks.append(1 - done)

        states.append(torch.cat(state_tensor, dim=-1))
        actions.append(action)
        values.append(state_value)
        log_probs.append(dist.log_prob(action))

        state = next_state
        
        env.render(screen)
        pygame.display.update()

        if done:
            break

    _, next_value = model(next_state)
    returns = compute_returns(next_value, rewards, masks)

    states_tensor = torch.stack(states)
    actions_tensor = torch.stack(actions)
    log_probs_tensor = torch.stack(log_probs)
    returns_tensor = torch.cat(returns).detach()
    values_tensor = torch.cat(values).detach()
    advantages = returns_tensor - values_tensor

    ppo_update(ppo_epochs, mini_batch_size, states_tensor, actions_tensor, log_probs_tensor, returns_tensor, advantages)
    episode_rewards.append(episode_reward)
    if episode%100==0:
        torch.save(model.state_dict(), f'./RL/model_ep_{episode}.pth')
    
    avg_reward = np.mean(episode_rewards[-1])
    print(f"Episode: {episode} Reward: {avg_reward}")
