from util import *
from agent import *
from lidar import *
from colors import *
from config import *

class Environment:

    # obstacles --> List of polygons. polygon --> [centre,vertices]
    # agentPoses, agentGoals --> List of (x,y,theta). theta is in radians 
    # agentStates --> List of objects of class AgentState.
    # agent 0 is robot, others are humans 
    
    def reset(self,obstacles,agentSubGoals,agentRadius=AGENT_RADIUS):
        self.obstacles=obstacles
        self.agentStates=[]
        self.agentSubGoals=agentSubGoals
        self.agentRadius=agentRadius
        self.agentProgress=[0]*len(agentSubGoals)
        self.agentPoses=[]
        self.agentGoals=[]
        for agent in agentSubGoals:
            self.agentPoses.append(agent[0])
            self.agentGoals.append(agent[1])
        self.updateAgentStates()
    
    def render(self,screen,robotColor=(255,0,0)):
        agentColors=[Colors.red,Colors.blue,Colors.cyan,Colors.yellow,Colors.green]
        for i in range(len(self.agentPoses)):
            agentCoordinates=(int(self.agentPoses[i][0]),int(self.agentPoses[i][1]))
            goalCoordinates=(int(self.agentGoals[i][0]),int(self.agentGoals[i][1]))
            pygame.draw.circle(screen,agentColors[i],agentCoordinates,self.agentRadius)
            pygame.draw.circle(screen,agentColors[i],goalCoordinates,self.agentRadius,2)            
        rayColors=[Colors.green,Colors.blue]
        lidarAngles,lidarDepths=self.agentStates[0].lidarData
        for i in range(len(lidarAngles)):
            curAngle=normalAngle(self.agentPoses[0][2]+lidarAngles[i])
            robotCoordinates=(self.agentPoses[0][0],self.agentPoses[0][1])
            lidarHitpoint=(robotCoordinates[0]+(lidarDepths[i]+self.agentRadius)*cos(curAngle),
                           robotCoordinates[1]+(lidarDepths[i]+self.agentRadius)*sin(curAngle))
            if lidarDepths[i]>=1e9:
                pygame.draw.line(screen,rayColors[0],robotCoordinates,lidarHitpoint)
            else: 
                pygame.draw.line(screen,rayColors[1],robotCoordinates,lidarHitpoint)    

    def renderSubGoals(self,screen):
        agentColors=[Colors.red,Colors.blue,Colors.cyan,Colors.yellow,Colors.green]
        for i in range(len(self.agentSubGoals)):
            for j in range(len(self.agentSubGoals[i])):
                subGoalCoordinate=(int(self.agentSubGoals[i][j][0]),int(self.agentSubGoals[i][j][1]))
                if(j==0):
                    pygame.draw.circle(screen,agentColors[i],subGoalCoordinate,self.agentRadius)
                else:
                    pygame.draw.circle(screen,agentColors[i],subGoalCoordinate,self.agentRadius,2)              

    def updateAgentStates(self,agentVelocities=[]):
        if(len(agentVelocities)==0):
            agentVelocities=[0 for i in range(len(self.agentSubGoals))]
        for i in range(len(self.agentPoses)):
            lidarData=get_lidar_depths(i,self.agentPoses,self.agentRadius,self.obstacles,max_lidar_distance=MAX_LIDAR_DISTANCE,
                                       field_of_view=FIELD_OF_VIEW,number_of_lidar_angles=NUMBER_OF_LIDAR_ANGLES)
            pose=self.agentPoses[i]
            goal=self.agentGoals[i]
            distanceGoal=euclidean((pose[0],pose[1]),(goal[0],goal[1]))
            thetaGoal=normalAngle(atan2(goal[1]-pose[1],goal[0]-pose[0])-pose[2])
            if not len(self.agentStates)==len(self.agentSubGoals):
                self.agentStates.append(AgentState(distanceGoal,thetaGoal,lidarData,agentVelocities[i],pose))
            else:
                self.agentStates[i].update(distanceGoal,thetaGoal,lidarData,agentVelocities[i],pose)
    
    def executeAction(self,robotAction,apfParams,noise=NOISE,goalDistanceThreshold=12):
        oldEnvironmentState=(self.obstacles,self.agentPoses,self.agentGoals,self.agentStates)
        oldPoses=self.agentPoses.copy()
        agentVelocities=[]
        noiseAction=robotAction
        for i in range(len(self.agentPoses)):
            action=robotAction
            if not i==0:
                action=self.agentStates[i].selectAction(algorithm="APF",apfParams=apfParams)

            v=action[0]
            w=action[1]
            if noise>0.000001:
                rnoise=random.uniform(-noise,noise)
                v=min(max(v*(1+rnoise),0),VMAX)
                w=normalAngle(w*(1+rnoise))
                
                if i==0:
                    noiseAction=(v,w)
            agentVelocities.append(v)
            self.agentPoses[i]=kinematic_equation(self.agentPoses[i],v,w,dT=1) 
            if euclidean((self.agentPoses[i][0],self.agentPoses[i][1]),(self.agentGoals[i][0],self.agentGoals[i][1]))<goalDistanceThreshold:
                if not (self.agentProgress[i]+1)==(len(self.agentSubGoals[i])-1):
                    self.agentProgress[i]+=1
                    self.agentGoals[i]=self.agentSubGoals[i][self.agentProgress[i]+1]
        self.updateAgentStates(agentVelocities)        
        newEnvironmentState=(self.obstacles,self.agentPoses,self.agentGoals,self.agentStates)
        newPoses=self.agentPoses.copy()
        return oldPoses,newPoses

    def rewardFunction(oldEnvironmentState,robotAction,newEnvironmentState):
        return 0
    def check_safe(self,pose,radius):
        agentClearance=INF
        center=(pose[0],pose[1])
        for obstacle in self.obstacles:
            for i in range(len(obstacle)):
                edge=(obstacle[i],obstacle[(i+1)%len(obstacle)])
                d=getDistancePointLineSegment(center,edge)
                if(d-radius<=0): 
                    agentClearance=-1
                else: 
                    agentClearance=min(agentClearance,d-radius)
        return (agentClearance!=-1 and agentClearance>=0.2)

    # NOTE: Would fail to detect collision if agent is completely inside obstacle
    def getAgentClearances(self):
        agentClearances=[]
        for agentId in range(len(self.agentSubGoals)):
            agentClearance=INF
            center=(self.agentPoses[agentId][0],self.agentPoses[agentId][1])
            radius=self.agentRadius
            for obstacle in self.obstacles:
                for i in range(len(obstacle)):
                    edge=(obstacle[i],obstacle[(i+1)%len(obstacle)])
                    d=getDistancePointLineSegment(center,edge)
                    if(d-radius<=0): 
                        agentClearance=-1
                    else: 
                        agentClearance=min(agentClearance,d-radius)
            for j in range(len(self.agentSubGoals)):
                if (agentId==j): continue
                center2=(self.agentPoses[j][0],self.agentPoses[j][1])
                radius2=self.agentRadius
                d=euclidean(center,center2)
                if(d-radius-radius2<=0): 
                    agentClearance=-1
                else: 
                    agentClearance=min(agentClearance,d-radius-radius2)
            agentClearances.append(agentClearance)
        return agentClearances
    

    def getCongestion(self):
        congestion=0
        minD=INF
        sigma=10
        center=(self.agentPoses[0][0],self.agentPoses[0][1])
        for j in range(1,len(self.agentSubGoals)):
                center2=(self.agentPoses[j][0],self.agentPoses[j][1])
                radius2=self.agentRadius
                d=euclidean(center,center2)
                minD=min(minD,d)
                congestion+=exp(-d/(sigma*sigma))
        congestion/=max(1,len(self.agentSubGoals)-1)
        return congestion,minD
   
    def check_for_race_condition(self,agent1_pose, agent1_next, agent2_pose, agent2_next, angle_threshold=FIELD_OF_VIEW):
        # Extract agent pose values (x, y, theta)
        agent1_pos, agent1_theta = agent1_pose[:2], radians(agent1_pose[2])
        agent1_next_pos = agent1_next[:2]
        agent2_pos, agent2_theta = agent2_pose[:2], radians(agent2_pose[2])
        agent2_next_pos = agent2_next[:2]

        # Calculate the distance and direction from the current agent's position to their next position
        d1 = ((agent1_next_pos[0] - agent1_pos[0])**2 + (agent1_next_pos[1] - agent1_pos[1])**2)**0.5
        d2 = ((agent2_next_pos[0] - agent2_pos[0])**2 + (agent2_next_pos[1] - agent2_pos[1])**2)**0.5

        # Calculate the virtual intersection points following each agent's orientation and distance
        virtual_intersection1 = (agent1_pos[0] + d1 * cos(agent1_theta + radians(angle_threshold)),
                                agent1_pos[1] + d1 * sin(agent1_theta + radians(angle_threshold)))

        virtual_intersection2 = (agent1_pos[0] + d1 * cos(agent1_theta - radians(angle_threshold)),
                                agent1_pos[1] + d1 * sin(agent1_theta - radians(angle_threshold)))

        # Calculate the viewing angle range for the second agent, considering their angle threshold
        max_angle2 = agent2_theta + radians(angle_threshold)
        min_angle2 = agent2_theta - radians(angle_threshold)

        def angle_is_between(a, min_a, max_a):
            return min_a <= a <= max_a or min_a <= a + 2*np.pi <= max_a or min_a <= a - 2*np.pi <= max_a

        # Check if either of the virtual intersection points falls within the second agent's viewing angle range
        for intersection in [virtual_intersection1, virtual_intersection2]:
            angle = atan2(intersection[1] - agent2_pos[1], intersection[0] - agent2_pos[0])
            if angle_is_between(angle, min_angle2, max_angle2):
                return True

        return False


    def check_race_condition(self,oldPoses,newPoses,goalDistanceThreshold=12):
        center=(self.agentPoses[0][0],self.agentPoses[0][1])
       
        for j in range(1,len(newPoses)):
            center2=(self.agentPoses[j][0],self.agentPoses[j][1])
            d=euclidean(center,center2)
            if d>100:
                continue

            if self.agentProgress[j]==len(self.agentSubGoals[j])-2 and euclidean((self.agentPoses[j][0],self.agentPoses[j][1]),(self.agentGoals[j][0],self.agentGoals[j][1]))<goalDistanceThreshold:
                continue
            race_flag=self.check_for_race_condition(oldPoses[0],newPoses[0],oldPoses[j],newPoses[j])
            if race_flag:
                return race_flag
        return False

    def generate_prompt(self, oldPoses=None, newPoses=None, fov_bonus=False):
        prompt = ""
        agent_clearance=self.getAgentClearances()[0]
        v_multiplier, w_multiplier = 1, 1  # initialize
        
        clearance_threshold=0.2
        if agent_clearance < clearance_threshold:
            prompt += "Maintain a safe distance from nearby obstacles to avoid collision. "
            v_multiplier *= 0.5  # Reduce velocity
            w_multiplier *= 2    # Increase angular velocity

        congestion_level,nearest_agent_distance=self.getCongestion()
        congestion_threshold=0.4
        if congestion_level > congestion_threshold:
            prompt += "Caution: The area ahead is highly congested. Consider slowing down to navigate safely. "
            v_multiplier *= 0.7  # Slow down in high congestion
            w_multiplier *= 1.5  # Be ready to make more adjustments 

        proximity_alert_threshold=50
        if nearest_agent_distance < proximity_alert_threshold:
            prompt += "Watch out! There's a large obstacle approaching. Make sure to maintain safe distance. "
            v_multiplier *= 0.5  # Major reduction in velocity
            w_multiplier *= 2    # Increase angular velocity for evasive maneuvers

        if fov_bonus:
            prompt += "Good news! The destination is within your line of sight. Proceed towards it, ensuring to avoid any collisions. "
            v_multiplier *= 1.2  # Increase velocity
            w_multiplier *= 0.8  # Reduce angular velocity as we are heading straight to goal

        race_flag=self.check_race_condition(oldPoses,newPoses)
        if race_flag:
            prompt += "Alert: You are in a potential deadlock situation. It's recommended to wait and monitor the other agent's actions to avoid collision. "
            v_multiplier *= 0.3  # Major reduction in velocity
            w_multiplier *= 1.5  # Be ready to make more adjustments 
            
        if prompt=="":
            prompt="All clear. Proceed towards the destination."

        return prompt, v_multiplier, w_multiplier