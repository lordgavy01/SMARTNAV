from util import*

def get_lidar_depths(index,agentPoses,agentRadius,obstacles,max_lidar_distance,
                     field_of_view=FIELD_OF_VIEW,number_of_lidar_angles=NUMBER_OF_LIDAR_ANGLES):
  # Assuming agent_theta (with positive x-axis) and field_of_veiew are in radians
  agent_x,agent_y,agent_theta=agentPoses[index]
  lidar_angles=[]  
  angle_spacing=(field_of_view)/(number_of_lidar_angles-1)
  for i in range(number_of_lidar_angles-1):
    theta=field_of_view/2-i*angle_spacing
    lidar_angles.append(normalAngle(theta))
  lidar_angles.append(normalAngle(-field_of_view/2))
  if number_of_lidar_angles==1 :
    lidar_angles=[0]

  lidar_depths=[]
  for lidar_angle in lidar_angles:
    cur_angle=normalAngle(lidar_angle+agent_theta)
    min_distance=INF
    checker_line=((agent_x,agent_y),(agent_x+INF*cos(cur_angle),agent_y+INF*sin(cur_angle)))
    for obstacle in obstacles:
      for i in range(len(obstacle)):
        edge=(obstacle[i],obstacle[(i+1)%len(obstacle)])        
        p=getLinesegmentsIntersection(checker_line,edge)
        if(p==False):
          continue
        min_distance=min(min_distance,euclidean((agent_x,agent_y),p)-agentRadius)
    for i in range(len(agentPoses)):
      if i==index:
        continue
      edge=((agent_x,agent_y),(agentPoses[i][0],agentPoses[i][1]))
      p=getLinesegmentCircleIntersection(checker_line,(agentPoses[i],agentRadius))
      if(p==False):
        continue
      min_distance=min(min_distance,euclidean((agent_x,agent_y),p)-agentRadius)
    min_distance=min(min_distance,max_lidar_distance)
    lidar_depths.append(min_distance)
  return lidar_angles,lidar_depths
