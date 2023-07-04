from math import *

# 1 pixel = 4cm
# 15 timesteps = 1 sec
MAP_IMAGE_FILENAME="small_map.png"
MAP_OBSTACLES_FILENAME="map_obstacles.txt"
APF_DATA_FILENAME="apf_data.csv"
NOISE=0.2
AGENT_RADIUS=10
FIELD_OF_VIEW=radians(180)
NUMBER_OF_LIDAR_ANGLES=50
MAX_LIDAR_DISTANCE=1e9
VMAX=2
WMAX=radians(60)



def prepare_param_dict(kAttr, distanceThresholdAttraction, kRep, sigma, kParam):
    return {
        'kAttr': kAttr,
        'distanceThresholdAttraction': distanceThresholdAttraction,
        'kRep': kRep,
        'sigma': sigma,
        'kParam': kParam,
    }


def init_params(MAP_NAME):
    MAP_ID=int(MAP_NAME[3:])
    if MAP_ID==3:
        return prepare_param_dict(90,1,1e5,2,0.5)
    elif MAP_ID==6:
        return prepare_param_dict(1000,1,2e5,2,0.7)
    elif MAP_ID==7: #TODO: change this
        return prepare_param_dict(125,2,2e5,3,0.7)
    elif MAP_ID==8: #TODO: change this
        return prepare_param_dict(125,2,2e5,2,0.8)
    elif MAP_ID in [9,10]: 
        return prepare_param_dict(125,2,2e5,2,0.7)
   
    return prepare_param_dict(50,1,1e5,2,0.5)
    

