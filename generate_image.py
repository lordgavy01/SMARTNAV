import pandas as pd
from lidar_converter import *

file_path='Datasets/train.csv'
df=pd.read_csv(file_path)

for i in range(len(df)):
    angle=df.iloc[i]['angle_from_goal']
    depths=[]
    name='lidar_depth_'
    for j in range(1,51):
        nname=name+str(j)
    depths.append(df.iloc[i][nname])
    if depths[-1]==1000000000.0:
        depths[-1]=0
    generate_lidar_image(depths,f'./Images/train/image{i}.png',base_angle=angle)
