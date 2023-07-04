import pandas as pd
from generate_embedding import get_clip_embeddings
from lidar_converter import generate_lidar_image
df=pd.read_csv('Datasets/combined.csv').drop_duplicates()

new_df=df[['output_linear_velocity','output_angular_velocity','distance_from_goal','angle_from_goal']]
fields=[f"embedding_dim_{i}" for i in range(512)]
dimensionss
for i in range(len(df)):
    depths=[]
    name='lidar_depth_'
    for j in range(1,51):
        nname=name+str(j)
        depths.append(df.iloc[i][nname])
    generate_lidar_image(depths)
    embedding=get_clip_embeddings()
    for k,v in enumerate(embedding):
        new_df.iloc[i][fields[k]]=v
    if i==5:
        break

print(new_df.head())
