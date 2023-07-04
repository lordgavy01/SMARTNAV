# SMARTNAV

# Robotic Navigation Simulator
Our Robotic Navigation Simulator is a practical and efficient tool to simulate and evaluate the navigation and control systems of robots in various dynamic and static environments. It can be used to train and test robot navigation models on different tasks, including multi-robot navigation, dynamic obstacle detection, collision avoidance, trajectory planning and much more.

## Features

- **Customizability**: It allows for a high degree of customizability - Users can determine the number of robots and dynamic obstacles, the layout and area of the map, and include static obstacles according to their needs.

- **Iterative Learning**: Facilitates iterative learning with the initial policy. As new data is generated with each passing iteration, it becomes input for the existing policy, thus constantly improving the system.

- **Checkpoints**: It provides functionality to save/load different versions of policies or models, which is helpful in handling different scenarios and improving model's performance. 

- **Loss Curve**: It enables users to save loss curves for the models trained using iterative learning for detailed insights on training or learning effectiveness.

- **Map Generator**: The simulator offers a map generator to facilitate the generation of polygon maps used as static obstacles.

- **Dynamic Environment**: The simulator dynamically links the database to produce lidar signal-to-action mapping using algorithms like APF (Artificial Potential Field).

- **Debugging**: The simulator allows for pausing the simulation for more convenient debugging and issue tracking. 

## Usage
To operate the Simulator, you will be provided with a User Manual detailing the software's operation, along with terms and conditions that must be agreed to in order to proceed with using the software.

## Objectives
Our main objectives for this project include: efficient pedestrian simulation with a dynamic set of obstacles and a robot navigating on metrics like collision avoidance, clearance, and trajectory smoothness, deadlock avoidance and detection, collision avoidance, and the robot's ability to adapt based on textual instructions in real-time.

## Speech Into Textual Prompts
We have implemented a robust Speech-to-Text transformation workflow that leverages the state-of-the-art Wav2Vec2 machine learning model trained on an extensive data set of audio for conversion of raw audio data into human-readable transcriptions.

## Intent Prediction Via Textual Prompts
We have developed a system to interpret commands given in natural language into actionable instruction for velocity and angular velocity adjustment. Our system can understand free-style natural language and adapt the robot's control signals accordingly.

## Instruction-based Control Signal Adaptation
Our system can adapt the robot's actions based on generated prompts under various conditions, like the bot's clearance from nearby obstacles, region's congestion level, proximity to a large obstacle, and a clear line of sight. It can also take user-generated free-style prompts for more complex and realistic scenarios.

## Clone this repository
'''
git clone https://github.com/lordgavy01/SMARTNAV.git
'''

## Download Maps/Checkpoints/Datasets
'''
https://drive.google.com/drive/folders/1fiGEpnTH3LGC4y8w9HcCKNQfwQ71Z2x4?usp=sharing
'''
## Results Sheet and Intent Prediction Datasets
'''
https://docs.google.com/spreadsheets/d/1oueNS_yTCgeKNzwsBZFGF66fDl9W80E7/edit?usp=drive_link&ouid=100155854218838992736&rtpof=true&sd=true
'''

## Install requirements
'''
pip install -r requirements.txt
'''
## Steps to run the simulator
'''
python main.py
'''
