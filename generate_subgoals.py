from util import *
from colors import *

pygame.init()

# mapBackground=getMapBackground(mapImageFilename="./Maps/Map9.png")
mapBackground=getMapBackground(mapImageFilename="small_map.png")
screen=pygame.display.set_mode((mapBackground.image.get_width(),mapBackground.image.get_height()))
screen.blit(mapBackground.image, mapBackground.rect)
pygame.display.update()

running=True
allAgentSubGoals=[]
curAgentSubGoal=[]
agentColors=[Colors.red,Colors.blue,Colors.cyan,Colors.yellow,Colors.green]

while running:
    for event in pygame.event.get():
        if event.type==pygame.QUIT:
            running=False
            pygame.quit()
            break
        if event.type==pygame.MOUSEBUTTONDOWN:
            pos=pygame.mouse.get_pos()
            pygame.draw.circle(screen,agentColors[len(allAgentSubGoals)],pos,AGENT_RADIUS,2)
            pygame.display.update()
            curAgentSubGoal.append((pos[0],pos[1],0))
            print(pos)
        if event.type==pygame.KEYDOWN:
            key_name=pygame.key.name(event.key)
            key_name=key_name.upper()
            print(key_name)  
            if(key_name=="RETURN"):     
                allAgentSubGoals.append(curAgentSubGoal.copy())
                curAgentSubGoal=[]
            if(key_name=="S"):
                print(allAgentSubGoals)