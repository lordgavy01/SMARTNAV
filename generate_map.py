import pygame

pygame.init()
screen=pygame.display.set_mode((800,600))
screen.fill((255,255,255))
pygame.display.update()

running=True
obstacles=[]
cur_obstacle=[]

while running:
    for event in pygame.event.get():
        if event.type==pygame.QUIT:
            running=False
            pygame.quit()
            break
        if event.type==pygame.MOUSEBUTTONDOWN:
            pos=pygame.mouse.get_pos()
            pygame.draw.circle(screen,(0,0,0),pos,2)
            pygame.display.update()            
            cur_obstacle.append(pos)
            print(pos)
        if event.type==pygame.KEYDOWN:
            key_name=pygame.key.name(event.key)
            key_name=key_name.upper()
            print(key_name)  
            if(key_name=="RETURN"):
                pygame.draw.polygon(screen,(0,0,0),cur_obstacle)                
                pygame.display.update()
                obstacles.append(cur_obstacle.copy())
                cur_obstacle=[]
            if(key_name=="S"):
                pygame.image.save(screen,"small_map.png")
                with open("small_map_obstacles.txt","w") as f:
                    for obstacle in obstacles:
                        for point in obstacle:
                            f.write(str(point[0])+" "+str(point[1])+" ")
                        f.write("\n")
                    f.write("\n")

