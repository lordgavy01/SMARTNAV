from util import *


def APF(distanceGoal,thetaGoal,lidarData,oldV,apfParams,vmax=VMAX,wmax=WMAX):
    lidarAngles,lidarDepths=lidarData
    
    # Attraction Modelling
    kAttr=apfParams['kAttr']
    distanceThresholdAttraction=apfParams['distanceThresholdAttraction']

    if distanceGoal<=distanceThresholdAttraction:
        fAttr=(kAttr*(distanceGoal),thetaGoal)
    else:
        fAttr=(distanceThresholdAttraction*kAttr,thetaGoal)

    # Repulsion Modelling
    kRep=apfParams['kRep']
    sigma=apfParams['sigma']

    fRep=(0,0)
    for i in range(len(lidarAngles)):
        obsAngle=lidarAngles[i]
        obsDistance=lidarDepths[i]
        curFMagnitude=kRep*exp(-obsDistance/sigma)
        curFTheta=normalAngle(-obsAngle)
        fRep=addForces(fRep,(curFMagnitude,curFTheta))

    # Converting resultant force to (linear velocity,angular velocity) for non-holonomic robot
    fRes=addForces(fAttr,fRep)
    kParam=apfParams['kParam']
    w=normalAngle(kParam*(fRes[1]))
    if abs(w)>wmax:
        w=wmax*(w/abs(w))
    v=min(max(oldV+kParam*fRes[0]*cos(fRes[1]),0),vmax)

    bestAction=(v,w)
    return bestAction