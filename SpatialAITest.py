# Jack Link
# Created for the Brain Dynamics and Control Research Group at Washington University in St. Louis
# Last edited 6/6/22
# This program is an adaptation of the Corsi Block-Tapping Test for visuospatial working memory for artificial systems, designed to be
# adaptable to experimenter requirements. Display can be scaled to accomodate additional blocks, number of blocks can be changed to alter
# difficulty of tasks, initial Corsi span can be changed to alter difficulty, and criteria for success and failure can be changed to alter
# scoring. Alternatively, test can be set to run the same trial continuously and/or reverse the order the sequence must be recalled in.
# Placement of blocks changes between sequences to match human test, where it serves to counter proactive interference.

import random
import tkinter as tk
import time

#CONFIGS
continuous = False #If true, test stays at span and repeats forever; if false, test starts at span
                  #and increases when successCriteria is met, or ends when failureCriteria is met
reverse = False #reverses order blocks must be clicked
scale = 2 #scales the environment, necessary for larger block sizes. General rule is max blocks = 50*scale, but higher blocks can be used at
          #cost of speed
blocks = 9 #number of blocks (traditional block number is 9)
span = 6 #number of blocks that light up (Average human capability is 5-7 blocks)
successCriteria = 3 #number of successes needed to graduate to next level when continuous==False
failureCriteria = 3 #number of successes needed to fail a stage when continuous==False

if span > blocks:
    print("span can not be greater than number of blocks") #blocks can only be selected once, so span cannot exceed number of blocks
    span = blocks
blockSize = 50
position = []
for i in range(0,blocks):
    position.append([None,None])
algorithmInput = []
width = int(500*scale)
height = int(500*scale)
successes = 0
failures = 0
def generate(): #generates new trial
    global algorithmInput
    for i in range(0,blocks): #places blocks randomly, retries placement of block if it overlaps with existing blocks
        placed = False
        while not placed: 
            testPositionX = random.randint(0,width-blockSize)
            testPositionY = random.randint(0,height-blockSize)
            placed = True
            for j in range(0,i):
                if testPositionX > position[j][0] and testPositionX < position[j][0]+blockSize:
                    if testPositionY > position[j][1] and testPositionY < position[j][1]+blockSize:
                        placed = False
                        break
                if testPositionX+blockSize > position[j][0] and testPositionX+blockSize < position[j][0]+blockSize:
                    if testPositionY > position[j][1] and testPositionY < position[j][1]+blockSize:
                        placed = False
                        break
                if testPositionX > position[j][0] and testPositionX < position[j][0]+blockSize:
                    if testPositionY+blockSize > position[j][1] and testPositionY+blockSize < position[j][1]+blockSize:
                        placed = False
                        break
                if testPositionX+blockSize > position[j][0] and testPositionX+blockSize < position[j][0]+blockSize:
                    if testPositionY+blockSize > position[j][1] and testPositionY+blockSize < position[j][1]+blockSize:
                        placed = False
                        break
        position[i][0] = int(testPositionX+(blockSize/2))
        position[i][1] = int(testPositionY+(blockSize/2))
    possibilities = list(range(0,blocks))
    for i in range(0,span): #generates solution
        chosen = random.choice(possibilities)
        possibilities.remove(chosen)
        algorithmInput[i][0] = position[i][0]
        algorithmInput[i][1] = position[i][1]
    if reverse:
        algorithmInput = algorithmInput[::-1]
def algorithm(algorithmInput): #placeholder algorithm
    algorithmOutput = algorithmInput
    return algorithmOutput
def evaluate(algorithmOutput): #evaluates if algorithm output matches sequence
    for i in range(0,span):
        if algorithmOutput[i][0] > algorithmInput[i][0]-(blockSize/2) and algorithmOutput[i][0] < algorithmInput[i][0]+(blockSize/2):
            if algorithmOutput[i][1] > algorithmInput[i][1]-(blockSize/2) and algorithmOutput[i][1] < algorithmInput[i][1]+(blockSize/2):
                continue
        return False
    return True
def test(): #main testing function
    global continuous
    global algorithmInput
    global span
    global successes
    global failures
    algorithmInput = []
    for i in range(0,span):
        algorithmInput.append([None,None])
    generate()
    print(algorithmInput)
    algorithmOutput = algorithm(algorithmInput) #Algorithm is placeholder, currently just returns unaltered input
    correct = evaluate(algorithmOutput)
    if correct:
        successes+=1
        if continuous:
            print("Success! "+str(successes)+"/"+str(successes+failures))
        else:
            print("Success! "+str(successes)+"/"+str(successCriteria+failureCriteria-1)+" (Needs "+str(successCriteria)+"/"+str(successCriteria+failureCriteria-1)+")")
        if not continuous:
            if successes == successCriteria:
                successes = 0
                failures = 0
                if span < blocks:
                    span+=1
                    print("Stage passed, advancing to span of "+str(span))
                else:
                    print("Maximum stage completed, final score is "+str(span))
                    quit()
    else:
        failures+=1
        if continuous:
            print("Incorrect "+str(successes)+"/"+str(successes+failures))
        else:
            print("Incorrect "+str(successes)+"/"+str(successCriteria+failureCriteria-1)+" (Needs "+str(successCriteria)+"/"+str(successCriteria+failureCriteria-1)+")")
        if not continuous:
            if failures == failureCriteria:
                successes = 0
                failures = 0
                print("Stage failed, final score is "+str(span-1))
                quit()
while True:
    test()
    time.sleep(0.1) #for debugging; without algorithm, code runs too fast and crashes