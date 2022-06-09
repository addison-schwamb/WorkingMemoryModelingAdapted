from train_input import *
#from itertools import permutations
import itertools
import pickle

# variables
file_name = 'network_performance'
g = [1.3, 1.4, 1.5]
pg = [0.4, 0.6, 0.7, 0.9]
fb = [40, 30, 20, 10, 5, 1]
s =  [0, 1, 2, 3]

combs = [[x,y] for x in g for y in pg]
temp = [[x,[y]] for x in combs for y in fb]
combs = []

for i in range(len(temp)):
    combs.append(list(itertools.chain(*temp[i])))
    
temp = [[x,[y]] for x in combs for y in s]
combs = []
for i in range(len(temp)):
    combs.append(list(itertools.chain(*temp[i])))

attractors = dict()
attractors['pre_damage'] = []
attractors['post_damage'] = []
pct_correct = dict()
pct_correct['pre_damage'] = []
pct_correct['post_damage'] = []
for i in range(len(combs)):
    d = "{\"g\":" + str(combs[i][0]) + ", \"pg\":" + str(combs[i][1]) + ",\"fb_var\":" + str(combs[i][2]) + ", \"input_var\": 50.0, \"n_train\": 500, \"encoding\": [0.5, 1.0, 1.5], \"seed\":" + str(combs[i][3]) + ", \"init_dist\": \"Gauss\", \"input\": 0, \"pct_rmv\": 0.01}"
    pre_dmg_att, post_dmg_att, pre_pct_correct, post_pct_correct = main(d)
    attractors['pre_damage'].append(pre_dmg_att)
    attractors['post_damage'].append(post_dmg_att)
    pct_correct['pre_damage'].append(pre_pct_correct)
    pct_correct['post_damage'].append(post_pct_correct)
    
networks = dict()
networks['attractors'] = attractors
networks['pct_correct'] = pct_correct

with open(file_name, 'wb') as f:
    pickle.dump(networks, f, protocol=-1)

print('done')