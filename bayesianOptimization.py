from hyperopt import hp, fmin, tpe, rand, STATUS_OK, Trials

best_score = float('inf')

def form_search_space(**kwargs):
    #print(kwargs)
    space = {}
    for key, val in kwargs.items():
        #print(key, val)
        space[key] = val
    return space


a(alpha=1, C=3, v=9)