import numpy as np
import pandas as pd

def dependent_mav(bandits, pulls, structure, manualProbs, problist, contrast): 
    n_bandits = bandits
    n_pulls = pulls
    epsilon = 0.1
    k = 2 #number of arms
    c = structure #degree of structure 
    t = 0.1 #temperature
    a = 0.1 #learning rate
    if manualProbs == False:
        prob_array = [0.1, 0.2,  0.3, 0.4, 0.6, 0.7, 0.8, 0.9]
    else:
        prob_array = problist
    
    r_pulls_aray = []
    #Q is carried over
    for i in range(n_bandits):   
        Q = np.ones(2)/2
        if manualProbs == False:
            p1 = np.random.choice(prob_array)
            p2 = 1-p1
            prob = [p1, p2]
        if manualProbs == True:
            p1 = np.random.choice(prob_array)
            p2 = p1 + contrast
            prob = [p1, p2]
        # regret_pull = []
        # cum_regret = []    
        r_pulls = []
        #print('prob:' , prob) 
        for pull in range(n_pulls):
            
            exp_Q = np.exp(Q/t) 
            sfx_Q = exp_Q/np.sum(exp_Q) # softmax probabilities
            j= np.random.choice(range(k),p=sfx_Q) # picks one arm based on softmax probability
            if np.random.random()<prob[j]: r = 1
            else: r = 0
            
            r_pulls.append(r)
            
            Q[j] = Q[j] + ((r - Q[j])*a)
            # if j == 0:
            #     Q[j+1] = Q[j+1] + c*((1-r) - Q[j+1]*a)
            # if j == 1:
            #     Q[j-1] = Q[j-1] + c*((1-r) - Q[j-1]*a)    
            if j == 0:
                Q[j+1] = Q[j+1] - c*(r - Q[j+1]*a)
            if j == 1:
                Q[j-1] = Q[j-1] - c*(r - Q[j-1]*a) 
        #     print('sfx:', sfx_Q)
        #     print('arm:' , j)
        #     print('reward' , r)
        #     print('Q:', Q[0], Q[1])
        #     print('-------------')
        # print('--------------------')
            
        r_pulls_aray.append(r_pulls)
        
        # cum_regret_array_i.append(cum_regret)
    r_pulls_average = np.mean(r_pulls_aray, axis = 0)

        
    # cum_regret_array_av = np.mean(cum_regret_array_i, axis = 0)
    mav = pd.DataFrame(r_pulls_average).rolling(1, center = True).mean()
    #fig1.plot(range(0, n_pulls), cum_regret_array_av) 
    #fig1.plot(range(0, n_pulls), mav )
    #fig1.plot(range(0, n_pulls), j_array, 'o')
    # print(prob)
    # print(Q)
    return mav   
    
def independent_mav(bandits, pulls, structure, manualProbs, problist, contrast): 
    n_bandits = bandits
    n_pulls = pulls
    epsilon = 0.1
    k = 2 #number of arms
    c = structure #degree of structure 
    t = 0.1 #temperature
    a = 0.1 #learning rate
    if manualProbs == False:
        prob_array = [0.1, 0.2,  0.3, 0.4, 0.6, 0.7, 0.8, 0.9]
    else:
        prob_array = problist    
    r_pulls_aray = []
    #Q is carried over
    for i in range(n_bandits):   
        Q = np.ones(2)/2
        if manualProbs == False:
            p1 = np.random.choice(prob_array)
            p2 = 1-p1
            prob = [p1, p2]
        if manualProbs == True:
            p1 = np.random.choice(prob_array)
            p2 = p1 + contrast
            prob = [p1, p2]
        # regret_pull = []
        # cum_regret = []    
        r_pulls = []
        #print('prob:' , prob) 
        for pull in range(n_pulls):
            
            exp_Q = np.exp(Q/t) 
            sfx_Q = exp_Q/np.sum(exp_Q) # softmax probabilities
            j= np.random.choice(range(k),p=sfx_Q) # picks one arm based on softmax probability
            if np.random.random()<prob[j]: r = 1
            else: r = 0
            
            r_pulls.append(r)
            
            Q[j] = Q[j] + ((r - Q[j])*a)
            if j == 0:
                Q[j+1] = Q[j+1] - c*((r - Q[j+1])*a)
            if j == 1:
                Q[j-1] = Q[j-1] - c*((r - Q[j-1])*a)    
        #     print('probs: ', prob)
        #     print('sfx:', sfx_Q)
        #     print('arm:' , j)
        #     print('reward' , r)
        #     print('Q:', Q[0], Q[1])
        #     print('-------------')
        # print('------------------------------------')    
        r_pulls_aray.append(r_pulls)
        
        # cum_regret_array_i.append(cum_regret)
    r_pulls_average = np.mean(r_pulls_aray, axis = 0)

        #print('--------------------')
    # cum_regret_array_av = np.mean(cum_regret_array_i, axis = 0)
    mav = pd.DataFrame(r_pulls_average).rolling(1
                                                , center = True).mean()
    #fig1.plot(range(0, n_pulls), cum_regret_array_av) 
    #fig1.plot(range(0, n_pulls), mav )
    #fig1.plot(range(0, n_pulls), j_array, 'o')
    # print(prob)
    # print(Q)
    return mav   
    
# One function per environment
# Inputs : n_trials, n_sessions, coupling factor, output (reward rate/regret)

def structured(trials, sessions, probs, coupling, policy, params, window, output, plot):
    ''' 
    coupling parameter defines degree of structure
    store all choices and rewards in separate arrays
    params in order: alpha, epsilon, tau
    policy: softmax or epsilon-greedy
    '''
    arms = 2
    alpha = params[0]
    epsilon = params[1]
    tau = params[2]
    c = coupling #between zero  and one
    if probs == 'all':
        probs = np.delete(np.arange(0.1, 1, 0.1), 4)
    

    else:
        probs = probs  
    choices = [] #all choices across all sessions
    rewards = [] #all rewards across all sessions
    rewards_array = [] #all session-wise reward lists
    regret_array = [] #all session-wise regret lists
    p_left = []
    for i in range(sessions):
        Q = np.ones(2)/2
        p1 = np.random.choice(probs)
        p2 = 1-p1
        prob = [p1, p2]
        p_left.append(p2)
        # print(prob)
        best = np.maximum(prob[0], prob[1]) #optimal arm for this session
        r_session = [] # records rewards for each session, then resets
        regret_session = [] # records regret for each session, then resets
        for trial in range(trials):
            if policy == 'softmax':
                exp_Q = np.exp(Q/tau) 
                sfx_Q = exp_Q/np.sum(exp_Q) # softmax probabilities
                j= np.random.choice(range(arms),p=sfx_Q) # picks one arm based on softmax probability
                if np.random.random()<prob[j]: r = 1
                else: r = 0  
                # print(j,r)
            if policy == 'e-greedy':
                if np.random.random()<epsilon:
                    j = np.random.choice(range(arms))
                else:
                    j = np.argmax(Q)
                if np.random.random()<prob[j]: r = 1
                else: r = 0 
                # print(j, r)
            Q[j] = Q[j] + ((r - Q[j])*alpha)
            if j == 0:
                Q[1] = Q[1] - c*((r - Q[j])*alpha)
            if j == 1:
                Q[0] = Q[0] - c*((r - Q[j])*alpha)                
            r_session.append(r) 
            rewards.append(r)
            choices.append(j) 
            #computing regret:
            if output == 'soft regret':
                regret = abs(np.subtract(best, prob[j]))
            if output == 'hard regret':
                if np.max(prob) == prob[j]:
                    regret = 0
                else:
                    regret = 1
            if (output == 'soft regret') | (output == 'hard regret'):
                regret_session.append(regret)
        rewards_array.append(pd.DataFrame(r_session).rolling(window).mean())
        if (output == 'soft regret') | (output == 'hard regret'):
            regret_array.append(pd.DataFrame(regret_session).rolling(window).mean())

        
    #plotting:
    if plot == True:
        if output == 'reward rate':
            rewards_average = np.mean(rewards_array, axis = 0).tolist()
            plt.plot((np.arange(0, len(rewards_average))), rewards_average)
            sns.despine()
            plt.show()
        if (output == 'soft regret') | (output == 'hard regret'):
            regret_average = np.mean(regret_array, axis = 0).tolist()
            plt.plot((np.arange(0, len(regret_average))), regret_average)
            sns.despine()
            plt.show()
        if output == 'none':
            print()
    else:
        if output == 'reward rate':
            rewards_average = np.mean(rewards_array, axis = 0).tolist()            
            return rewards_average
        if (output == 'soft regret') | (output == 'hard regret'):
            regret_average = np.mean(regret_array, axis = 0).tolist()
            return regret_average
    return choices, p_left, rewards
    
        
def unstructured(trials, sessions, coupling, policy, params, window, output, plot):
    ''' 
    coupling parameter defines degree of structure
    store all choices and rewards in separate arrays
    params in order: alpha, epsilon, tau
    policy: softmax or epsilon-greedy
    '''
    arms = 2
    alpha = params[0]
    epsilon = params[1]
    tau = params[2]
    c = coupling #between zero  and one
    probs = np.delete(np.arange(0.1, 1, 0.1), 4)
    choices = [] #all choices across all sessions
    rewards = [] #all rewards across all sessions
    rewards_array = [] #all session-wise reward lists
    regret_array = [] #all session-wise regret lists
    p_left = []
    for i in range(sessions):
        Q = np.zeros(2)
        p1 = np.random.choice(probs)
        p2 = np.random.choice(probs)
        prob = [p1, p2]
        p_left.append(p2)
        best = np.maximum(prob[0], prob[1]) #optimal arm for this session
        r_session = [] # records rewards for each session, then resets
        regret_session = [] # records regret for each session, then resets
        for trial in range(trials):
            if policy == 'softmax':
                exp_Q = np.exp(Q/tau) 
                sfx_Q = exp_Q/np.sum(exp_Q) # softmax probabilities
                j= np.random.choice(range(arms),p=sfx_Q) # picks one arm based on softmax probability
                if np.random.random()<prob[j]: r = 1
                else: r = 0  
                # print(j,r)
            if policy == 'e-greedy':
                if np.random.random()<epsilon:
                    j = np.random.choice(range(arms))
                else:
                    j = np.argmax(Q)
                if np.random.random()<prob[j]: r = 1
                else: r = 0 
                # print(j, r)
            Q[j] = Q[j] + ((r - Q[j])*alpha)
            if j == 0:
                Q[1] = Q[1] - c*((r - Q[j])*alpha)
            if j == 1:
                Q[0] = Q[0] - c*((r - Q[j])*alpha)                
            r_session.append(r) 
            rewards.append(r)
            choices.append(j) 
            #computing regret:
            if output == 'soft regret':
                regret = abs(np.subtract(best, prob[j]))
            if output == 'hard regret':
                if np.max(prob) == prob[j]:
                    regret = 0
                else:
                    regret = 1
            if (output == 'soft regret') | (output == 'hard regret'):
                regret_session.append(regret)
        rewards_array.append(pd.DataFrame(r_session).rolling(window).mean())
        if (output == 'soft regret') | (output == 'hard regret'):
            regret_array.append(pd.DataFrame(regret_session).rolling(window).mean())

        
    #plotting:
    if plot == True:
        if output == 'reward rate':
            rewards_average = np.mean(rewards_array, axis = 0).tolist()
            plt.plot((np.arange(0, len(rewards_average))), rewards_average)
            sns.despine()
            plt.show()
        if (output == 'soft regret') | (output == 'hard regret'):
            regret_average = np.mean(regret_array, axis = 0).tolist()
            plt.plot((np.arange(0, len(regret_average))), regret_average)
            sns.despine()
            plt.show()
        if output == 'none':
            print()
    else:
        if output == 'reward rate':
            rewards_average = np.mean(rewards_array, axis = 0).tolist()            
            return rewards_average
        if (output == 'soft regret') | (output == 'hard regret'):
            regret_average = np.mean(regret_array, axis = 0).tolist()
            return regret_average

    return choices, p_left, rewards
        
        
        
        
            
        
            
        
        