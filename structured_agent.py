import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def structured(trials, sessions, probs, coupling, policy, params, window, output, plot): #vectorise!
    ''' 
    probs; 'all' or specify as percentages
    coupling parameter defines degree of structure (0=<c=<1)
    policy: 'softmax' or 'epsilon-greedy'
    params in order: alpha, epsilon, tau
    window; window size for moving average
    output; 'reward rate' / 'soft regret' / 'hard regret'
    plot; Bool
    store choice, reward, trial#, session#, rewprob1, rewprob2, better rewprob
    '''
    arms = 2
    alpha = params[0]
    epsilon = params[1]
    tau = params[2]
    c = coupling #between zero  and one
    if probs == 'all':
        probs = [10, 20, 30, 40, 60, 70, 80, 90]
    else:
        probs = probs
    choices = [] #all choices across all sessions
    rewards = [] #all rewards across all sessions
    rewards_array = [] #all session-wise reward lists
    regret_array = [] #all session-wise regret lists
    session_num = []
    Trial = 0
    trial_num = []
    p_left = []
    rewprobfull1 = []
    rewprobfull2 = []
    rw = []
    for i in range(sessions):
        Q = np.ones(2)/2
        p1 = np.random.choice(probs)
        p2 = 100-p1
        prob = [p1, p2]
        p_left.append(p1)
        # print('prob: ', prob)
        best = np.maximum(prob[0], prob[1]) #optimal arm for this session
        r_session = [] # records rewards for each session, then resets
        regret_session = [] # records regret for each session, then resets
        for trial in range(trials):
            session_num.append(i+1) #add to 'session#' column of df
            trial_num.append(Trial+1) #add to 'trial' column of df
            Trial += 1
            rewprobfull1.append(p1) #add to 'rewprobfull1 column of df
            rewprobfull2.append(p2) #add to 'rewprobfull2' column of df
            if policy == 'softmax':
                exp_Q = np.exp(Q/tau) 
                sfx_Q = exp_Q/np.sum(exp_Q) # softmax probabilities
                j= np.random.choice(range(arms),p=sfx_Q) # picks one arm based on softmax probability
                if j == 0:
                    choice = 1
                elif j == 1:
                    choice = 2
                if np.random.random()*100<prob[j]: r = 1
                else: r = 0  
                # print('choice, r:',  choice,r)
            elif policy == 'e-greedy':
                if np.random.random()<epsilon:
                    j = np.random.choice(range(arms))
                else:
                    j = np.argmax(Q)
                if j == 0:
                    choice = 1
                elif j == 1:
                    choice = 2
                if np.random.random()*100<prob[j]: r = 1
                else: r = 0 
                # print(j, r)
            rw.append(prob[j])
            Q[j] = Q[j] + ((r - Q[j])*alpha)
            if j == 0:
                Q[1] = Q[1] - c*((r - Q[j])*alpha)
            if j == 1:
                Q[0] = Q[0] - c*((r - Q[j])*alpha) 
            # print('q:', Q)
            r_session.append(r)  
            rewards.append(r)
            choices.append(choice) 
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
            return pd.DataFrame(data = {'trial#' : trial_num, 'session#' : session_num, 'port': choices, 'reward' : rewards, 'rewprobfull1' : rewprobfull1, 'rewprobfull2' : rewprobfull2, 'rw' : rw})
        if (output == 'soft regret') | (output == 'hard regret'):
            regret_average = np.mean(regret_array, axis = 0).tolist()
            return pd.DataFrame(data = {'trial#' : trial_num, 'session#' : session_num, 'port': choices, 'reward' : rewards, 'rewprobfull1' : rewprobfull1, 'rewprobfull2' : rewprobfull2, 'rw' : rw})
    
def unstructured(trials, sessions, probs, coupling, policy, params, window, output, plot):
    ''' 
    probs; 'all' or specify as percentages
    coupling parameter defines degree of structure (0=<c=<1)
    policy: 'softmax' or 'epsilon-greedy'
    params in order: alpha, epsilon, tau
    window; window size for moving average
    output; 'reward rate' / 'soft regret' / 'hard regret'
    plot; Bool
    store all choices and rewards in separate arrays
    '''
    arms = 2
    alpha = params[0]
    epsilon = params[1]
    tau = params[2]
    c = coupling #between zero  and one
    if probs == 'all':
        probs = [10, 20, 30, 40, 60, 70, 80, 90]
    else:
        probs = probs
    choices = [] #all choices across all sessions
    rewards = [] #all rewards across all sessions

    
    rewards_array = [] #all session-wise reward lists
    regret_array = [] #all session-wise regret lists
    session_num = []
    trial_num = []
    Trial = 0
    p_left = []
    rewprobfull1 = []
    rewprobfull2 = []
    rw = []
    for i in range(sessions):
        Q = np.ones(2)/2
        p1 = np.random.choice(probs)
        p2 = np.random.choice(probs)
        prob = [p1, p2]
        p_left.append(p1)
        # print(prob)
        best = np.maximum(prob[0], prob[1]) #optimal arm for this session
        r_session = [] # records rewards for each session, then resets
        regret_session = [] # records regret for each session, then resets
        for trial in range(trials):
            session_num.append(i+1) #add to 'session#' column of df
            trial_num.append(Trial+1) #add to 'trial' column of df
            Trial += 1
            rewprobfull1.append(p1) #add to 'rewprobfull1 column of df
            rewprobfull2.append(p2) #add to 'rewprobfull2' column of df
            if policy == 'softmax':
                exp_Q = np.exp(Q/tau) 
                sfx_Q = exp_Q/np.sum(exp_Q) # softmax probabilities
                j= np.random.choice(range(arms),p=sfx_Q) # picks one arm based on softmax probability
                if np.random.random()*100<prob[j]: r = 1
                else: r = 0  
                # print(j,r)
            if policy == 'e-greedy':
                if np.random.random()<epsilon:
                    j = np.random.choice(range(arms))
                else:
                    j = np.argmax(Q)
                if np.random.random()*100<prob[j]: r = 1
                else: r = 0 
                # print(j, r)
            rw.append(prob[j])
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
            return pd.DataFrame(data = {'trial#' : trial_num, 'session#' : session_num, 'port': choices, 'reward' : rewards, 'rewprobfull1' : rewprobfull1, 'rewprobfull2' : rewprobfull2, 'rw' : rw})
        if (output == 'soft regret') | (output == 'hard regret'):
            regret_average = np.mean(regret_array, axis = 0).tolist()
            return pd.DataFrame(data = {'trial#' : trial_num, 'session#' : session_num, 'port': choices, 'reward' : rewards, 'rewprobfull1' : rewprobfull1, 'rewprobfull2' : rewprobfull2, 'rw' : rw})




        
        
            
        
            
        
        
