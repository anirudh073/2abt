import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix, classification_report


def choice_history(df, history_length, n_trials):
    
    ##extract only the first 100 trials from each session.
    pd.options.display.max_rows = 500
    df_truncated = df.groupby('session#').filter(lambda x: len(x['trial#'])>=n_trials)
    df_truncated = df_truncated.groupby('session#').head(n_trials).reset_index()
    sessions = pd.unique(df_truncated['session#'])
    df_choice = pd.DataFrame()
    hl_constant = history_length
    I = [] # stores indices to iterate over df_truncated
    i = 0
    
    ##produce indices to iterate over in df_truncated
    for e in range(len(sessions)): 
        for n in range(n_trials-history_length):
            I.append(i)
            i+= 1
        i+= (history_length)
    

    ##produce each column and insert it into df_chocie
    while history_length >= 0:
        df_choice.insert(0, ('c'+ str(hl_constant - history_length)), df_truncated['port'][[(e + history_length) for e in I]].reset_index(drop = True))
        history_length -= 1
    
    ##modify df_chocie  and split it into training and testing subsets
    df_choice = df_choice.replace({1: -1 , 2: 1})
    df_choice_train = df_choice[:][:-1000]
    df_choice_test = df_choice[:][-1000:]
    
    ##produce formula string based on history_length
    formula_string = 'c1'
    for i in range(hl_constant-1):
        formula_string = formula_string + ('+' + 'c'+str(i+2))
    formula = ('C(c0) ~ ' + formula_string)
    print('Formula: ', formula, '\n')
    

    ##fit model
    model = smf.glm(data = df_choice_train, formula = formula, family = sm.families.Binomial())
    result = model.fit()
    print(result.summary(), '\n')

    ##make predictions and confusion matrix
    predictions = result.predict(df_choice_test)
    predictions_nominal = [-1 if x>0.5 else 1 for x in predictions]
    cm = confusion_matrix(df_choice_test['c0'], predictions_nominal, labels = [-1, 1])
    
    ##output heatmap
    ax = sns.heatmap(cm, annot = True, xticklabels = ['-1', '1'], fmt = 'g', yticklabels = ['-1', '1'], cmap=sns.cubehelix_palette(as_cmap=True))
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    
    ##print classification report
    print('CLASSIFICATION REPORT: ')
    print(pd.DataFrame(classification_report(df_choice_test['c0'], predictions_nominal, digits = 3, output_dict = True)))


    
def reward_history(df, history_length, n_trials):
    
    ##extract only the first 100 trials from each session.
    pd.options.display.max_rows = 500
    df_truncated = df.groupby('session#').filter(lambda x: len(x['trial#'])>=n_trials)
    df_truncated = df_truncated.groupby('session#').head(n_trials).reset_index()
    sessions = pd.unique(df_truncated['session#'])
    df_reward = pd.DataFrame()
    hl_constant = history_length
    I = [] # stores indices to iterate over df_truncated
    i = 0
    
    ##produce indices to iterate over in df_truncated
    for e in range(len(sessions)): 
        for n in range(n_trials-history_length):
            I.append(i)
            i+= 1
        i+= (history_length)
    
    ## create and modify list of choices made:
    choice_list = df_truncated['port'][[(e + hl_constant) for e in I]].reset_index(drop = True)
    for m in range(len(choice_list)):
        if choice_list[m] == 1:
            choice_list[m] = -1
        if choice_list[m] == 2:
            choice_list[m] = 1
    
    ##produce each column and insert it into df_reward
    while history_length >= 0:
        df_reward.insert(0, ('r'+ str(hl_constant - history_length)), df_truncated['reward'][[(e + history_length) for e in I]].reset_index(drop = True))
        history_length -= 1
    
    ##modify df_reward and split it into training and testing dataframes
    remove_r0 = df_reward.pop('r0')
    df_reward = df_reward.replace({1: -1 , 2: 1})
    df_reward.insert(0, 'c0', choice_list)
    df_reward_train = df_reward[:][:-1000]
    df_reward_test = df_reward[:][-1000:]
    # print(df_reward)
    ##produce formula string based on history_length
    formula_string = 'C(r1)'
    for i in range(hl_constant-1):
        formula_string = formula_string + ('+' + 'C(r'+str(i+2) + ')')
    formula = ('C(c0) ~ ' + formula_string)
    print('Formula: ', formula, '\n')
    

    ##fit model
    model = smf.glm(data = df_reward_train, formula = formula, family = sm.families.Binomial())
    result = model.fit()
    print(result.summary(), '\n')
    

    ##make predictions and confusion matrix - 
    predictions = result.predict(df_reward_test)
    predictions_nominal = [-1 if x>0.5 else 1 for x in predictions]
    cm = confusion_matrix(df_reward_test['c0'], predictions_nominal, labels = [-1, 1])
    
    ##output heatmap
    ax = sns.heatmap(cm, annot = True, xticklabels = ['-1', '1'], fmt = 'g', yticklabels = ['-1', '1'], cmap=sns.cubehelix_palette(as_cmap=True))
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    
    ##print classification report
    print('CLASSIFICATION REPORT: ')
    print(pd.DataFrame(classification_report(df_reward_test['c0'], predictions_nominal, digits = 3, output_dict = True)))
    

def interaction_history(df, history_length, n_trials):
    pd.options.display.max_rows = 500
    df_truncated = df.groupby('session#').filter(lambda x: len(x['trial#'])>=n_trials)
    df_truncated = df_truncated.groupby('session#').head(n_trials).reset_index()
    sessions = pd.unique(df_truncated['session#'])
    df_reward = pd.DataFrame()
    df_choice = pd.DataFrame()
    hl_constant = history_length
    I = [] # stores indices to iterate over df_truncated
    i = 0
    
    ##produce indices to iterate over in df_truncated
    for e in range(len(sessions)): 
        for n in range(n_trials-history_length):
            I.append(i)
            i+= 1
        i+= (history_length)
        
    while history_length >= 0:
        df_choice.insert(0, ('c'+ str(hl_constant - history_length)), df_truncated['port'][[(e + history_length) for e in I]].reset_index(drop = True))
        history_length -= 1

    df_choice = df_choice.replace({1: -1 , 2: 1})
    
    history_length = hl_constant
    while history_length >= 0:
        df_reward.insert(0, ('r'+ str(hl_constant - history_length)), df_truncated['reward'][[(e + history_length) for e in I]].reset_index(drop = True))
        history_length -= 1    
    # print(df_reward)
    remove_r0 = df_reward.pop('r0')
    df_reward = df_reward.replace({0: 0 , 1: 1})

    frames = [df_reward, df_choice]
    df_concat = pd.concat(frames, axis = 1)
    df_concat_train = df_concat[:][:-1000]
    df_concat_test = df_concat[:][-1000:]

    ##produce formula string based on history_length #reward is catagorical #other ways to make reward catagorical? 
    formula_string = '(c1:C(r1))'
    
    for i in range(hl_constant-1): 
        formula_string = formula_string + ('+ ' + '(' +  'c'+str(i+2) + ':' + 'C(r' + str(i+2) + ')' + ')') #add what should be catagorical in the function
        # formula_string = formula_string + ('+' + 'c' + str(i+2) + '' + '+' +  'r' + str(i+2))
    formula = ('C(c0) ~ ' + formula_string)
    print('Formula: ', formula, '\n')
    
 
    ##fit model
    model = smf.glm(data = df_concat_train, formula = formula, family = sm.families.Binomial())
    result = model.fit()
    print(result.summary(), '\n')

    ##make predictions and confusion matrix
    predictions = result.predict(df_concat_test)
    predictions_nominal = [-1 if x>0.5 else 1 for x in predictions]
    cm = confusion_matrix(df_concat_test['c0'], predictions_nominal, labels = [-1, 1])
    
    ##output heatmap
    ax = sns.heatmap(cm, annot = True, xticklabels = ['-1', '1'], fmt = 'g', yticklabels = ['-1', '1'], cmap=sns.cubehelix_palette(as_cmap=True))
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    
    ##print classification report
    print('CLASSIFICATION REPORT: ')
    print(pd.DataFrame(classification_report(df_concat_test['c0'], predictions_nominal, digits = 3, output_dict = True)))
    
    return result.params
    
def coef_plots(coefficients, history):   
    coefficients
    choice = []
    reward = []
    c_r = []
    i, j, k = 1, 2, 3
    for e in range(int(len(coefficients)/3)):
        choice.append(coefficients[0][i])
        i += 3
        reward.append(coefficients[0][j])
        j += 3
        c_r.append(coefficients[0][k])
        k += 3
    print(choice)
    fig, ax = plt.subplots(3, layout= 'tight')
    for i in range(3):
        ax[i].set_ylim(-2, 1)
        ax[i].set_xlim(0, history)
        ax[i].set_xticks(np.arange(0, history, 1))
        ax[i].plot(range(history+1), np.zeros(history + 1), linewidth = 0.5)
    ax[0].plot(np.arange(1, (history + 1), 1), choice, 'b' )
    ax[0].set_ylabel('chocie')
    ax[1].plot(np.arange(1, (history + 1), 1), reward, 'g')
    ax[1].set_ylabel('reward')
    ax[2].plot(np.arange(1, (history + 1), 1), c_r, 'r')
    ax[2].set_ylabel('chocie*reward')
    ax[2].set_xlabel('trial#')
    sns.despine()        
    
def coef_plots_int(coefficients, history):   
    coefficients
    c_r = []
    i = 1
    for e in range(int(len(coefficients)-1)):
        c_r.append(coefficients[0][i])
        i += 1
    plt.plot(np.arange(1, (history+1), 1), c_r, 'r') 
    plt.ylim(-0.5, 0)
    plt.ylabel('choice*reward')
    sns.despine()        
    
    
    
# create df with switch coded as a catagorical(0 or 1)
# for trial n, switch equals 1 if choice(n) - choice(n-1) = 0
def switch_int(df, history_length, n_trials):
    pd.options.display.max_rows = 500
    df_truncated = df.groupby('session#').filter(lambda x: len(x['trial#'])>=n_trials)
    df_truncated = df_truncated.groupby('session#').head(n_trials).reset_index()
    sessions = pd.unique(df_truncated['session#'])
    print(sessions)
    df_reward = pd.DataFrame()
    df_choice = pd.DataFrame()
    hl_constant = history_length
    I = [] # stores indices to iterate over df_truncated
    i = 0
    
    ##produce indices to iterate over in df_truncated
    for e in range(len(sessions)): 
        for n in range(n_trials-history_length):
            I.append(i)
            i+= 1
        i+= (history_length)
        
    while history_length >= 0:
        df_choice.insert(0, ('c'+ str(hl_constant - history_length)), df_truncated['port'][[(e + history_length) for e in I]].reset_index(drop = True))
        history_length -= 1

    df_choice = df_choice.replace({1: -1 , 2: 1}) 
    
    history_length = hl_constant # reset history length
    while history_length >= 0:
        df_reward.insert(0, ('r'+ str(hl_constant - history_length)), df_truncated['reward'][[(e + history_length) for e in I]].reset_index(drop = True))
        history_length -= 1    
    # print(df_reward)
    switches = []
    s = 0
    for l in range(len(df_choice)-1):
        if df_choice['c0'][s+1] == df_choice['c0'][s]:
            switches.append(0)
        else:
            switches.append(1)
        s += 1
       
    #calculate 'probability of switching' as a moving average with a gaussian window
    print('start',  '\n')
    print('switches: ', (pd.Series(switches).head(100)))
    print('end', '\n')
    #compute a moving average over 'switches' using a gaussian window:
    switches_av = pd.DataFrame(switches).rolling(10, win_type = 'gaussian').mean(std = 1, sym = True)
    switches_av = switches_av.rename(columns = {0:'s'})
    
    print('switches: ', switches_av[40:100])
   
    remove_r0 = df_reward.pop('r0')
    remove_c0 = df_choice.pop('c0') #remove unwaneted columns from df
    df_reward = df_reward.replace({0: -1 , 1: 1})

    frames = [df_reward, df_choice, switches] #store dfs to concatenate, use 'switches' or 'switches_av'  
    # df_rc = pd.concat(frames, axis = 1)
    df_concat = pd.concat(frames, axis = 1)
    print(len(df_concat))
    print(df_concat.dropna()) ## why do NANs exist?
    df_concat_train = df_concat.dropna()[:][:-900]
    df_concat_test = df_concat.dropna()[:][-900:]

    ##produce formula string based on history_length #reward is catagorical #other ways to make reward catagorical? 
    formula_string = '(r1)'
    
    for i in range(hl_constant-1): 
        formula_string = formula_string + ('+ ' + '(r' + str(i+2) + ')' ) #add what should be catagorical in the function
        # formula_string = formula_string + ('+' + 'c' + str(i+2) + '' + '+' +  'r' + str(i+2))
    formula = ('C(s) ~ ' + formula_string)
    print('Formula: ', formula, '\n')
    
 
    ##fit model
    model = smf.glm(data = df_concat_train, formula = formula, family = sm.families.Binomial())
    result = model.fit()
    print(result.summary(), '\n')

    ##make predictions and confusion matrix

    predictions = result.predict(df_concat_test)
    # for i in predictions:
    #     if i < 0.9:
            # print(i)
    predictions_nominal = [0 if x>0.5 else 1 for x in predictions]
    # print(predictions_nominal[:])
    cm = confusion_matrix(df_concat_test['s'], predictions_nominal, labels = [0, 1])
    
    ##output heatmap
    ax = sns.heatmap(cm, annot = True, xticklabels = ['0', '1'], fmt = 'g', yticklabels = ['0', '1'], cmap=sns.cubehelix_palette(as_cmap=True))
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    
    ##print classification report
    print('CLASSIFICATION REPORT: ')
    print(pd.DataFrame(classification_report(df_concat_test['s'], predictions_nominal, digits = 3, output_dict = True)))
    
    return result.params