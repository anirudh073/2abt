from patsy import dmatrices
from patsy import ContrastMatrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def obtain_df(df, history_length, n_trials):
    df_truncated = df.groupby('session#').filter(lambda x: len(x['trial#'])>=n_trials)
    df_truncated = df_truncated.groupby('session#').head(n_trials).reset_index()
    sessions = pd.unique(df_truncated['session#'])
    # print(sessions)
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
       
    # switches_av = pd.DataFrame(switches).rolling(10, win_type = 'gaussian').mean(std = 1, sym = True)
    # switches_av = switches_av.rename(columns = {0:'s'})
       
    remove_r0 = df_reward.pop('r0')
    # remove_c0 = df_choice.pop('c0') #remove unwaneted columns from df
    df_reward = df_reward.replace({0: -1 , 1: 1})
    # print(pd.DataFrame(switches))
    frames = [df_reward, df_choice, pd.DataFrame(switches).rename(columns = {0:'s'})] #store dfs to concatenate, use 'switches' or 'switches_av'  
    # df_rc = pd.concat(frames, axis = 1)
    df_concat = pd.concat(frames, axis = 1)
    return df_concat.dropna()

def design_matrix(df, dep_var, variables, history_length, intercept = False):
    if variables == 'reward':
        formula_string = 'r1'
        for i in range(history_length-1): 
            formula_string = formula_string + ('+ ' + 'r' + str(i+2))
        if dep_var == 'c':
            formula = ('c0 ~ ' + formula_string)
        if dep_var == 's':
            formula = ('s ~' + formula_string)
        print(formula)
        y,x = dmatrices(formula, df, return_type = 'dataframe')
    elif variables == 'choice':
        formula_string = 'c1'
        for i in range(history_length-1): 
            formula_string = formula_string + ('+ ' + 'c' + str(i+2))
        if dep_var == 'c':
            formula = ('c0 ~ ' + formula_string)
        if dep_var == 's':
            formula = ('s ~' + formula_string)
        print(formula)
        y,x = dmatrices(formula, df, return_type = 'dataframe')   
    elif variables == 'interaction':
        formula_string = '(c1:r1)'
        for i in range(history_length-1): 
            formula_string = formula_string + ('+ ' + '(' +  'c'+str(i+2) + ':' + 'r' + str(i+2) + ')')
        if dep_var == 'c':
            formula = ('c0 ~ ' + formula_string)
        if dep_var == 's':
            formula = ('s ~' + formula_string)   
        print(formula)
        y,x = dmatrices(formula, df, return_type = 'dataframe')  
    elif variables == 'all':
        formula_string = '(c1*r1)'
        for i in range(history_length-1): 
            formula_string = formula_string + ('+ ' + '(' +  'c'+str(i+2) + '*' + 'r' + str(i+2) + ')')
        if dep_var == 'c':
            formula = ('c0 ~ ' + formula_string)
        if dep_var == 's':
            formula = ('s ~' + formula_string)    
        print(formula)
        y,x = dmatrices(formula, df, return_type = 'dataframe')        
    if intercept == False:
        remove_intercept = x.pop('Intercept')
    return y, x
    
    
def logreg(x, y, intercept = False):
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25)
    model = LogisticRegression(fit_intercept = intercept, solver  = 'sag', C = 100000, random_state = 42, tol = 0.000001, max_iter =10000).fit(x_train, np.ravel(y_train))
    predictions = model.predict(x_test)
    cm = metrics.confusion_matrix(y_test, predictions)
    cm_norm = cm/cm.astype(np.float64).sum(axis = 1)[:, None]
    print('Confusion matrix: ')
    ax = sns.heatmap(cm_norm, annot = True, fmt = 'g', cmap = 'Blues')
    ax.set(xlabel = 'Predicted', ylabel = 'True')
    plt.show()
    print('CLASSIFICATION REPORT: ')
    print(pd.DataFrame(classification_report(y_test, predictions, digits = 3, output_dict = True)))
    return model, predictions, y_test
    
    
    