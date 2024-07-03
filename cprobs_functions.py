#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 12:46:23 2021

@author: celiaberon
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

def list_to_str(seq):
    
    '''take list of ints/floats and convert to string'''
    
    seq = [str(el) for el in seq] # convert element of sequence to string
    
    return ''.join(seq) # flatten list to single string

def encode_as_ab(row, symm):
    
    '''
    converts choice/outcome history to character code where where letter represents choice and case outcome
    INPUTS:
        - row: row from pandas DataFrame containing named variables 'decision_seq' and 'reward_seq' (previous N decisions/rewards) 
        - symm (boolean): if True, symmetrical encoding with A/B for direction (A=first choice in sequence)
                          if False, R/L encoding right/left choice
    OUTPUTS:
        - (string): string of len(decision_seq) trials encoding each choice/outcome combination per trial
    
    '''
    
    if int(row.decision_seq[0]) & symm: # symmetrical mapping based on first choice in sequence 1 --> A
        mapping = {('0','0'): 'b', ('0','1'): 'B', ('1','0'): 'a', ('1','1'): 'A'} 
    elif (int(row.decision_seq[0])==0) & symm: # symmetrical mapping for first choice 0 --> A    
        mapping = {('0','0'): 'a', ('0','1'): 'A', ('1','0'): 'b', ('1','1'): 'B'} 
    else: # raw right/left mapping (not symmetrical)
        mapping = {('0','0'): 'r', ('0','1'): 'R', ('1','0'): 'l', ('1','1'): 'L'} 

    return ''.join([mapping[(c,r)] for c,r in zip(row.decision_seq, row.reward_seq)])


def add_history_cols(df, N):
    
    '''
    INPUTS:
        - df (pandas DataFrame): behavior dataset
        - N (int): number trials prior to to previous trial to sequence (history_length)
        
    OUTPUTS:
        - df (pandas DataFrame): add columns:
            - 'decision_seq': each row contains string of previous decisions t-N, t-N+1,..., t-1
            - 'reward_seq': as in decision_seq, for reward history
            - 'history': encoded choice/outcome combination (symmetrical)
            - 'RL_history': encoded choice/outcome combination (raw right/left directionality)
       
    '''
    from numpy.lib.stride_tricks import sliding_window_view
    
    df['decision_seq']=np.nan # initialize column for decision history (current trial excluded)
    df['reward_seq']=np.nan # initialize column for laser stim history (current trial excluded)

    df = df.reset_index(drop=True) # need unique row indices (likely no change)

    for session in df.Session.unique(): # go by session to keep boundaries clean

        d = df.loc[df.Session == session] # temporary subset of dataset for session
        df.loc[d.index.values[N:], 'decision_seq'] = \
                                    list(map(list_to_str, sliding_window_view(d.Decision.astype('int'), N)))[:-1]

        df.loc[d.index.values[N:], 'reward_seq'] = \
                                    list(map(list_to_str, sliding_window_view(d.Reward.astype('int'), N)))[:-1]

        df.loc[d.index.values[N:], 'history'] = \
                                    df.loc[d.index.values[N:]].apply(encode_as_ab, args=([True]), axis=1)

        df.loc[d.index.values[N:], 'RL_history'] = \
                                    df.loc[d.index.values[N:]].apply(encode_as_ab, args=([False]), axis=1)
        
    return df
        
           
def calc_conditional_probs(df, symm, action=['Switch'], run=0):

    '''
    calculate probabilities of behavior conditional on unique history combinations
    
    Inputs:
        df (pandas DataFrame): behavior dataset
        symm (boolean): use symmetrical history (True) or raw right/left history (False)
        action (string): behavior for which to compute conditional probabilities (should be column name in df)
        
    OUTPUTS:
        conditional_probs (pandas DataFrame): P(action | history) and binomial error, each row for given history sequence
    '''
 
    group = 'history' if symm else 'RL_history' # define columns for groupby function

    max_runs = len(action) - 1 # run recursively to build df that contains summary for all actions listed

    conditional_probs = df.groupby(group).agg(
        paction=pd.NamedAgg(action[run], np.mean),
        n = pd.NamedAgg(action[run], len),
    ).reset_index()
    conditional_probs[f'p{action[run].lower()}_err'] = np.sqrt((conditional_probs.paction * (1 - conditional_probs.paction))
                                                  / conditional_probs.n) # binomial error
    conditional_probs.rename(columns={'paction': f'p{action[run].lower()}'}, inplace=True) # specific column name
    
    if not symm:
        conditional_probs.rename(columns={'RL_history':'history'}, inplace=True) # consistent naming for history
    
    if max_runs == run:
    
        return conditional_probs
    
    else:  
        run += 1
        return pd.merge(calc_conditional_probs(df, symm, action, run), conditional_probs.drop(columns='n'), on='history')

def sort_cprobs(conditional_probs, sorted_histories):
    
    '''
    sort conditional probs by reference order for history sequences to use for plotting/comparison
    
    INPUTS:
        - conditional_probs (pandas DataFrame): from calc_conditional_probs
        - sorted_histories (list): ordered history sequences from reference conditional_probs dataframe
    OUTPUTS:
        - (pandas DataFrame): conditional_probs sorted by reference history order
    '''
    
    from pandas.api.types import CategoricalDtype
    
    cat_history_order = CategoricalDtype(sorted_histories, ordered=True) # make reference history ordinal
    
    conditional_probs['history'] = conditional_probs['history'].astype(cat_history_order) # apply reference ordinal values to new df
    
    return conditional_probs.sort_values('history') # sort by reference ordinal values for history

def rename_df(df): #rename rat df to match with the inputs of celiaberon funcs
    df = df.rename(columns = {'session#':'Session', 'port':'Decision', 'reward':'Reward', 'trial#':'Trial'})
    df['Decision'] = df['Decision'].replace({1:0, 2:1})
    return df
def get_mod_df(df, N): #add history columns and insert 'Switch' column
    mod_df = add_history_cols(df.groupby('Session').filter(lambda x: len(x)>50), N = N)
    actions = np.abs(np.diff(mod_df['Decision'].values.astype('float64')))
    actions = np.insert(actions, 0, 0)
    mod_df.insert(loc=4, column='Switch', value=actions)
    return mod_df
def get_conditional_probs(df, N = 3): 
    ''' input - original rat df, N(histoy length)
        output - conditional switch probabilities
    '''
    renamed_df = rename_df(df)
    mod_df = get_mod_df(renamed_df, N)
    conditional_probs = calc_conditional_probs(mod_df, symm = True)
    return conditional_probs


def group_choices(cprobs_str, cprobs_ustr,sort_order, comb_list = ['aaa', 'aab', 'aba', 'abb']):
    '''
    group conditinal probabilities by choice history
    cprobs_str, cprobs_ustr : averaged dataframes for str and unstr environment. Insert empty dataframe to exclude.
    comb_list: list of choice history combinaitons
    sort_order: sorting for history combinations, list
    '''
    
    cprobs_str_list = cprobs_str
    cprobs_ustr_list = cprobs_ustr
    
    cprobs_str_df = pd.DataFrame(data = {'history':sort_order, 'pswitch': cprobs_str})
    cprobs_ustr_df = pd.DataFrame(data = {'history':sort_order, 'pswitch': cprobs_ustr})
    
    cprobs_str_lower = cprobs_str_df['history'].map(lambda x : x.lower())
    cprobs_ustr_lower = cprobs_ustr_df['history'].map(lambda x : x.lower())


    fig, ax = plt.subplots(2, 2, figsize = (12, 6), layout = 'tight')
    fig.suptitle('Group by choice history')
    loc = [(0, 0), (0, 1), (1, 0), (1,1)] #list of (row, column) tuples
    i = 0
    for comb in comb_list:
        row, column = loc[i][0], loc[i][1] 
        #store indices of each combination:
        comb_str = cprobs_str_lower[cprobs_str_lower== comb].index.tolist()
        comb_ustr = cprobs_ustr_lower[cprobs_ustr_lower== comb].index.tolist()
        #calculate standard error
        err_str = stats.sem([cprobs_str_list[comb_str], cprobs_str_list[comb_str]])
        err_ustr = stats.sem([cprobs_ustr_list[comb_ustr], cprobs_ustr_list[comb_ustr]]) 
        #plot
        df_str = cprobs_str_df
        df_ustr = cprobs_ustr_df
        
        ax[row][column].bar(x =df_str[(df_str['history'].map(lambda x : x.lower())) == comb].history , height = df_str[(df_str['history'].map(lambda x : x.lower())) == comb].pswitch, color = 'g', alpha = 0.5, yerr = err_str)
        ax[row][column].bar(x = df_ustr[(df_ustr['history'].map(lambda x : x.lower())) == comb].history, height = df_ustr[(df_ustr['history'].map(lambda x : x.lower())) == comb].pswitch, color = 'r', alpha = 0.5, yerr = err_ustr)
        ax[row][column].set_ylim(bottom = 0)
        ax[row][column].title.set_text(comb_list[i].upper())
        i+=1
    
    return

def group_choices_v_model(cprobs_str, cprobs_model, cprobs_str_err, cprobs_model_err, sort_order, comb_list = ['aaa', 'aab', 'aba', 'abb']):
    '''
    group conditinal probabilities by choice history
    cprobs_str, cprobs_ : averaged dataframes for str and unstr environment. Insert empty dataframe to exclude.
    comb_list: list of choice history combinaitons
    sort_order: sorting for history combinations, list
    '''
    
    cprobs_str_list = cprobs_str
    cprobs_ustr_list = cprobs_model
    
    cprobs_str_df = pd.DataFrame(data = {'history':sort_order, 'pswitch': cprobs_str})
    cprobs_ustr_df = pd.DataFrame(data = {'history':sort_order, 'pswitch': cprobs_model})
    
    cprobs_str_lower = cprobs_str_df['history'].map(lambda x : x.lower())
    cprobs_ustr_lower = cprobs_ustr_df['history'].map(lambda x : x.lower())


    fig, ax = plt.subplots(2, 2, figsize = (12, 6), layout = 'tight')
    fig.suptitle('Group by choice history')
    loc = [(0, 0), (0, 1), (1, 0), (1,1)] #list of (row, column) tuples
    i = 0
    for comb in comb_list:
        row, column = loc[i][0], loc[i][1] 
        #store indices of each combination:
        comb_str = cprobs_str_lower[cprobs_str_lower== comb].index.tolist()
        comb_ustr = cprobs_ustr_lower[cprobs_ustr_lower== comb].index.tolist()
        #calculate standard error
        err_str = stats.sem([cprobs_str_list[comb_str], cprobs_str_list[comb_str]])
        err_ustr = stats.sem([cprobs_ustr_list[comb_ustr], cprobs_ustr_list[comb_ustr]]) 
        #plot
        df_str = cprobs_str_df
        df_ustr = cprobs_ustr_df
        
        ax[row][column].bar(x =df_str[(df_str['history'].map(lambda x : x.lower())) == comb].history , height = df_str[(df_str['history'].map(lambda x : x.lower())) == comb].pswitch, color = 'g', alpha = 0.5, yerr = err_str)
        ax[row][column].bar(x = df_ustr[(df_ustr['history'].map(lambda x : x.lower())) == comb].history, height = df_ustr[(df_ustr['history'].map(lambda x : x.lower())) == comb].pswitch, color = 'r', alpha = 0.5, yerr = err_ustr)
        ax[row][column].set_ylim(bottom = 0)
        ax[row][column].title.set_text(comb_list[i].upper())
        i+=1
    
    return

    

def convert_to_bool(str):
    '''
    
    input - str of length n
    output - bool (1/0) number representing the upper/lowercase pattern of input str
    '''
    bool_string = ''
    for e in str:
        if e.isupper() == True:
            bool_string += '1'
        else: 
            bool_string += '0'

    return bool_string
            

def group_reward(cprobs_str, cprobs_ustr, comb_list, sort_order):
    '''
    '''
    #reward histories: [111, 110, 101, 100, 000, 011, 010, 001]
    #convert upper-lower combinatins to 10
    cprobs_str_df = pd.DataFrame(data = {'history':sort_order, 'pswitch': cprobs_str})
    cprobs_ustr_df = pd.DataFrame(data = {'history':sort_order, 'pswitch': cprobs_ustr})    
    
    fig, ax = plt.subplots(4, 4, layout = 'tight', figsize = (8, 3))
    
    cprobs_str_df_bool = pd.DataFrame(data = {'history': cprobs_str_df['history'].map(lambda x: convert_to_bool(x)), 'pswitch':cprobs_str_df['pswitch']})     
    cprobs_ustr_df_bool = pd.DataFrame(data = {'history': cprobs_ustr_df['history'].map(lambda x: convert_to_bool(x)), 'pswitch':cprobs_ustr_df['pswitch']})     

    cprobs_str_df_bool.groupby('history').mean(numeric_only = True)
    cprobs_ustr_df_bool.groupby('history').mean(numeric_only = True)


    ax.bar(x = cprobs_str_df_bool['history'], height = cprobs_str_df_bool['pswitch'], alpha = 0.5)
    ax.bar(x = cprobs_ustr_df_bool['history'], height  = cprobs_ustr_df_bool['pswitch'], alpha = 0.5)
    
    
def group_by_reward(cprobs_str, cprobs_ustr, cprobs_array_str, cprobs_array_ustr,  sort_order, comb_list = ['111', '110', '101', '100', '000', '001', '010', '011']):
    '''
    '''
    cprobs_str_df = pd.DataFrame(data = {'history':sort_order, 'pswitch': cprobs_str})
    cprobs_ustr_df = pd.DataFrame(data = {'history':sort_order, 'pswitch': cprobs_ustr})     

    fig, ax = plt.subplots(2, 4, layout = 'tight', figsize = (15, 5))
    fig.suptitle('Group by reward')

    cprobs_str_df_bool = pd.DataFrame(data = {'history_bool': cprobs_str_df['history'].map(lambda x: convert_to_bool(x)), 'history':cprobs_str_df['history'] , 'pswitch':cprobs_str_df['pswitch']})     
    cprobs_ustr_df_bool = pd.DataFrame(data = {'history_bool': cprobs_ustr_df['history'].map(lambda x: convert_to_bool(x)),'history': cprobs_ustr_df['history'] ,  'pswitch':cprobs_ustr_df['pswitch']})  
    loc = [(0, 0), (0, 1), (0, 2), (0,3), (1, 0), (1, 1), (1, 2), (1,3) ] #list of (row, column) tuples
    i = 0
    
    for comb in comb_list:
        row, column = loc[i][0], loc[i][1]
        comb_str = cprobs_str_df_bool[cprobs_str_df_bool['history_bool']==comb].index.tolist()
        comb_ustr = cprobs_str_df_bool[cprobs_ustr_df_bool['history_bool']==comb].index.tolist()
        print(cprobs_str_list[1][comb_str])

        err_str = stats.sem([cprobs_str_list[0][comb_str], cprobs_str_list[1][comb_str]])
        err_ustr = stats.sem([cprobs_ustr_list[0][comb_str], cprobs_ustr_list[1][comb_ustr]])

        df_str = cprobs_str_df_bool
        df_ustr = cprobs_ustr_df_bool

        ax[row][column].bar(x = df_str[(df_str['history_bool'])== comb].history, height = df_str[(df_str['history_bool'])==comb].pswitch, color = 'g', alpha = 0.5, yerr = err_str)
        ax[row][column].bar(x = df_ustr[(df_ustr['history_bool'])== comb].history, height = df_ustr[(df_ustr['history_bool'])==comb].pswitch, color = 'r', alpha = 0.5, yerr = err_ustr)
        ax[row][column].set_ylim(bottom = 0)
        ax[row][column].title.set_text(comb_list[i])
        i+=1
    
        

    
        

























    



