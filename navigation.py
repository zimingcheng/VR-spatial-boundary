# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 22:21:57 2023

@author: czm19
"""

#### Load in packages ####
# data management packages
import numpy as np
import pandas as pd
# date packages
from datetime import datetime
# stats packages
from scipy.stats import norm
# plotting packages
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go

removed = ['04','09']
ptps = ['14','15','16','17','18','19','21','22','23','24','27','28','29','30',
        '31','32','33','34','35','36','37','38','39','41','42','43','45','46',
        '47','48','49','50','51','52']
# outliers = ['20','26']

#### Read in and clean data sheet ####
# store each type of dataframe within a list
dfs_event, dfs_location, dfs_property = [],[],[]
for ptp in ptps:
    # read csv files
    df_ptp_event = pd.read_csv(f'../Data/Navigation/sub-{ptp}/EpochData.csv')
    df_ptp_location = pd.read_csv(f'../Data/Navigation/sub-{ptp}/ObjectData.csv')
    df_ptp_property = pd.read_csv(f'../Data/Navigation/sub-{ptp}/PropertyData.csv')
    # remove unnecessary rows
    df_ptp_property['non-numeric'] = pd.to_numeric(df_ptp_property[' PropertyName'], errors='coerce')
    df_ptp_property = df_ptp_property[df_ptp_property['non-numeric'].isnull()]
    df_ptp_property = df_ptp_property.dropna(axis=1, how='all')
    df_ptp_property = df_ptp_property[df_ptp_property[' Event'].isin([' Block-End',' Trial-End',' Task-End'])]
    # add a column of participant number to each resulting dataframe
    df_ptp_event['ptp'] = ptp
    df_ptp_location['ptp'] = ptp
    df_ptp_property['ptp'] = ptp
    # add the dataframe to its respective list
    dfs_event.append(df_ptp_event)
    dfs_location.append(df_ptp_location)
    dfs_property.append(df_ptp_property)
# concat the lists to form a mega dataframe with all participants
df_event = pd.concat(dfs_event).reset_index(drop=True)
df_location = pd.concat(dfs_location).reset_index(drop=True)
df_property = pd.concat(dfs_property).reset_index(drop=True)
# strip the column names
df_event.columns = [col.strip() for col in df_event.columns]
df_location.columns = [col.strip() for col in df_location.columns]
df_property.columns = [col.strip() for col in df_property.columns]
# strip all the string columns
df_event_str = df_event.select_dtypes(['object'])
df_event[df_event_str.columns] = df_event_str.apply(lambda x: x.str.strip())
df_location_str = df_location.select_dtypes(['object'])
df_location[df_location_str.columns] = df_location_str.apply(lambda x: x.str.strip())
df_property_str = df_property.select_dtypes(['object'])
df_property[df_property_str.columns] = df_property_str.apply(lambda x: x.str.strip())
# output df contents to csv for inspection
df_event.to_csv('cleaned/df_event_cleaned.csv')
df_location.to_csv('cleaned/df_location_cleaned.csv')
df_property.to_csv('cleaned/df_property_cleaned.csv')

# get the order of conditions
df_conditions = df_event[(df_event['TaskName'].str.contains('Object1'))][['TaskName','ptp','BlockNumber']].reset_index(drop=True)
list_condition_order = [1,2,3,4,5,6] * len(df_conditions['ptp'].unique())
df_conditions['ConditionOrder'] = list_condition_order
df_conditions['TaskName'] = df_conditions['TaskName'].apply(lambda x: x.split('_')[0])
df_conditions.columns = ['TaskName','ptp','BlockNumber','ConditionOrder']

# get the condition and order of each object
df_object = df_property[
    (df_property['Event']=='Block-End') &
    (df_property['BlockNumber']!=1) &
    (df_property['PropertyName'].isin(['Object_SLSR','Object_SLDR','Object_DLDR']))]\
    [['ptp','BlockNumber','PropertyName','ListItems']].reset_index(drop=True)
df_object['1'],df_object['2'],df_object['3'],df_object['4'] = '','','',''
df_object['5'],df_object['6'],df_object['7'],df_object['8'] = '','','',''
for index, row in df_object.iterrows():
    # get the list of selected objects for the specific participant
    block2_selected_object = df_property[
        (df_property['Event']=='Block-End') &
        (df_property['BlockNumber']==2) &
        (df_property['TaskName']=='TE_route') &
        (df_property['ptp']==row['ptp']) &
        # the last bit was done to convert a list in string (i.e. '[1,2,3]') to a real list
        (df_property['PropertyName']=='Selected_Objects')].reset_index().at[0, 'ListItems'][1:-1].split('; ')
    block3_selected_object = df_property[
        (df_property['Event']=='Block-End') &
        (df_property['BlockNumber']==3) &
        (df_property['TaskName']=='TE_route') &
        (df_property['ptp']==row['ptp']) &
        # the last bit was done to convert a list in string (i.e. '[1,2,3]') to a real list
        (df_property['PropertyName']=='Selected_Objects')].reset_index().at[0, 'ListItems'][1:-1].split('; ')
    # keep only the condition name
    df_object.at[index,'PropertyName'] = row['PropertyName'].split('_')[-1]
    # get the list of items (in index of <all_selected_object>)
    objects_idx = row['ListItems'][1:-1].split(';')
    objects = []
    for object_id in objects_idx:
        if row['BlockNumber'] == 2:
            objects.append(block2_selected_object[int(object_id)])
        elif row['BlockNumber'] == 3:
            objects.append(block3_selected_object[int(object_id)])
    df_object.at[index,'1']=objects[0]
    df_object.at[index,'2']=objects[1]
    df_object.at[index,'3']=objects[2]
    df_object.at[index,'4']=objects[3]
    df_object.at[index,'5']=objects[4]
    df_object.at[index,'6']=objects[5]
    df_object.at[index,'7']=objects[6]
    df_object.at[index,'8']=objects[7]
df_object = pd.melt(df_object, id_vars=['ptp','BlockNumber','PropertyName','ListItems'],var_name='ObjectOrder',value_name='Object')
df_object.columns = ['ptp','BlockNumber','TaskName','ObjectIdx','ObjectOrder','Object']
df_object = df_object.merge(df_conditions, on=['ptp','BlockNumber','TaskName'])
df_object = df_object.sort_values(['ptp','ConditionOrder','ObjectOrder'])
df_object.to_csv('cleaned/df_object_cleaned.csv')

#### TE route ####
# get the estimated time for the entire route
# <df_TE_route> has one row per condition per participant
df_TE_route = df_property[
    (df_property['Event']=='Trial-End') & 
    (df_property['TaskName']=='TE_route') & 
    (df_property['PropertyName']=='TE_route')][['ptp','BlockNumber','PropertyValue']].reset_index(drop=True)
df_TE_route.columns = ['ptp','block','TE_route']
df_TE_route['TE_route'] = pd.to_numeric(df_TE_route['TE_route'])
df_TE_route['condition'] = df_conditions['TaskName']

# get the ground truth time for the entire route
# <df_truth_route> has one row per condition per participant
# get all the events related to each condition
df_truth_route = df_event[
    (df_event['TaskName'].str.contains('SLSR_')) |
    (df_event['TaskName'].str.contains('SLDR_')) |
    (df_event['TaskName'].str.contains('DLDR_'))][['ptp','BlockNumber','TaskName','Duration']]
# remove instruction and examples
df_truth_route = df_truth_route[
    (~df_event['TaskName'].str.contains('Instr')) &
    (~df_event['TaskName'].str.contains('Eg'))].reset_index(drop=True)
# rename each task to only reflect the condition
df_truth_route['TaskName'] = df_truth_route['TaskName'].apply(lambda x: x.split('_')[0])
# change duration from string to float
df_truth_route['Duration'] = df_truth_route['Duration'].apply(lambda x: float(x[6:]))
# calculate the sum of duration
df_truth_route = df_truth_route.groupby(['ptp','BlockNumber','TaskName']).sum().reset_index()
df_truth_route.columns = ['ptp','block','condition','truth_route']

# Combine estimated with truth time
df_TE_route = df_TE_route.merge(df_truth_route, on=['ptp','block','condition'])

# Calculate the difference between estimated and truth time
# df_TE_route['diff_TE_route'] = (df_TE_route['TE_route'] - df_TE_route['truth_route']) / (df_TE_route['TE_route'] + df_TE_route['truth_route'])
df_TE_route['diff_TE_route'] = df_TE_route['TE_route'] / df_TE_route['truth_route']

# mean center by participant to control for difference in individual baseline
# this is because we don't care about the absolute values of estimated and truth navigation time
# but rather we care about how different conditions compare to each other
df_TE_route_mean = df_TE_route.groupby('ptp').mean().reset_index()
df_TE_route_mean.columns = ['ptp','block','TE_route_mean','truth_route_mean','diff_TE_route_mean']
df_TE_route = df_TE_route.merge(df_TE_route_mean, on='ptp')
df_TE_route['TE_route_centered'] = df_TE_route['TE_route'] - df_TE_route['TE_route_mean']
df_TE_route['truth_route_centered'] = df_TE_route['truth_route'] - df_TE_route['truth_route_mean']
df_TE_route['diff_TE_route_centered'] = df_TE_route['diff_TE_route'] - df_TE_route['diff_TE_route_mean']

df_TE_route = df_TE_route.sort_values(['ptp','block_x','condition'])

# <df_TE_route> has one row per condition per participant
df_TE_route.to_csv('cleaned/df_TE_route.csv')

# df_TE_route = df_TE_route[df_TE_route['ptp'] != '01']
df_TE_route = df_TE_route.groupby(['ptp','condition']).mean().reset_index()

p = sns.barplot(data=df_TE_route, x='condition', y='diff_TE_route')
p.set_xticklabels(['different','duplicated','same'])
plt.title('')
plt.xlabel('navigation condition')
plt.ylabel('time estimation (navigation)')
plt.savefig('plots/TE_route.svg', format='svg')
plt.show()


#### TE trial ####
# get the estimated time for each trial
df_TE = df_property[
    (df_property['Event']=='Task-End') & 
    (df_property['TaskName']=='TE') & 
    (df_property['PropertyName'].isin(['leftTE','rightTE','OrdDisc_correct','TE_Response']))]\
    [['ptp','BlockNumber','TaskNumber','PropertyName','PropertyValue']].reset_index(drop=True)
df_TE = df_TE.pivot(index=['ptp','BlockNumber','TaskNumber'],columns='PropertyName',values='PropertyValue').reset_index()
df_TE['LaterObject'] = ''
for index, row in df_TE.iterrows():
    if row['OrdDisc_correct'] == '1':
        df_TE.at[index,'LaterObject'] = row['leftTE']
    else:
        df_TE.at[index,'LaterObject'] = row['rightTE']
        
# one caveat here is that the actual TE includes the M/N decision time of the later object. I don't think this is a big deal, 
# because the decision time should not differ by conditions, and when the participant answer the question, they might have taken
# the decision time into account anyways
df_TE_truth = df_event[
    (df_event['TaskName'].str.contains('SLSR_')) |
    (df_event['TaskName'].str.contains('SLDR_')) |
    (df_event['TaskName'].str.contains('DLDR_'))][['ptp','BlockNumber','TaskName','Duration']]
# remove instruction and examples
df_TE_truth = df_TE_truth[
    (~df_event['TaskName'].str.contains('Instr')) &
    (~df_event['TaskName'].str.contains('Eg'))].reset_index(drop=True)
# change duration from string to float
df_TE_truth['Duration'] = df_TE_truth['Duration'].apply(lambda x: float(x[6:]))

# add the WP duration to the next object (e.g. WP1 duration is added to Object2, as it is between Object1 and Object2)
df_TE_truth['BetweenObjectDuration'] = np.nan
for index, row in df_TE_truth.iterrows():
    if 'Object1' in row['TaskName']:
        wp_duration = 0
    elif ('WP' in row['TaskName']) and ('8' not in row['TaskName']):
        wp_duration = row['Duration']
    elif 'Object' in row['TaskName']:
        df_TE_truth.at[index,'BetweenObjectDuration'] = row['Duration'] + wp_duration
    orig_TaskName = row['TaskName']
    df_TE_truth.at[index,'TaskName'] = orig_TaskName.split('_')[0]
    
# keep only the object trials
df_TE_truth = df_TE_truth.dropna()
# add object order
ObjectOrder = ['2','3','4','5','6','7','8'] * (6 * len(ptps))
df_TE_truth['ObjectOrder'] = ObjectOrder
# match object name based on object order and condition
df_TE_truth = df_TE_truth.merge(df_object, on=['ptp','BlockNumber','TaskName','ObjectOrder'])
df_TE_truth.rename(columns = {'Object':'LaterObject'}, inplace = True)

df_TE = df_TE.merge(df_TE_truth, on=['ptp','BlockNumber','LaterObject'])
df_TE = df_TE[['ptp','BlockNumber','TaskName','TE_Response','BetweenObjectDuration','ObjectOrder','ConditionOrder']]
df_TE.rename(columns = {'BetweenObjectDuration':'TE_Truth'}, inplace = True)
df_TE['TE_Response'] = pd.to_numeric(df_TE['TE_Response'])

df_TE['TE_diff'] = df_TE['TE_Response'] / df_TE['TE_Truth']

# mean center by participant to control for difference in individual baseline
df_TE_mean = df_TE.groupby('ptp').mean().reset_index()
df_TE_mean.columns = ['ptp','BlockNumber','TE_Response_mean','TE_Truth_mean','ConditionOrder','TE_diff_mean']
df_TE = df_TE.merge(df_TE_mean, on='ptp')
df_TE['TE_Response_centered'] = df_TE['TE_Response'] - df_TE['TE_Response_mean']
df_TE['TE_Truth_centered'] = df_TE['TE_Truth'] - df_TE['TE_Truth_mean']
df_TE['TE_diff_centered'] = df_TE['TE_diff'] - df_TE['TE_diff_mean']

df_TE = df_TE.sort_values(['ptp','BlockNumber_x','TaskName'])


# <df_TE> has one row per trial
df_TE.to_csv('cleaned/df_TE.csv')

df_TE_group = df_TE.groupby(['ptp','TaskName']).mean().reset_index()

p = sns.barplot(data=df_TE_group, x='TaskName', y='TE_diff')
p.set_xticklabels(['different','duplicated','same'])
plt.title('')
plt.xlabel('navigation condition')
plt.ylabel('time estimation (trial)')
plt.savefig('plots/TE_trial.svg', format='svg')
plt.show()

df_TE['ObjectOrder'] = pd.to_numeric(df_TE['ObjectOrder'])
# p = sns.catplot(data=df_TE, x='ObjectOrder',col='TaskName',y='TE_diff',kind='bar',order=['2','3','4','5','6','7','8'])
p = sns.lineplot(data=df_TE, x='ObjectOrder',hue='TaskName',y='TE_diff')
p.set_xticks([2,3,4,5,6,7,8])
p.set_xticklabels(['1','2','3','4','5','6','7'])
plt.xlabel('object order')
plt.ylabel('time estimation (trial)')
plt.savefig('plots/TE_trial_order.svg', format='svg')
plt.show()

#### Order Discrimination ####
# get the participant response and ground truth order for each trial
df_OrdDisc = df_property[
    (df_property['Event']=='Task-End') & 
    (df_property['TaskName']=='TE') & 
    (df_property['PropertyName'].isin(['leftTE','rightTE','OrdDisc_Response','OrdDisc_correct']))]\
    [['ptp','BlockNumber','TaskNumber','PropertyName','PropertyValue','ListItems']].reset_index(drop=True)
# due to a bug, property value does not update after each trial. However, the data is not lost, but rather also recorded in 
# ListItems properly, so the below code manually update PropertyValue to be the last item in ListItems
for index, row in df_OrdDisc.iterrows():
    if row['PropertyName'] == 'OrdDisc_Response':
        df_OrdDisc.at[index,'PropertyValue'] = row['ListItems'][-2]
df_OrdDisc = df_OrdDisc.drop(columns=['ListItems'])
df_OrdDisc = df_OrdDisc.pivot(index=['ptp','BlockNumber','TaskNumber'],columns='PropertyName',values='PropertyValue').reset_index()
df_OrdDisc['OrdDisc_correct'] = df_OrdDisc['OrdDisc_correct'].map({'0':'L','1':'R'})
df_OrdDisc['OrdDisc_Accuracy'] = np.where(df_OrdDisc['OrdDisc_correct'] == df_OrdDisc['OrdDisc_Response'], 1, 0)

df_OrdDisc_n = dict(df_OrdDisc.groupby('ptp').count()['TaskNumber'])
df_OrdDict_condition_order = []
for ptp in ptps:
    n_repetition = df_OrdDisc_n[ptp]//6
    df_OrdDict_condition_order.extend([1] * n_repetition)
    df_OrdDict_condition_order.extend([2] * n_repetition)
    df_OrdDict_condition_order.extend([3] * n_repetition)
    df_OrdDict_condition_order.extend([4] * n_repetition)
    df_OrdDict_condition_order.extend([5] * n_repetition)
    df_OrdDict_condition_order.extend([6] * n_repetition)
df_OrdDisc['ConditionOrder'] = df_OrdDict_condition_order

# get the condition for each trial
df_OrdDisc = df_OrdDisc.merge(df_conditions[['ptp','ConditionOrder','TaskName']], on=['ptp','ConditionOrder'])
df_OrdDisc = df_OrdDisc.sort_values(['ptp','BlockNumber','TaskName'])

# study how the trial order during navigation affects order discrimination 
df_OrdDisc['Object'] = np.where(df_OrdDisc['OrdDisc_correct']=='L',
                                     df_OrdDisc['leftTE'],df_OrdDisc['rightTE'])
df_OrdDisc = df_OrdDisc.merge(df_object[['ptp','BlockNumber','TaskName','Object','ObjectOrder']], 
                              on=['ptp','BlockNumber','TaskName','Object'])
df_OrdDisc['ObjectOrder'] = pd.to_numeric(df_OrdDisc['ObjectOrder'])

# mean center by participant to control for difference in individual baseline
df_OrdDisc_Acc = df_OrdDisc.groupby(['ptp','BlockNumber','TaskName']).mean()['OrdDisc_Accuracy'].reset_index()
df_OrdDisc_Acc_mean = df_OrdDisc_Acc.groupby('ptp').mean().reset_index()
df_OrdDisc_Acc_mean.columns = ['ptp','BlockNumber','OrdDisc_Accuracy_mean']
df_OrdDisc_Acc = df_OrdDisc_Acc.merge(df_OrdDisc_Acc_mean, on='ptp')
df_OrdDisc_Acc['OrdDisc_Accuracy_centered'] = df_OrdDisc_Acc['OrdDisc_Accuracy'] - df_OrdDisc_Acc['OrdDisc_Accuracy_mean']


# <df_OrdDisc_Acc> with one row per participant per condition
df_OrdDisc_Acc.to_csv('cleaned/df_OrdDisc_Acc.csv')

df_OrdDisc_Acc = df_OrdDisc_Acc.groupby(['ptp','TaskName']).mean().reset_index()

#### RKN ####
# note: RKN_foil_DLSR is a typo when constructing the experiment, should be SLDR 
df_ON = df_property[
    (df_property['Event']=='Task-End') & 
    (df_property['TaskName'].isin(['RKN_target_SLSR','RKN_target_SLDR','RKN_target_DLDR',
                                   'RKN_lure_SLSR','RKN_lure_SLDR','RKN_lure_DLDR',
                                   'RKN_foil_SLSR','RKN_foil_DLSR','RKN_foil_DLDR'])) & 
    (df_property['PropertyName'].isin(['curr_RKN','RKN_correct','ON_response']))]\
    [['ptp','BlockNumber','TaskNumber','TaskName','PropertyName','PropertyValue']].reset_index(drop=True)
    
df_RK = df_property[
    (df_property['Event']=='Task-End') & 
    (df_property['TaskName'].isin(['RK'])) & 
    (df_property['PropertyName'].isin(['curr_RKN','RK_response']))]\
    [['ptp','BlockNumber','TaskNumber','TaskName','PropertyName','PropertyValue']].reset_index(drop=True)

df_ON = df_ON.pivot(index=['ptp','BlockNumber','TaskNumber','TaskName'],columns='PropertyName',values='PropertyValue').reset_index()
df_RK = df_RK.pivot(index=['ptp','BlockNumber','TaskNumber'],columns='PropertyName',values='PropertyValue').reset_index()
df_RK['TaskNumber'] = df_RK['TaskNumber'] - 1
df_RKN = pd.merge(df_ON, df_RK, on=['ptp','BlockNumber','TaskNumber','curr_RKN']).reset_index()
df_RKN['TaskName'] = df_RKN['TaskName'].replace(to_replace='RKN_foil_DLSR', value='RKN_foil_SLDR')

df_RKN['ON_correct'] = df_RKN['RKN_correct'].map({'0':'O','1':'S','2':'N'})
df_RKN['RK_response'] = df_RKN['RK_response'].map({'R':1,'K':0})
df_RKN['ON_Accuracy'] = np.nan
df_RKN['ON_Detection'] = ''
df_RKN = df_RKN.rename(columns={'curr_RKN':'Object'})
for index, row in df_RKN.iterrows():
    df_RKN.at[index,'TaskName'] = row['TaskName'].split('_')[-1]
    if row['ON_correct']=='O' and row['ON_response']=='O':
        df_RKN.at[index,'ON_Accuracy'] = 1
        df_RKN.at[index,'ON_Detection'] = 'O->O'
    elif row['ON_correct']=='O' and row['ON_response']=='N':
        df_RKN.at[index,'ON_Accuracy'] = 0
        df_RKN.at[index,'ON_Detection'] = 'O->N'
    elif row['ON_correct']=='O' and row['ON_response']=='S':
        df_RKN.at[index,'ON_Accuracy'] = 0
        df_RKN.at[index,'ON_Detection'] = 'O->S'
    elif row['ON_correct']=='N' and row['ON_response']=='O':
        df_RKN.at[index,'ON_Accuracy'] = 0
        df_RKN.at[index,'ON_Detection'] = 'N->O'
    elif row['ON_correct']=='N' and row['ON_response']=='N':
        df_RKN.at[index,'ON_Accuracy'] = 1
        df_RKN.at[index,'ON_Detection'] = 'N->N'
    elif row['ON_correct']=='N' and row['ON_response']=='S':
        df_RKN.at[index,'ON_Accuracy'] = 0
        df_RKN.at[index,'ON_Detection'] = 'N->S'
    elif row['ON_correct']=='S' and row['ON_response']=='O':
        df_RKN.at[index,'ON_Accuracy'] = 0
        df_RKN.at[index,'ON_Detection'] = 'S->O'
    elif row['ON_correct']=='S' and row['ON_response']=='N':
        df_RKN.at[index,'ON_Accuracy'] = 0
        df_RKN.at[index,'ON_Detection'] = 'S->N'
    elif row['ON_correct']=='S' and row['ON_response']=='S':
        df_RKN.at[index,'ON_Accuracy'] = 1
        df_RKN.at[index,'ON_Detection'] = 'S->S'
df_RKN_Acc = df_RKN.groupby(['ptp','TaskName']).mean().reset_index()[['ptp','TaskName','ON_Accuracy','RK_response']]
df_RKN_count = df_RKN.groupby(['ptp','TaskName','ON_Detection']).count().reset_index()[['ptp','TaskName','ON_Detection','ON_response']]
# construct the signal detection matrix
df_RKN_count_index = df_RKN_count.set_index(['ptp','TaskName','ON_Detection'])
df_RKN_count_0filled = df_RKN_count_index.unstack(level=['TaskName','ON_Detection']).fillna(0).stack(dropna=False)
df_RKN_count_0filled = df_RKN_count_0filled.fillna(0)
df_RKN_count_0filled = df_RKN_count_0filled.reset_index()
df_RKN_count_0filled.columns = pd.MultiIndex.from_tuples([('ptp','ptp'),('ON_Detection','ON_Detection'),
                                                          ('TaskName','DLDR'),('TaskName','SLDR'),('TaskName','SLSR')])
df_RKN_count_0filled_long = df_RKN_count_0filled.melt(col_level=1, id_vars=['ptp','ON_Detection'], var_name='TaskName', value_name='count')
df_RKN_count_group = df_RKN_count_0filled_long.groupby(['TaskName','ON_Detection']).mean().reset_index()

def get_RKN_count_group(df_RKN_count_group, task, ON_Detection):
    target_row = df_RKN_count_group[(df_RKN_count_group['TaskName']==task) & 
                                    (df_RKN_count_group['ON_Detection']==ON_Detection)]
    if len(target_row) == 0:
        return 0
    else:
        return target_row['count']
    
for task in ['DLDR','SLDR','SLSR']:
    df_RKN_count_group_sub = df_RKN_count_group[df_RKN_count_group['TaskName']==task]
    RKN_count_result = np.zeros((3,3))
    RKN_count_result[0,0] = get_RKN_count_group(df_RKN_count_group, task, 'O->O')
    RKN_count_result[0,1] = get_RKN_count_group(df_RKN_count_group, task, 'O->S')
    RKN_count_result[0,2] = get_RKN_count_group(df_RKN_count_group, task, 'O->N')
    RKN_count_result[1,0] = get_RKN_count_group(df_RKN_count_group, task, 'S->O')
    RKN_count_result[1,1] = get_RKN_count_group(df_RKN_count_group, task, 'S->S')
    RKN_count_result[1,2] = get_RKN_count_group(df_RKN_count_group, task, 'S->N')
    RKN_count_result[2,0] = get_RKN_count_group(df_RKN_count_group, task, 'N->O')
    RKN_count_result[2,1] = get_RKN_count_group(df_RKN_count_group, task, 'N->S')
    RKN_count_result[2,2] = get_RKN_count_group(df_RKN_count_group, task, 'N->N')
    sns.heatmap(RKN_count_result, annot=True, xticklabels=['Old','Sim','New'],
                yticklabels=['Target','Lure','Foil'], cmap='Reds', vmax=4, vmin=0)
    plt.title(f'signal detection in condition {task}')
    plt.xlabel('response')
    plt.ylabel('truth')
    plt.show()

# calculate lure discrimination index (shown in Start 2019, TICS)
# df_RKN_count_LDI = df_RKN_count_0filled_long[df_RKN_count_0filled_long['ON_Detection'].isin(['S->S','N->S','O->O','S->O'])]
df_RKN_count_LDI = df_RKN_count_0filled_long.pivot(index=['ptp','TaskName'],columns='ON_Detection',values='count').reset_index()
try: 
    df_RKN_count_LDI['N->S']
except:
    df_RKN_count_LDI['N->S'] = 0
df_RKN_count_LDI['LDI'] = df_RKN_count_LDI['S->S']/8 - df_RKN_count_LDI['N->S']/8
df_RKN_count_LDI['d_prime'] = df_RKN_count_LDI['S->S']/8 - df_RKN_count_LDI['S->O']/8

# mean center by participant to control for difference in individual baseline
df_RKN_count_LDI_mean = df_RKN_count_LDI.groupby('ptp').mean().reset_index()[['ptp','LDI','d_prime']]
df_RKN_count_LDI_mean.columns = ['ptp','LDI_mean','d_prime_mean']
df_RKN_count_LDI = df_RKN_count_LDI.merge(df_RKN_count_LDI_mean, on='ptp')
df_RKN_count_LDI['LDI_centered'] = df_RKN_count_LDI['LDI'] - df_RKN_count_LDI['LDI_mean']
df_RKN_count_LDI['d_prime_centered'] = df_RKN_count_LDI['d_prime'] - df_RKN_count_LDI['d_prime_mean']

p = sns.barplot(data=df_RKN_count_LDI, x='TaskName', y='LDI')
p.set_xticklabels(['different','duplicated','same'])
plt.title('')
plt.xlabel('navigation condition')
plt.savefig('plots/LDI.svg', format='svg')
plt.show()


p = sns.barplot(data=df_RKN_count_LDI, x='TaskName', y='d_prime')
p.set_xticklabels(['different','duplicated','same'])
plt.title('')
plt.xlabel('navigation condition')
plt.savefig('plots/d_prime.svg', format='svg')
plt.show()

# mean center by participant to control for difference in individual baseline
df_RKN_Acc_mean = df_RKN_Acc.groupby('ptp').mean().reset_index()
df_RKN_Acc_mean.columns = ['ptp','ON_Accuracy_mean','RK_response_mean']
df_RKN_Acc = df_RKN_Acc.merge(df_RKN_Acc_mean, on='ptp')
df_RKN_Acc['ON_Accuracy_centered'] = df_RKN_Acc['ON_Accuracy'] - df_RKN_Acc['ON_Accuracy_mean']
df_RKN_Acc['RK_response_centered'] = df_RKN_Acc['RK_response'] - df_RKN_Acc['RK_response_mean']

df_RKN_Acc.to_csv('cleaned/df_RKN_Acc.csv')
df_RKN_count_LDI.to_csv('cleaned/df_RKN_count_LDI.csv')

df_RKN_Rprop = df_RKN.groupby(['ptp','TaskName','ON_Detection']).mean().reset_index()[['ptp','TaskName','ON_Detection','RK_response']]
df_RKN_Rprop = df_RKN_Rprop[df_RKN_Rprop['ON_Detection'].isin(['O->O','N->N','S->S','S->O'])]
df_RKN_Rprop.to_csv('cleaned/df_RKN_Rprop.csv')

p = sns.barplot(data=df_RKN_Rprop, x='TaskName', y='RK_response', hue='ON_Detection', hue_order = ['O->O','S->O','S->S','N->N'])
p.set_xticklabels(['different','duplicated','same'])
p.legend(loc='best')
plt.title('')
plt.xlabel('navigation condition')
plt.ylabel('re-experience percentage')
plt.savefig('plots/R_percentage.svg', format='svg')
plt.show()

# order effect
df_object_order = df_object[['ptp','BlockNumber','TaskName','Object','ObjectOrder','ConditionOrder']]
df_object_order['Object'] = df_object_order['Object'].str[:-1]
df_RKN['Object'] = df_RKN['Object'].str.slice(start=0, stop=3)
df_RKN = df_RKN.merge(df_object_order, on=['ptp','BlockNumber','TaskName','Object'])
df_RKN['ObjectOrder'] = pd.to_numeric(df_RKN['ObjectOrder'])

df_RKN['ObjectOrder'] = pd.to_numeric(df_RKN['ObjectOrder'])
p = sns.lineplot(data=df_RKN, x='ObjectOrder',hue='TaskName',y='RK_response')
plt.xlabel('object order')
plt.ylabel('re-experience percentage')
plt.savefig('plots/R_percentage_order.svg', format='svg')
plt.show()

df_RKN.to_csv('cleaned/df_RKN.csv')




#### Order Discrimination with only Re-experienced trials ####
df_OrdDisc['leftTE'] = df_OrdDisc['leftTE'].str.slice(start=0, stop=3)
df_OrdDisc['rightTE'] = df_OrdDisc['rightTE'].str.slice(start=0, stop=3)
df_RKN_OrdDisc = df_RKN[['ptp','BlockNumber','TaskName','Object','RK_response']]
df_RKN_OrdDisc.columns = ['ptp','BlockNumber','TaskName','leftTE','left_RK']
df_OrdDisc_R = df_OrdDisc.merge(df_RKN_OrdDisc,
                                on=['ptp','BlockNumber','TaskName','leftTE'])
df_RKN_OrdDisc.columns = ['ptp','BlockNumber','TaskName','rightTE','right_RK']
df_OrdDisc_R = df_OrdDisc_R.merge(df_RKN_OrdDisc,
                                on=['ptp','BlockNumber','TaskName','rightTE'])
df_OrdDisc_R = df_OrdDisc_R[(df_OrdDisc_R['left_RK']==1) & (df_OrdDisc_R['right_RK']==1)].reset_index(drop=True)

df_OrdDisc_R_Acc = df_OrdDisc_R.groupby(['ptp','BlockNumber','TaskName']).mean()['OrdDisc_Accuracy'].reset_index()
df_OrdDisc_R_Acc_mean = df_OrdDisc_R_Acc.groupby('ptp').mean().reset_index()
df_OrdDisc_R_Acc_mean.columns = ['ptp','BlockNumber','OrdDisc_Accuracy_mean']
df_OrdDisc_R_Acc = df_OrdDisc_R_Acc.merge(df_OrdDisc_R_Acc_mean, on='ptp')
df_OrdDisc_R_Acc['OrdDisc_Accuracy_centered'] = df_OrdDisc_R_Acc['OrdDisc_Accuracy'] - df_OrdDisc_R_Acc['OrdDisc_Accuracy_mean']

df_OrdDisc_R_Acc.to_csv('cleaned/df_OrdDisc_R_Acc.csv')
df_OrdDisc_R.to_csv('cleaned/df_OrdDisc_R.csv')

p = sns.barplot(data=df_OrdDisc_R_Acc, x='TaskName', y='OrdDisc_Accuracy')
p.set_xticklabels(['different','duplicated','same'])
plt.title('')
plt.xlabel('navigation condition')
plt.ylabel('accuracy')
plt.savefig('plots/Ord_Disc.svg', format='svg')
plt.show()

df_OrdDisc_R['ObjectOrder'] = pd.to_numeric(df_OrdDisc_R['ObjectOrder'])
p = sns.lineplot(data=df_OrdDisc_R, x='ObjectOrder',hue='TaskName',y='OrdDisc_Accuracy')
plt.xlabel('object order')
plt.ylabel('accuracy')
plt.savefig('plots/Ord_Disc_order.svg', format='svg')
plt.show()



