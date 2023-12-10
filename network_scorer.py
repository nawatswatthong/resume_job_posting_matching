import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import ast
from tqdm import tqdm

def create_count_df(df, col_name, 
    remove_list = ['e', 'ul', 'css', 'equal', 'tools', 'color', 'webkit', 'center', 'industry',
                   'disability', 'disabilities', 'accessible', 'com', 'level', 'maintain', 'best practices', 'additional',
                   'big datum','track','resources','medical','personal','programming languages','work environment','receive','agile','job description','limited','equity','www','translate',
                   'data data','consumer','data sets','datasets', 'physical','recruitment process', 'pipelines','access', 'addition','levels','long term','reach', 'data models',
                   'data','data sources','libraries','maintaining','business process','execute',
                   'use cases','vaccinated','analytic','resource','safe','data collection',
                   'subject matter','adoption','nice'],
    skill_count = 300
):
    
    all_value_list = set()
    for vals in df[col_name]:
        val_list = [val for val in set(ast.literal_eval(vals)) if val not in remove_list]
        for val in val_list:
            all_value_list.add(val)
    all_value_list = list(all_value_list)
    count_df = pd.DataFrame(
        np.zeros((len(all_value_list), len(all_value_list))), 
        columns=all_value_list,
        index=all_value_list)

    for vals in tqdm(df[col_name]):
        val_list = [val for val in set(ast.literal_eval(vals)) if val not in remove_list]
        for val in val_list:
            count_df.loc[val, val_list] += 1
    
    top_skill_idx = np.argsort(np.diag(count_df))[::-1][:skill_count]
    count_df_final = count_df.iloc[top_skill_idx,top_skill_idx]
    count_df_final = np.log(count_df_final+1) # normalizing values
    top_skill_list = [count_df.columns[i] for i in top_skill_idx]
    return count_df_final, top_skill_list
    
def create_network(count_df):  
    count_matrix = np.array(count_df) + 0.00001
    graph = nx.from_numpy_array(1/count_matrix, create_using=nx.DiGraph)
    return graph

def resume_jd_matching_score(
        r_val_list:list, jd_val_list:list, all_val_list:list, 
        network, count_df:pd.DataFrame, show_sub_score=False):
    score = 0
    for r_val in list(set(r_val_list)):
        if r_val in all_val_list and r_val in list(set(jd_val_list)): # skill matched
            node_score = count_df.loc[r_val, r_val]
            if show_sub_score:
                print(r_val, node_score)
            score += node_score
        elif r_val in all_val_list: # use shortest path
            shortest_path = 999999
            for jd_val in list(set(jd_val_list)):
                if jd_val not in all_val_list:
                    continue
                s = nx.shortest_path_length(
                    network, 
                    source=all_val_list.index(r_val), 
                    target=all_val_list.index(jd_val), 
                    weight='weight'
                )
                if s < shortest_path:
                    shortest_path = s
            if show_sub_score:
                print(r_val, 1/shortest_path)
            score += 1/shortest_path
    return score