import pandas as pd
import numpy as np
import re
import os
import nltk.corpus
nltk.download('stopwords')
from nltk.corpus import stopwords
from  datetime import datetime
from collections import defaultdict
from tqdm import tqdm


def tokenize(text):
    stops = stopwords.words('english')
    text_list = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", " ", text).split()
    text_list = [word for word in text_list if word not in (stops)]
    return text_list

def find_index_in_tokens(tokens, flag_options):
    for flag in flag_options:
        if isinstance(flag, str):   # single word
            if flag in tokens:
                return tokens.index(flag)
        if isinstance(flag, list):  # multple words
            # print(1)
            next_idx_set = set([idx for idx in range(len(tokens))])
            for i, f in enumerate(flag):
                idx_set = {j for j in next_idx_set if j < len(tokens) and tokens[j] == f}
                if len(idx_set) == 0:
                    break
                next_idx_set = {j+1 for j in idx_set}
                if i == len(flag)-1:
                    # print(6)
                    return np.min(list(idx_set))
    return -1
def find_end_index(x, flag, flag_list):
    if x[f'{flag}_start_index'] == -1:
        return -1
    possible_index_list = [x[f'{f}_start_index'] for f in flag_list if x[f'{f}_start_index'] > x[f'{flag}_start_index']]
    if possible_index_list == []:
        return -1
    return np.sort(possible_index_list)[0]
def split_part(x, flag):
    if x[f'{flag}_start_index'] == -1:
        return []
    if x[f'{flag}_end_index'] == -1:
        return x['tokens'][x[f'{flag}_start_index']:]
    return x['tokens'][x[f'{flag}_start_index']:x[f'{flag}_end_index']]


def find_year_experience(tokens, experience_start_index, experience_end_index):
    MIN_YEAR = 1980
    MAX_YEAR = 2023
    month_list = {
        'january':1, 'february':2, 'march':3, 'april':4, 'may':5, 'june':6, 
        'july':7, 'august':8, 'september':9, 'october':10, 'november':11, 'december':12,
        'jan':1, 'feb':2, 'mar':3, 'apr':4, 'jun':6, 'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12
    }

    temp_tokens = [(token, idx) for idx, token in enumerate(tokens[experience_start_index:experience_end_index]) if token.isnumeric() or token in month_list]
    temp_tokens = [(int(token), idx) if token.isnumeric() else (month_list[token], idx) for token, idx in temp_tokens]

    t_exp_tokens = defaultdict(list)
    grp_idx = -999
    i = 0
    cutoff = 7
    while i < len(temp_tokens):
        # print('\t', temp_tokens[i][1])
        if temp_tokens[i][1] > grp_idx + cutoff:
            grp_idx = temp_tokens[i][1]
            # print('\t\t', temp_tokens[i][1])
        if temp_tokens[i][0] >= MIN_YEAR and temp_tokens[i][0] <= MAX_YEAR:
            t_exp_tokens[grp_idx].append((1, temp_tokens[i][1])) # add dummy month
            t_exp_tokens[grp_idx].append(temp_tokens[i])
            i += 1
        elif i < len(temp_tokens)-1 and temp_tokens[i][0] >= 1 and temp_tokens[i][0] <= 12 and \
            temp_tokens[i+1][0] >= MIN_YEAR and temp_tokens[i+1][0] <= MAX_YEAR and temp_tokens[i+1][1] < grp_idx + cutoff:
            t_exp_tokens[grp_idx].append(temp_tokens[i])
            t_exp_tokens[grp_idx].append(temp_tokens[i+1])
            i += 2
        elif i < len(temp_tokens)-2 and temp_tokens[i][0] >= 1 and temp_tokens[i][0] <= temp_tokens[i+1][0] and \
            temp_tokens[i+1][0] <= 12 and temp_tokens[i+1][1] < grp_idx + cutoff and\
            temp_tokens[i+2][0] >= MIN_YEAR and temp_tokens[i+2][0] <= MAX_YEAR and temp_tokens[i+2][1] < grp_idx + cutoff:
            t_exp_tokens[grp_idx].append(temp_tokens[i])
            t_exp_tokens[grp_idx].append((temp_tokens[i+2][0], temp_tokens[i][1])) # add dummy year
            t_exp_tokens[grp_idx].append(temp_tokens[i+1])
            t_exp_tokens[grp_idx].append(temp_tokens[i+2])
            i += 3
        else:
            i += 1

    start_date = datetime(1900, 1, 1)
    min_date = datetime.now()
    year_experience = -3
    for grp in t_exp_tokens:
        start_idx = -1
        temp_list = t_exp_tokens[grp]
        for i in range(len(temp_list)):
            if len(temp_list) == 2:
                cur_date = datetime(temp_list[i+1][0], temp_list[i][0], 1)
                if cur_date < min_date:
                    year_experience += (min_date-cur_date).days/365
                    min_date = cur_date
                break
            elif temp_list[i][0] <= 12:
                cur_date = datetime(temp_list[i+1][0], temp_list[i][0], 1)
                if start_idx == -1:
                    start_date = cur_date
                    start_idx = i
                elif cur_date < start_date and start_idx >= 0:
                    if start_date < min_date:
                        year_experience += (min_date-start_date).days/365
                        min_date = start_date
                    start_date = cur_date
                    start_idx = i
                    # print(grp, temp_list[i], (datetime.now()-start_date).days/365)
                elif cur_date > start_date and start_idx >= 0:
                    if start_date < min_date:
                        year_experience += (min(min_date,cur_date)-start_date).days/365
                        min_date = start_date
                    # year_experience += (cur_date-start_date).days/365
                    start_idx = -1
                    # print(grp, temp_list[i], (cur_date-start_date).days/365)
    return year_experience