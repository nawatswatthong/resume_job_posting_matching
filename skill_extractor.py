import pandas as pd
import numpy as np
import json
import spacy
from spacy.matcher import PhraseMatcher

from skillNer.general_params import SKILL_DB
from skillNer.skill_extractor_class import SkillExtractor

class Skill_extractor():
    def __init__(self) -> None:
        self.nlp = spacy.load("en_core_web_lg")
        self.skill_extractor = SkillExtractor(self.nlp, SKILL_DB, PhraseMatcher)

        self.DATA_PATH = "skill_db_relax_20.json"
        with open(self.DATA_PATH, 'r') as f:
            self.test = json.load(f)

        self.match_list = ['full_matches', 'ngram_scored']

    def preprocess_job_description(self, job_description): # limit each text to not longer than 3500 characters
        temp = job_description.split('\n')
        jd_list = []
        for s in temp:
            if s == '':
                continue
            num_split = int(np.ceil(len(s)/3000))
            len_per_split = len(s)/num_split
            for i in range(num_split):
                jd_list.append(s[int(i*len_per_split): int((i+1)*len_per_split)].strip())
        return jd_list
        
        
    def skill_extract(self, job_description):
        jd_list = self.preprocess_job_description(job_description)
        skill_dic = {'Hard Skill' : set(), 'Soft Skill' : set()}
        for jd in jd_list:
            try:
                annotations = self.skill_extractor.annotate(jd)
                for match in self.match_list:
                    for i in range(len(annotations['results'][match])):
                        idx = annotations['results'][match][i]['skill_id']
                        skill_type = self.test[idx]['skill_type']
                        if skill_type in ['Hard Skill', 'Soft Skill']:
                            skill_dic[skill_type].add(annotations['results'][match][i]['doc_node_value'])
            except:
                continue
        return skill_dic
