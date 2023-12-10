import spacy
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

class Education_extractor():
    def __init__(self, uni_rank_path='world_uni_rank.csv') -> None:
        
        self.nlp = spacy.load("en_core_web_lg")

        self.major = [
            {"label": "MAJOR", "pattern": "computer science"}, 
            {"label": "MAJOR", "pattern": "data science"},
            {"label": "MAJOR", "pattern": "statistics"},
            {"label": "MAJOR", "pattern": "engineering"},
            {"label": "MAJOR", "pattern": "quantitve"},
            {"label": "MAJOR", "pattern": "math"},
            {"label": "MAJOR", "pattern": "mathematics"},
            {"label": "MAJOR", "pattern": "information technology"},
            {"label": "MAJOR", "pattern": "data analytics"},
            {"label": "MAJOR", "pattern": "business"},
            {"label": "MAJOR", "pattern": "finance"},
            {"label": "MAJOR", "pattern": "economics"},
            {"label": "MAJOR", "pattern": "biostatistics"},
            {"label": "MAJOR", "pattern": "physics"}
        ]

        self.degree = [
            {"label": "DEGREE", "pattern": "master"},
            {"label": "DEGREE", "pattern": "phd"},
            {"label": "DEGREE", "pattern": "bachelor"},
            {"label": "DEGREE", "pattern": "ms"},
            {"label": "DEGREE", "pattern": "bs"},
            {"label": "DEGREE", "pattern": "doctor"}
        ]


        self.ruler = self.nlp.add_pipe("entity_ruler", before="ner")
        self.ruler.add_patterns(self.major)
        self.ruler.add_patterns(self.degree)

        self.world_uni = pd.read_csv(uni_rank_path)
        self.world_uni['institution_edited'] = self.world_uni['institution'].str.replace('University', '')
        self.world_uni['institution_edited'] = self.world_uni['institution_edited'].apply(lambda x: x.lower() if isinstance(x, str) else x)
        patterns = [{'label': 'University', 'pattern': ent} for ent in self.world_uni['institution_edited'].tolist()]
        self.ruler.add_patterns(patterns)

        self.model_name = "all-mpnet-base-v2" # ref: https://huggingface.co/sentence-transformers/all-mpnet-base-v2
        self.model = SentenceTransformer(self.model_name) 
        self.token_embedding = self.model.encode(self.world_uni['institution_edited'])

    def extract_education_features(self, text):
        text = text.lower()
        doc = self.nlp(text)

        majors = []
        degrees = []
        org = []

        for ent in doc.ents:
            if ent.label_ == "MAJOR":
                majors.append(ent.text)
            elif ent.label_ == "DEGREE":
                degrees.append(ent.text)
            elif ent.label_ == "University":
                org.append(ent.text)


        R = []
        for i in org:
            uni_extract_embedding = self.model.encode(i)
            r = np.argmax(self.token_embedding.dot(uni_extract_embedding)) +1
            R.append(r)
        if len(R) == 0:
            rank = -1
        else:
            rank = min(R)

        result_dict = {
            "majors": majors,
            "degrees": degrees,
            "org": org,
            "org_rank": rank
        }
        return result_dict