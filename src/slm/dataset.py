from pydantic.main import BaseModel
from typing import Dict, List, Optional, Set, Tuple, Union
import re
import random
from pathlib import Path
from argparse import Namespace
from logging import Logger
from tqdm import tqdm

from src.utils.utils import *


class Sample(BaseModel):
    sentence: str = ""
    sentence_ent_tag: str = ""
    relation: str = ""
    subject_entity: str = ""
    object_entity: str = ""
    subject_pattern: str = ""
    object_pattern: str = ""
    rel_pattern: str = ""
    token: List = []
    h: Dict = {}
    t: Dict = {}

    def encode_to_line(self) -> str:
        return json.dumps(dict(text=self.encode_to_x(), summary=self.encode_to_y())) + "\n"

    def encode_to_x(self) -> str:
        text = f"Sentence: {self.sentence_ent_tag} Subject entity: {self.subject_entity}. Object entity: {self.object_entity}."
        return text

    def encode_to_y(self) -> str:
        summary = f"Subject entity type: {self.subject_pattern}. Object entity type: {self.object_pattern}. " + \
                  f"Relationship: {self.rel_pattern}. " + \
                  f"Label: {convert_label2words(self.relation)}."
        return summary

    def to_dict(self) -> dict:
        return {
            'sentence': self.sentence,
            'sentence_ent_tag': self.sentence_ent_tag,
            'relation': self.relation,
            'subject_entity': self.subject_entity,
            'object_entity': self.object_entity,
            'subject_pattern': self.subject_pattern,
            'object_pattern': self.object_pattern,
            'rel_pattern': self.rel_pattern,
            'token': self.token,
            'h': self.h,
            't': self.t
        }

    @classmethod
    def decode(cls, rule: str) -> Optional[dict]:
        group = re.match(
            r'Subject entity type: (.+)\.\s*Object entity type: (.+)\.\s*Relationship: (.+)\.\s*Label: (.+)\.*',
            rule)
        if group:
            return {
                'rule': rule,
                'subject_pattern': group[1].strip(),
                'object_pattern': group[2].strip(),
                'rel_pattern': group[3].strip(),
                'premise': f"Subject entity type: {group[1]}. Object entity type: {group[2]}. " + \
                           f"Relationship: subject entity {group[3]} object entity. ",
                'pred_relation': remove_dot_end_symbol(group[4].strip()),
            }
        else:
            return {
                'rule': rule,
                'subject_pattern': None,
                'object_pattern': None,
                'rel_pattern': None,
                'premise': None,
                'pred_relation': None,
            }

class Dataset(BaseModel):
    samples: List[Sample]

    @classmethod
    def load(cls, path: str, max_count: int = None):
        # cls is the <class: Dataset>
        print(f"Loading training samples for SLM from {path}")
        res = []
        with open(path) as f:
            all_lines = f.readlines()
            for line in all_lines:
                if max_count is not None and len(res) >= max_count:
                    break
                ins = eval(line)
                # ins = json.loads(line)
                # print(ins.keys())
                tokens = ins['token'] if 'token' in ins.keys() else []
                h = ins['h'] if 'h' in ins.keys() else {}
                t = ins['t'] if 't' in ins.keys() else {}
                if 'sentence' in ins.keys():
                    sentence = ins['sentence']
                elif 'text' in ins.keys():
                    sentence = ins['text']
                elif 'token' in ins.keys():
                    sentence = ' '.join([convert_ptb_token(token) for token in ins['token']])
                else:
                    sentence = ''
                if 'sentence_ent_tag' not in ins.keys():
                    if 'text' in ins.keys():
                        tokens_ent_tag = []
                        for token_index, token in enumerate(ins['text']):
                            if token_index == ins['h']['pos'][0]:
                                tokens_ent_tag.append(' <Sub> ')
                            if token_index == ins['h']['pos'][1]:
                                tokens_ent_tag.append(' </Sub> ')
                            if token_index == ins['t']['pos'][0]:
                                tokens_ent_tag.append(' <Obj> ')
                            if token_index == ins['t']['pos'][1]:
                                tokens_ent_tag.append(' </Obj> ')
                            tokens_ent_tag.append(convert_ptb_token(token))
                        sentence_ent_tag = ''.join(tokens_ent_tag)
                    elif 'token' in ins.keys():
                        tokens_ent_tag = []
                        for token_index, token in enumerate(ins['token']):
                            if token_index == ins['h']['pos'][0]:
                                tokens_ent_tag.append('<Sub>')
                            if token_index == ins['h']['pos'][1]:
                                tokens_ent_tag.append('</Sub>')
                            if token_index == ins['t']['pos'][0]:
                                tokens_ent_tag.append('<Obj>')
                            if token_index == ins['t']['pos'][1]:
                                tokens_ent_tag.append('</Obj>')
                            tokens_ent_tag.append(convert_ptb_token(token))
                        sentence_ent_tag = ' '.join(tokens_ent_tag)
                    else:
                        sentence_ent_tag = ''
                else:
                    sentence_ent_tag = ins['sentence_ent_tag']
                if 'subject_entity' in ins.keys():
                    subject_entity = ins['subject_entity']
                elif 'head_entity' in ins.keys():
                    subject_entity = ins['head_entity']
                else:
                    subject_entity = ins['h']['name'].strip()
                    subject_entity = re.sub(r'\s*(\W+)\s*', r'\1', subject_entity)
                if 'object_entity' in ins.keys():
                    object_entity = ins['object_entity']
                elif 'tail_entity' in ins.keys():
                    object_entity = ins['tail_entity']
                else:
                    object_entity = ins['t']['name'].strip()
                    object_entity = re.sub(r'\s*(\W+)\s*', r'\1', object_entity)

                if 'subject_pattern' in ins.keys():
                    subject_pattern = ins['subject_pattern']
                elif 'head_entity_type' in ins.keys():
                    subject_pattern = ins['head_entity_type']
                else:
                    subject_pattern = ""

                if 'object_pattern' in ins.keys():
                    object_pattern = ins['object_pattern']
                elif 'object_entity_type' in ins.keys():
                    object_pattern = ins['object_entity_type']
                else:
                    object_pattern = ""

                if 'rel_pattern' in ins.keys():
                    rel_pattern = ins['rel_pattern']
                elif 'pred_rel_pattern' in ins.keys():
                    rel_pattern = ins['pred_rel_pattern']
                else:
                    rel_pattern = ""

                res.append(Sample(
                    sentence=sentence,
                    sentence_ent_tag=sentence_ent_tag,
                    subject_entity=subject_entity,
                    object_entity=object_entity,
                    relation=ins['relation'] if 'relation' in ins.keys() else "",
                    subject_pattern=subject_pattern,
                    object_pattern=object_pattern,
                    rel_pattern=rel_pattern,
                    token=tokens,
                    h=h,
                    t=t,
                ))
        return cls(samples=res)

    def save(self, path: str):
        Path(path).parent.mkdir(exist_ok=True, parents=True)
        with open(path, "w") as f:
            for s in self.samples:
                f.write(s.json() + "\n")

    @classmethod
    def decode_rules(cls, rules: List) -> List:
        res = []
        for rule in rules:
            res.append(Sample.decode(rule))
        return res
