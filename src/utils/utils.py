import copy
import json
import os
import shutil
import random
import torch
from typing import Tuple
from pathlib import Path
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from tqdm import tqdm


def read_data_file(file_path: str, max_count: int = None) -> Tuple[list, dict]:
    all_data, data_by_rel = [], {}
    with open(file_path, "r", encoding='utf-8') as reader:
        all_lines = reader.readlines()
        for line in all_lines:
            if max_count is not None and len(all_data) >= max_count:
                break
            if file_path[-4:] == '.txt':
                ins = eval(line)
            elif file_path[-5:] == '.json':
                ins = json.loads(line)
            else:
                raise Exception(f'The data format of file {file_path} should be .txt or .json')
            all_data.append(ins)

    # split real data by relation type
    for data in all_data:
        rel = data['relation']
        if rel not in data_by_rel:
            data_by_rel[rel] = [data]
        else:
            data_by_rel[rel].append(data)
    return all_data, data_by_rel


def remove_ent_tag(text):
    """ remove all entity tags """
    for tag in ['<sub>', '</sub>', '<obj>', '</obj>']:
        while tag in text.lower():
            left = text.lower().index(tag)
            right = left + len(tag)
            text = text[:left] + text[right:]
    return text


def remove_dot_end_symbol(text):
    """ remove '.' symbol """
    if len(text) > 0 and text[-1] == '.':
        text = text[:-1].strip()
    return text


def remove_square_bracket_symbol(text):
    """ remove [ ] symbol """
    text = text.replace('[', '')
    text = text.replace(']', '')
    return text


def find_word_in_text(word=None, text=None):
    last_symbol = text[-1]
    text = ' ' + text + ' '
    left, right = None, None
    if ' ' + word + ' ' in text:
        left = text.index(' ' + word + ' ')
        right = left + len(' ' + word + ' ')
    elif '\'' + word + '\'' in text:
        left = text.index('\'' + word + '\'')
        right = left + len('\'' + word + '\'')
    elif ' ' + word.capitalize() + ' ' in text:
        left = text.index(' ' + word.capitalize() + ' ')
        right = left + len(' ' + word.capitalize() + ' ')
    elif ' ' + word + last_symbol + ' ' in text:
        left = text.index(' ' + word + last_symbol + ' ')
        right = left + len(' ' + word + last_symbol + ' ')
    # elif word.lower() in text.lower():
    #     left = text.lower().index(word.lower())
    #     right = left + len(word.lower())
    return (left, right), text


def convert_ptb_token(token):
    """ Convert PTB tokens to normal tokens """
    if (token.lower() == '-lrb-'):
        return '('
    elif (token.lower() == '-rrb-'):
        return ')'
    elif (token.lower() == '-lsb-'):
        return '['
    elif (token.lower() == '-rsb-'):
        return ']'
    elif (token.lower() == '-lcb-'):
        return '{'
    elif (token.lower() == '-rcb-'):
        return '}'
    return token


def convert_label2words(labelname: str):
    if 'e1,e2' in labelname or 'e2,e1' in labelname:
        if '(e1,e2)' in labelname:
            labelname = labelname.lower().replace('(e1,e2)', "")
            e1 = labelname.split('-')[0]
            e2 = labelname.split('-')[1]
            return e1 + '(subject entity)-' + e2 + '(object entity)'
        elif '(e2,e1)' in labelname:
            labelname = labelname.lower().replace('(e2,e1)', "")
            e1 = labelname.split('-')[1]
            e2 = labelname.split('-')[0]
            return e1 + '(subject entity)-' + e2 + '(object entity)'
        else:
            return labelname.lower()
    else:
        return (labelname.lower().replace("_", " ").
                replace("-", " ").
                replace("per", "person").
                replace("org", "organization").
                replace("stateor", "state or "))


def delete_checkpoints(
        folder: str = ".", pattern="**/checkpoint*", delete: bool = True
):
    for p in Path(folder).glob(pattern):
        if (p.parent / "config.json").exists():
            print(p)
            if delete:
                if p.is_dir():
                    shutil.rmtree(p)
                elif p.is_file():
                    os.remove(p)
                else:
                    raise ValueError("Unknown Type")


def compute_cos_sim(first_tensor: torch.Tensor, second_tensor: torch.Tensor) -> torch.Tensor:
    # compute cosine similarity
    bsz = 256 if torch.cuda.is_available() else 16
    cosi = torch.nn.CosineSimilarity(dim=2)
    with torch.no_grad():
        output = []
        if first_tensor.shape[0] <= bsz:
            output.append(cosi(first_tensor.unsqueeze(1), second_tensor.unsqueeze(0)))
        else:
            for i in tqdm(range(0, first_tensor.shape[0], bsz), desc='Computing rule matching scores'):
                output.append(cosi(first_tensor[i:i + bsz].unsqueeze(1), second_tensor.unsqueeze(0)))
    return torch.concat(output, dim=0)


def evaluation(ground_truth, pred_result, rel2id, logger=None):
    correct = 0
    total = len(ground_truth)
    correct_positive = 0
    pred_positive = 0
    gold_positive = 0
    neg = -1
    for name in ['NA', 'na', 'no_relation', 'Other', 'Others', 'false', 'unanswerable']:
        if name in rel2id:
            neg = rel2id[name]
            break
    for i in range(total):
        golden = ground_truth[i]
        if golden == pred_result[i]:
            correct += 1
            if golden != neg:
                correct_positive += 1
        if golden != neg:
            gold_positive += 1
        if pred_result[i] != neg:
            pred_positive += 1
    if total == 0:
        result = {'acc': 0,
                  'micro_p': 0, 'micro_r': 0, 'micro_f1': 0,
                  'macro_p': 0, 'macro_r': 0, 'macro_f1': 0,
                  'f1_per_relation': 0,
                  }
        return result
    acc = float(correct) / float(total)

    y_true = ground_truth
    y_pred = pred_result
    pred_labels = list(set(y_true))
    poslabels = []
    if neg in y_true:
        poslabels = copy.deepcopy(pred_labels)
        poslabels.remove(neg)
    else:
        poslabels = copy.deepcopy(pred_labels)
    poslabels.sort()
    alllabels = list(rel2id.values())
    alllabels.sort()
    micro_pre = precision_score(y_true, y_pred, labels=poslabels, average='micro')
    micro_recall = recall_score(y_true, y_pred, labels=poslabels, average='micro')
    micro_f1 = f1_score(y_true=y_true, y_pred=y_pred, labels=poslabels, average='micro', zero_division=0)
    macro_pre = precision_score(y_true, y_pred, labels=poslabels, average='macro')
    macro_recall = recall_score(y_true, y_pred, labels=poslabels, average='macro')
    macro_f1 = f1_score(y_true=y_true, y_pred=y_pred, labels=poslabels, average='macro', zero_division=0)
    f1_per_relation = list(f1_score(y_true=y_true, y_pred=y_pred, labels=alllabels, average=None, zero_division=0))
    id2rel = {}
    for rel in rel2id.keys():
        id = rel2id[rel]
        id2rel[id] = rel
    report = classification_report([id2rel[res] for res in y_true], [id2rel[res] for res in y_pred])

    result = {'acc': acc,
              'micro_p': micro_pre, 'micro_r': micro_recall, 'micro_f1': micro_f1,
              'macro_p': macro_pre, 'macro_r': macro_recall, 'macro_f1': macro_f1,
              }

    if logger is not None:
        logger.info('Evaluation results:')
        logger.info('Micro F1: {}'.format(result['micro_f1']))
        logger.info('Micro recall: {}'.format(result['micro_r']))
        logger.info('Micro precision: {}'.format(result['micro_p']))

        logger.info('Macro F1: {}'.format(result['macro_f1']))
        logger.info('Macro recall: {}'.format(result['macro_r']))
        logger.info('Macro precision: {}'.format(result['macro_p']))

        logger.info('Accuracy: {}'.format(result['acc']))

        logger.info('F1 per Relation: {}'.format(f1_per_relation))
        logger.info('Classification Report: {}'.format(report))

    return result


def slm_pred_file_evaluation(test_file, rel2id_file_path, logger):
    rel2id = json.load(open(rel2id_file_path))
    # Label words, replacing abbreviated words
    rel2labelword = {}
    for rel in rel2id.keys():
        rel2labelword[rel] = convert_label2words(rel)
    NOTA_LABEL_ID = -1
    for name in ['NA', 'na', 'no_relation', 'Other', 'Others', 'other', 'false', 'unanswerable']:
        if name in rel2id:
            NOTA_LABEL_ID = rel2id[name]
            break
    labelword2rel = {}
    for k, v in rel2labelword.items():
        labelword2rel[v] = k
    all_data = []
    logger.info(f"Evaluating file {test_file}")
    with open(test_file, "r", encoding='utf-8') as reader:
        all_lines = reader.readlines()
        for line in all_lines:
            ins = json.loads(line)
            all_data.append(ins)
    true_rel_ids = []
    for data in all_data:
        true_rel_ids.append(rel2id[data['relation']])
    pred_rel_ids = []
    for data in all_data:
        if data['pred_relation']:
            pred_rel_id = NOTA_LABEL_ID  # no matched label, using NOTA label
            for labelword in labelword2rel.keys():
                if labelword in data['pred_relation']:
                    pred_rel_id = rel2id[labelword2rel[labelword]]
                    break
        else:  # if the output is not correctly verified, then using a random label
            pred_rel_id = rel2id[random.sample(rel2id.keys(), k=1)[0]]
        pred_rel_ids.append(pred_rel_id)
    evaluation(true_rel_ids, pred_rel_ids, rel2id, logger)


def passed_data_evaluation(passed_unlabeled_data, logger, rel2id):
    evaluation(ground_truth=[rel2id[data['gold_relation']] for data in passed_unlabeled_data],
               pred_result=[rel2id[data['relation']] for data in passed_unlabeled_data],
               rel2id=rel2id,
               logger=logger)
    total = 0.0
    error = 0.0
    for data in passed_unlabeled_data:
        if 'gold_relation' in data.keys():
            total += 1
            if data['gold_relation'] != data['relation']:
                error += 1
    if total != 0:
        logger.info(f'Total {total}, error {error}, error ratio {error / total}')
        print(f'Total {total}, error {error}, error ratio {error / total}')