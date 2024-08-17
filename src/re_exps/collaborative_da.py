import json
import re
import textwrap

import torch.cuda
from tqdm import tqdm
import os
import random
import time

from src.utils.openai_utils import call_openai_engine, get_openai_generation, estimate_cost
from src.utils.utils import *
from src.slm.t5_model import T5Model, Dataset
from src.slm.slm_finetune import slm_args_parser, post_processing_args, slm_finetune

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def new_passed_data(unlabeled_sample=None, prompt_kwargs=None) -> dict:
    new_data = {
        'relation': prompt_kwargs['pred_relation'],
        'subject_pattern': prompt_kwargs['subject_pattern'],
        'object_pattern': prompt_kwargs['object_pattern'],
        'rel_pattern': prompt_kwargs['rel_pattern'],
        'subject_entity': prompt_kwargs['subject_entity'],
        'object_entity': prompt_kwargs['object_entity'],
        'sentence': prompt_kwargs['sentence'],
        'token': unlabeled_sample['token'],
        'h': unlabeled_sample['h'],
        't': unlabeled_sample['t'],
        'sentence_ent_tag': prompt_kwargs['sentence_ent_tag'],
        'gold_relation': unlabeled_sample['relation'],
    }
    return new_data


def collaborative_da(args, logger):
    """ extract relations using rules induced from labeled and unlabeled data  """
    with open(f"src/configs/data_config.json", "r", encoding="utf-8") as f:
        data_configs = json.load(f)
        rel2id_file_path = data_configs[args.dataset]['rel2id']
        test_file_path = data_configs[args.dataset]['test']  # test file
        if args.lre_setting == 'k-shot':
            # load k-shot train file
            data_set_path = data_configs[args.dataset]['k-shot'] + \
                            f'{args.k_shot_lre}-{args.k_shot_id}/{args.split}.json'
            # load k-shot unlabeled file
            unlabeled_file_path = data_configs[args.dataset]['k-shot'] + \
                                  f'{args.k_shot_lre}-{args.k_shot_id}/unlabeled_train.json'
        else:
            raise ValueError('Unknown lre_setting hyperparameter!')

    rel2id = json.load(open(rel2id_file_path))
    id2rel = {v: k for k, v in rel2id.items()}
    NOTA_LABEL_ID = -1
    for name in ['NA', 'na', 'no_relation', 'Other', 'Others', 'other', 'false', 'unanswerable']:
        if name in rel2id:
            NOTA_LABEL_ID = rel2id[name]
            break

    # Label words, replacing abbreviated words
    rel2labelword = {}
    for rel in rel2id.keys():
        # for semeval dataset
        if args.dataset == 'semeval':
            if '(e1,e2)' in rel:
                rel2labelword[rel] = rel.lower().replace('(e1,e2)', "")
                e1 = rel2labelword[rel].split('-')[0]
                e2 = rel2labelword[rel].split('-')[1]
                rel2labelword[rel] = e1 + '(subject entity)-' + e2 + '(object entity)'
            elif '(e2,e1)' in rel:
                rel2labelword[rel] = rel.lower().replace('(e2,e1)', "")
                e1 = rel2labelword[rel].split('-')[1]
                e2 = rel2labelword[rel].split('-')[0]
                rel2labelword[rel] = e1 + '(subject entity)-' + e2 + '(object entity)'
            else:
                rel2labelword[rel] = rel.lower()
        else:
            if rel.lower() == 'per:identity':
                rel2labelword[rel] = (rel.lower().replace("per:identity", "per:same_person").
                                      replace("_", " ").
                                      replace("-", " ").
                                      replace("per", "person").
                                      replace("org", "organization").
                                      replace("stateor", "state or "))
            else:
                rel2labelword[rel] = (rel.lower().replace("_", " ").
                                      replace("-", " ").
                                      replace("per", "person").
                                      replace("org", "organization").
                                      replace("stateor", "state or "))
    # rel2labelword[id2rel[NOTA_LABEL_ID]] = 'no relation'
    labelword2rel = {}
    for k, v in rel2labelword.items():
        labelword2rel[v] = k

    random.seed(args.random_seed)  # fix random results

    if args.lre_setting == 'k-shot':
        output_dir = os.path.join(args.output_dir, f"{args.dataset}/k-shot/{args.k_shot_lre}-{args.k_shot_id}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        os.makedirs(os.path.join(output_dir, 'pr'))
        os.makedirs(os.path.join(output_dir, 'track'))
        os.makedirs(os.path.join(output_dir, 'slm'))

    # the data file tracking prompt and response
    prompt_track_file = os.path.join(output_dir, f'track/{args.exp_name}_prompt.json')
    prompt_track_file_handler = open(prompt_track_file, 'a')
    pr_file = os.path.join(output_dir, f'pr/{args.exp_name}_pr.json')
    pr_file_handler = open(pr_file, 'a')

    # load previous induced premises from labeled training samples
    with open(f"src/configs/rule_train_config.json", "r", encoding="utf-8") as f:
        train_rule_files = json.load(f)
    train_rule_dir = f'{output_dir}/pr/'
    if args.lre_setting == 'k-shot':
        train_rule_path = train_rule_dir + train_rule_files[args.dataset]['k-shot'][
            f'{args.k_shot_lre}-{args.k_shot_id}']
    labeled_data_w_rule, labeled_data_w_rule_by_rel = read_data_file(train_rule_path,
                                                                     max_count=args.debug_test_num if args.mode == 'debug' else None)
    logger.info(f'Loading rules from the file: {train_rule_path}, {len(labeled_data_w_rule)} induced rules.')


    for data in labeled_data_w_rule:
        # save the labeled data
        pr_file_handler.writelines(json.dumps({**data}, ensure_ascii=False))
        pr_file_handler.write('\n')
    pr_file_handler.flush()

    # -------------- Loading small pretrained model --------------
    with open(f"src/configs/slm_ckpt_config.json", "r", encoding="utf-8") as f:
        slm_save_dirs = json.load(f)
    if args.lre_setting == 'k-shot':
        slm_save_dir = f"src/slm/ckpt/{args.dataset}/k-shot/{args.k_shot_lre}-{args.k_shot_id}/" + \
                       slm_save_dirs[args.dataset]['k-shot'][f'{args.k_shot_lre}-{args.k_shot_id}']

    model = T5Model(resume_from_save_dir=True, save_dir=slm_save_dir,
                    batch_size=64 if torch.cuda.is_available() else 16)

    if args.lre_setting == 'k-shot':
        unlabeled_slm_pred_path = f"{output_dir}/slm/" + f'{slm_save_dirs}_rule_match.json'

    logger.info(f"Reading unlabeled data with SLM predicted rules from {unlabeled_slm_pred_path}")
    unlabeled_data_w_rules, _ = read_data_file(unlabeled_slm_pred_path,
                                               max_count=args.debug_test_num - 1 if args.mode == 'debug' else None)

    # obtain rule embeddings of labeled data
    def rule_verbalization(ins):
        return f"Subject entity type: {ins['subject_pattern']}. Object entity type: {ins['object_pattern']}. " + \
            f"Relationship: subject entity {ins['pred_rel_pattern']} object entity."

    match_degree_method = 'PATTERN2PATTERN'
    if match_degree_method == 'PREMISE2PERMISE':
        # obtain rule embeddings of unlabeled data
        unlabeled_rule_embs = model.get_sentence_embeddings(
            texts=[x['premise'] if x['premise'] is not None else '' for x in unlabeled_data_w_rules])
        labeled_rule_embs = model.get_sentence_embeddings(
            texts=[rule_verbalization(ins) for ins in labeled_data_w_rule])
        rule_sim_scores = compute_cos_sim(unlabeled_rule_embs, labeled_rule_embs)
    elif match_degree_method == 'PATTERN2PATTERN':
        labeled_rule2data = {}
        for ins in labeled_data_w_rule:
            premise = ins['subject_pattern'].lower() + ' ' + ins['rel_pattern'].lower() + ' ' + ins[
                'object_pattern'].lower()
            if premise not in labeled_rule2data:
                labeled_rule2data[premise] = [ins]
            else:
                labeled_rule2data[premise].append(ins)
        labeled_rule_subj_patterns = [labeled_rule2data[premise][0]['subject_pattern'].lower() for premise in
                                      labeled_rule2data.keys()]
        labeled_rule_obj_patterns = [labeled_rule2data[premise][0]['object_pattern'].lower() for premise in
                                     labeled_rule2data.keys()]
        labeled_rule_rel_patterns = [labeled_rule2data[premise][0]['rel_pattern'].lower() for premise in
                                     labeled_rule2data.keys()]
        labeled_rule_premise = [premise for premise in labeled_rule2data.keys()]
        labeled_rule_relation = [labeled_rule2data[premise][0]['relation'] for premise in labeled_rule2data.keys()]
        # # 1. SLM predicted patterns
        unlabeled_subj_pattern_embs = model.get_sentence_embeddings(
            texts=[x['subject_pattern'].lower() if x['subject_pattern'] is not None else '' for x in
                   unlabeled_data_w_rules])
        unlabeled_obj_pattern_embs = model.get_sentence_embeddings(
            texts=[x['object_pattern'].lower() if x['object_pattern'] is not None else '' for x in
                   unlabeled_data_w_rules])
        unlabeled_rel_pattern_embs = model.get_sentence_embeddings(
            texts=[x['rel_pattern'].lower() if x['rel_pattern'] is not None else '' for x in unlabeled_data_w_rules])
        # # 2. patterns for labeled data
        labeled_subj_pattern_embs = model.get_sentence_embeddings(
            texts=labeled_rule_subj_patterns)
        labeled_obj_pattern_embs = model.get_sentence_embeddings(
            texts=labeled_rule_obj_patterns)
        labeled_rel_pattern_embs = model.get_sentence_embeddings(
            texts=labeled_rule_rel_patterns)
        # # 3. unlabeled data-rule matching scores
        rule_sim_scores = 1.0 / 3.0 * (compute_cos_sim(unlabeled_subj_pattern_embs, labeled_subj_pattern_embs) +
                                       compute_cos_sim(unlabeled_obj_pattern_embs, labeled_obj_pattern_embs) +
                                       compute_cos_sim(unlabeled_rel_pattern_embs, labeled_rel_pattern_embs))
    elif match_degree_method == 'SEN2SEN':
        labeled_sens = [x['sentence_ent_tag'] for x in labeled_data_w_rule]
        labeled_sen_embs = model.get_sentence_embeddings(texts=labeled_sens)
        unlabeled_sens = [x['sentence_ent_tag'] for x in unlabeled_data_w_rules]
        unlabeled_sen_embs = model.get_sentence_embeddings(texts=unlabeled_sens)
        rule_sim_scores = compute_cos_sim(unlabeled_sen_embs, labeled_sen_embs)

    slm_rel2labelword = {}
    for rel in rel2id.keys():
        slm_rel2labelword[rel] = convert_label2words(rel)
    slm_labelword2rel = {}
    for k, v in slm_rel2labelword.items():
        slm_labelword2rel[v] = k

    for sample_id, unlabeled_sample in enumerate(unlabeled_data_w_rules):
        # post-processing SLM predicted relations
        slm_pred_relation = unlabeled_sample['pred_relation']

        pred_rel_id = NOTA_LABEL_ID
        if slm_pred_relation:
            # for labelword in labelword2rel.keys():
            for labelword in slm_labelword2rel.keys():
                if labelword in slm_pred_relation:
                    pred_rel_id = rel2id[slm_labelword2rel[labelword]]
                    break
            slm_pred_relation = id2rel[pred_rel_id]
        else:
            slm_pred_relation = None
        unlabeled_sample['pred_relation'] = slm_pred_relation
        unlabeled_sample['sim_scores'] = rule_sim_scores[sample_id]
        top_k_similar_rule_ids = torch.sort(rule_sim_scores[sample_id],
                                            descending=True).indices[:args.in_context_size].cpu().tolist()
        for i, rule_id in enumerate(top_k_similar_rule_ids):
            unlabeled_sample[f'top{i + 1}_matched_rule'] = {
                'premise': labeled_rule_premise[rule_id],
                'relation': labeled_rule_relation[rule_id],
                'confidence': rule_sim_scores[sample_id][rule_id].tolist(),
            }

    # using unlabeled data with valid patterns
    valid_unlabeled_data = list()
    counter = 0
    for data in unlabeled_data_w_rules:
        if data['subject_pattern'] is not None:
            data['sample_id'] = counter
            counter += 1
            valid_unlabeled_data.append(data)
    unlabeled_data = valid_unlabeled_data
    unlabeled_data_by_rel = {}
    for data in unlabeled_data:
        rel = data['relation']
        if rel not in unlabeled_data_by_rel:
            unlabeled_data_by_rel[rel] = [data]
        else:
            unlabeled_data_by_rel[rel].append(data)

    # start annotating unlabeled data by iteration
    max_iter = 10
    start_iter = 0
    if args.mode == 'rerun':
        unlabeled_slm_pred_path = f"{output_dir}/pr/{args.exp_name}_iters/{start_iter - 1}_unlabeled_rules.json"
        slm_finetune_args = f"--exp_name {args.exp_name}_iter{start_iter - 1} " + \
                            f"--fix_exp_name True " + \
                            f"--dataset {args.dataset} " + \
                            f"--lre_setting {args.lre_setting} " + \
                            f"--k_shot_lre {args.k_shot_lre} --k_shot_id {args.k_shot_id} " + \
                            f"--train_file {args.exp_name}_pr.json"
        slm_finetune_args = slm_args_parser().parse_args(slm_finetune_args.split())
        post_processing_args(slm_finetune_args)
        slm_save_dir = slm_finetune_args.model_save_dir + f"/{args.exp_name}_iter{start_iter - 1}"
        logger.info(f'-------------Rerun, loading previous trained SLM from {slm_save_dir} ---------------')
        model = T5Model(resume_from_save_dir=True,
                        save_dir=slm_save_dir,
                        batch_size=64 if torch.cuda.is_available() else 16)
    else:
        start_iter = 0
    all_passed_unlabeled_data = []
    iter_labeled_data_w_rule = labeled_data_w_rule
    for iteration_id in range(start_iter, max_iter):
        logger.info(f"------------------------------")
        logger.info(f"Iteration {iteration_id}")
        if iteration_id >= 1:
            unlabeled_data_w_rules, _ = read_data_file(unlabeled_slm_pred_path,
                                                       max_count=args.debug_test_num - 1 if args.mode == 'debug' else None)
            logger.info(f"Reading unlabeled data {len(unlabeled_data_w_rules)} with SLM predicted rules from {unlabeled_slm_pred_path}")
            if match_degree_method == 'PATTERN2PATTERN':
                iter_labeled_data_w_rule = labeled_data_w_rule + all_passed_unlabeled_data
                labeled_rule2data = {}
                for ins in iter_labeled_data_w_rule:
                    premise = ins['subject_pattern'].lower() + ' ' + ins['rel_pattern'].lower() + ' ' + ins['object_pattern'].lower()
                    if premise not in labeled_rule2data:
                        labeled_rule2data[premise] = [ins]
                    else:
                        labeled_rule2data[premise].append(ins)
                labeled_rule_subj_patterns = [labeled_rule2data[premise][0]['subject_pattern'].lower() for premise in
                                              labeled_rule2data.keys()]
                labeled_rule_obj_patterns = [labeled_rule2data[premise][0]['object_pattern'].lower() for premise in
                                             labeled_rule2data.keys()]
                labeled_rule_rel_patterns = [labeled_rule2data[premise][0]['rel_pattern'].lower() for premise in
                                             labeled_rule2data.keys()]
                labeled_rule_premise = [premise for premise in labeled_rule2data.keys()]
                labeled_rule_relation = [labeled_rule2data[premise][0]['relation'] for premise in
                                         labeled_rule2data.keys()]
                # # 1. SLM predicted patterns
                unlabeled_subj_pattern_embs = model.get_sentence_embeddings(
                    texts=[x['subject_pattern'].lower() if x['subject_pattern'] is not None else '' for x in
                           unlabeled_data_w_rules])
                unlabeled_obj_pattern_embs = model.get_sentence_embeddings(
                    texts=[x['object_pattern'].lower() if x['object_pattern'] is not None else '' for x in
                           unlabeled_data_w_rules])
                unlabeled_rel_pattern_embs = model.get_sentence_embeddings(
                    texts=[x['rel_pattern'].lower() if x['rel_pattern'] is not None else '' for x in
                           unlabeled_data_w_rules])
                # # 2. patterns for labeled data
                labeled_subj_pattern_embs = model.get_sentence_embeddings(
                    texts=labeled_rule_subj_patterns)
                labeled_obj_pattern_embs = model.get_sentence_embeddings(
                    texts=labeled_rule_obj_patterns)
                labeled_rel_pattern_embs = model.get_sentence_embeddings(
                    texts=labeled_rule_rel_patterns)
                # # 3. unlabeled data-rule matching scores
                rule_sim_scores = 1.0 / 3.0 * (compute_cos_sim(unlabeled_subj_pattern_embs, labeled_subj_pattern_embs) +
                                               compute_cos_sim(unlabeled_obj_pattern_embs, labeled_obj_pattern_embs) +
                                               compute_cos_sim(unlabeled_rel_pattern_embs, labeled_rel_pattern_embs))

                for sample_id, unlabeled_sample in enumerate(unlabeled_data_w_rules):
                    # post-processing SLM predicted relations
                    slm_pred_relation = unlabeled_sample['pred_relation']
                    pred_rel_id = NOTA_LABEL_ID
                    if slm_pred_relation:
                        # for labelword in rel2id.keys():
                        #     if labelword in slm_pred_relation:
                        #         pred_rel_id = rel2id[labelword]
                        #         break

                        # for labelword in labelword2rel.keys():
                        for labelword in slm_labelword2rel.keys():
                            if labelword in slm_pred_relation:
                                pred_rel_id = rel2id[slm_labelword2rel[labelword]]
                                break
                        slm_pred_relation = id2rel[pred_rel_id]
                    else:
                        slm_pred_relation = None
                    unlabeled_sample['pred_relation'] = slm_pred_relation
                    unlabeled_sample['sim_scores'] = rule_sim_scores[sample_id]
                    top_k_similar_rule_ids = torch.sort(rule_sim_scores[sample_id],
                                                        descending=True).indices[:args.in_context_size].cpu().tolist()
                    for i, rule_id in enumerate(top_k_similar_rule_ids):
                        unlabeled_sample[f'top{i + 1}_matched_rule'] = {
                            'premise': labeled_rule_premise[rule_id],
                            'relation': labeled_rule_relation[rule_id],
                            'confidence': rule_sim_scores[sample_id][rule_id].tolist(),
                        }
                # using unlabeled data with valid patterns
                valid_unlabeled_data = list()
                counter = 0
                for data in unlabeled_data_w_rules:
                    if data['subject_pattern'] is not None:
                        data['sample_id'] = counter
                        counter += 1
                        valid_unlabeled_data.append(data)
                unlabeled_data = valid_unlabeled_data
                unlabeled_data_by_rel = {}
                for data in unlabeled_data:
                    rel = data['relation']
                    if rel not in unlabeled_data_by_rel:
                        unlabeled_data_by_rel[rel] = [data]
                    else:
                        unlabeled_data_by_rel[rel].append(data)
            elif match_degree_method == 'SEN2SEN':
                iter_labeled_data_w_rule = labeled_data_w_rule + all_passed_unlabeled_data
                labeled_sens = [x['sentence_ent_tag'] for x in iter_labeled_data_w_rule]
                labeled_sen_embs = model.get_sentence_embeddings(texts=labeled_sens)
                unlabeled_sens = [x['sentence_ent_tag'] for x in unlabeled_data_w_rules]
                unlabeled_sen_embs = model.get_sentence_embeddings(texts=unlabeled_sens)
                rule_sim_scores = compute_cos_sim(unlabeled_sen_embs, labeled_sen_embs)
                for sample_id, unlabeled_sample in enumerate(unlabeled_data_w_rules):
                    # post-processing SLM predicted relations
                    slm_pred_relation = unlabeled_sample['pred_relation']
                    pred_rel_id = NOTA_LABEL_ID
                    if slm_pred_relation:
                        # for labelword in labelword2rel.keys():
                        for labelword in slm_labelword2rel.keys():
                            if labelword in slm_pred_relation:
                                pred_rel_id = rel2id[slm_labelword2rel[labelword]]
                                break
                        slm_pred_relation = id2rel[pred_rel_id]
                    else:
                        slm_pred_relation = None
                    unlabeled_sample['pred_relation'] = slm_pred_relation
                    unlabeled_sample['sim_scores'] = rule_sim_scores[sample_id]
                    top_k_similar_rule_ids = torch.sort(rule_sim_scores[sample_id],
                                                        descending=True).indices[:args.in_context_size].cpu().tolist()
                    for i, rule_id in enumerate(top_k_similar_rule_ids):
                        unlabeled_sample[f'top{i + 1}_matched_rule'] = {
                            'confidence': rule_sim_scores[sample_id][rule_id].tolist(),
                        }
                # using unlabeled data with valid patterns
                valid_unlabeled_data = list()
                counter = 0
                for data in unlabeled_data_w_rules:
                    if data['pred_relation'] is not None:
                        data['sample_id'] = counter
                        counter += 1
                        valid_unlabeled_data.append(data)
                unlabeled_data = valid_unlabeled_data
                unlabeled_data_by_rel = {}
                for data in unlabeled_data:
                    rel = data['relation']
                    if rel not in unlabeled_data_by_rel:
                        unlabeled_data_by_rel[rel] = [data]
                    else:
                        unlabeled_data_by_rel[rel].append(data)

        # save new passed unlabeled data in each iteration
        iter_new_passed_data_path = f"{output_dir}/pr/{args.exp_name}_iters/{iteration_id}_new_passed.json"
        Path(iter_new_passed_data_path).parent.mkdir(exist_ok=True, parents=True)
        iter_new_passed_save_handler = open(iter_new_passed_data_path, 'a')
        # save accumulated all passed unlabeled data in each iteration
        iter_all_passed_data_path = f"{output_dir}/pr/{args.exp_name}_iters/{iteration_id}_all_passed.json"
        Path(iter_all_passed_data_path).parent.mkdir(exist_ok=True, parents=True)
        iter_all_passed_save_handler = open(iter_all_passed_data_path, 'a')
        for data in all_passed_unlabeled_data:
            iter_all_passed_save_handler.writelines(json.dumps({**data}, ensure_ascii=False))
            iter_all_passed_save_handler.write('\n')
        # -------------- Querying the LLM --------------
        total_cost = 0  # total_cost is used to estimate money cost
        start_time = time.time()
        high_conf_passed_limit = 200
        low_conf_passed_limit = 200
        pbar = tqdm(total=high_conf_passed_limit + low_conf_passed_limit,
                    desc=f'Choosing {high_conf_passed_limit + low_conf_passed_limit} unlabeled data')

        ##############
        # high confidence unlabeled data
        # using SLM label confidence
        unlabeled_data.sort(key=lambda x: x['confidence']['label_confidence'], reverse=True)

        passed_unlabeled_samples = []
        iter_passed_unlabeled_data = []
        passed_counter_by_rel = {}

        # for _, unlabeled_sample in enumerate(unlabeled_data):
        for _, unlabeled_sample in enumerate(unlabeled_data[:high_conf_passed_limit]):
            if len(passed_unlabeled_samples) >= high_conf_passed_limit:
                break
            prompt_kwargs = {'sentence': unlabeled_sample['sentence'] if 'sentence' in unlabeled_sample.keys() else ' '.join([convert_ptb_token(token) for token in unlabeled_sample['token']]),
                             'subject_entity': unlabeled_sample['subject_entity'] if 'subject_entity' in unlabeled_sample.keys() else unlabeled_sample['h']['name'],
                             'object_entity': unlabeled_sample['object_entity'] if 'object_entity' in unlabeled_sample.keys() else unlabeled_sample['t']['name'],
                             'subject_pattern': unlabeled_sample['subject_pattern'],
                             'object_pattern': unlabeled_sample['object_pattern'],
                             'rel_pattern': unlabeled_sample['rel_pattern'],
                             'pred_relation': unlabeled_sample['pred_relation'],
                             }

            if prompt_kwargs['subject_pattern'] is None:
                logger.warning(
                    f"\n{unlabeled_sample['slm_pred_rule']} is not correctly post processed!")
                pbar.update(1)
                continue

            if prompt_kwargs['pred_relation'] is None:
                logger.warning(
                    f"\nFail to obtain labels from {unlabeled_sample['slm_pred_rule']}!")
                pbar.update(1)
                continue

            if 'sentence_ent_tag' in unlabeled_sample.keys():
                prompt_kwargs['sentence_ent_tag'] = unlabeled_sample['sentence_ent_tag']
            else:
                tokens_ent_tag = []
                for token_index, token in enumerate(unlabeled_sample['token']):
                    if token_index == unlabeled_sample['h']['pos'][0]:
                        tokens_ent_tag.append('<Sub>')
                    if token_index == unlabeled_sample['h']['pos'][1]:
                        tokens_ent_tag.append('</Sub>')
                    if token_index == unlabeled_sample['t']['pos'][0]:
                        tokens_ent_tag.append('<Obj>')
                    if token_index == unlabeled_sample['t']['pos'][1]:
                        tokens_ent_tag.append('</Obj>')
                    tokens_ent_tag.append(convert_ptb_token(token))
                prompt_kwargs['sentence_ent_tag'] = ' '.join(tokens_ent_tag)
            # remove possible space between character and symbol
            prompt_kwargs['subject_entity'] = prompt_kwargs['subject_entity'].strip()
            prompt_kwargs['subject_entity'] = re.sub(r'\s*(\W+)\s*', r'\1', prompt_kwargs['subject_entity'])
            prompt_kwargs['object_entity'] = prompt_kwargs['object_entity'].strip()
            prompt_kwargs['object_entity'] = re.sub(r'\s*(\W+)\s*', r'\1', prompt_kwargs['object_entity'])

            verify_method = 'LLM_GEN'
            if verify_method == 'LLM_GEN':

                top_k_similar_rule_ids = torch.sort(unlabeled_sample['sim_scores'],
                                                    descending=True).indices[:args.in_context_size].cpu().tolist()
                llm_gen_success = False
                while not llm_gen_success:
                    verify_prem_prompt = ''
                    demos = []
                    for rule_id in top_k_similar_rule_ids:
                        premise = labeled_rule_subj_patterns[rule_id] + ' ' + \
                                  labeled_rule_rel_patterns[rule_id] + ' ' + \
                                  labeled_rule_obj_patterns[rule_id]
                        if premise in labeled_rule2data:
                            demos.append(random.sample(labeled_rule2data[premise], k=1)[0])
                            # demos.extend(labeled_rule2data[premise])
                    demos = demos[:args.in_context_size]
                    demo_labels = [data['relation'] for data in demos]
                    if prompt_kwargs['pred_relation'] not in demo_labels:
                        if prompt_kwargs['pred_relation'] in labeled_data_w_rule_by_rel.keys():
                            demos.append(random.sample(labeled_data_w_rule_by_rel[prompt_kwargs['pred_relation']], k=1)[0])
                    ############
                    # # LLM generates premise for unlabeled data
                    for id, data in enumerate(demos):
                        verify_prem_prompt += f"Sentence: {data['sentence_ent_tag']} Subject entity: {data['subject_entity']}. Object entity: {data['object_entity']}.\n" + \
                                              f"1. The type of {data['subject_entity']} is {data['subject_pattern']}.\n" + \
                                              f"2. The type of {data['object_entity']} is {data['object_pattern']}.\n" + \
                                              f"3. {data['rel_pattern'].replace('subject entity', data['subject_entity']).replace('object entity', data['object_entity'])}.\n"
                    query = '''\
                            Sentence: {sentence_ent_tag} Subject entity: {subject_entity}. Object entity: {object_entity}.
                            '''.format(**prompt_kwargs)
                    query = textwrap.dedent(query)
                    query += f"Given that the sentence, along with subject entity '{prompt_kwargs['subject_entity']}' (enclosed by <Sub></Sub>) and object entity '{prompt_kwargs['object_entity']}' (enclosed by <Obj></Obj>), judge the types of subject and object entities, as well as the relationship between these two entities.\n"
                    query += f"The responses should adhere to the subsequent format without any supplementary information, explanations, or notes:\n" + \
                             f"1. The type of {prompt_kwargs['subject_entity']} is [Entity Type1]\n" + \
                             f"2. The type of {prompt_kwargs['object_entity']} is [Entity Type2]\n" + \
                             f"3. [Relationship Description between {prompt_kwargs['subject_entity']} and {prompt_kwargs['object_entity']}].\n"
                    query += "Note that the subject entity and object entity may possess a specific relationship or no relationship, which can be inferred from the provided sentence."
                    verify_prem_prompt += query

                    # call openai
                    response = call_openai_engine(engine=args.engine,
                                                  api_key=args.api_key,
                                                  prompt=verify_prem_prompt,
                                                  temperature=0.0)
                    openai_generation, usage = get_openai_generation(args.engine, response)
                    new_cost = estimate_cost(usage, args.engine)
                    total_cost += new_cost

                    prompt_track_file_handler.writelines(json.dumps({'prompt_type': 'verify_prem_prompt',
                                                                     'prompt': verify_prem_prompt,
                                                                     'response': openai_generation,
                                                                     # **prompt_kwargs,
                                                                     **{k: unlabeled_sample[k] for k in
                                                                        unlabeled_sample.keys() - {'sim_scores'}},
                                                                     }, ensure_ascii=False))
                    prompt_track_file_handler.write('\n')

                    try:
                        ############
                        # post-processing prompt
                        prem_valid = False
                        res = openai_generation.strip()
                        llm_gen_success = True
                        if prompt_kwargs['subject_entity'].lower() in openai_generation.lower() and \
                                prompt_kwargs['object_entity'].lower() in openai_generation.lower():
                            res = res[res.index('1.'):]
                            left = res.index(f"{prompt_kwargs['subject_entity']} is") + len(
                                f"{prompt_kwargs['subject_entity']} is")
                            right = res.index('2.')
                            subject_pattern = remove_square_bracket_symbol(
                                remove_dot_end_symbol(res[left:right].strip()))
                            if 'type ' in subject_pattern:  # remove possible redundant 'type' word
                                subject_pattern = subject_pattern.replace('type ', '').strip()
                            if ' type' in subject_pattern:  # remove possible redundant 'type' word
                                subject_pattern = subject_pattern.replace(' type', '').strip()
                            if '\"' in subject_pattern:
                                subject_pattern = subject_pattern.replace('\"', '').strip()
                            if '\'' in subject_pattern:
                                subject_pattern = subject_pattern.replace('\'', '').strip()
                            prompt_kwargs['subject_pattern'] = subject_pattern.strip()

                            res = res[res.index('2.'):]
                            left = res.index(f"{prompt_kwargs['object_entity']} is") + len(
                                f"{prompt_kwargs['object_entity']} is")
                            right = res.index('3.')
                            object_pattern = remove_square_bracket_symbol(
                                remove_dot_end_symbol(res[left:right].strip()))
                            if 'type ' in object_pattern:  # remove possible redundant 'type' word
                                object_pattern = object_pattern.replace('type ', '').strip()
                            if ' type' in object_pattern:  # remove possible redundant 'type' word
                                object_pattern = object_pattern.replace(' type', '').strip()
                            if '\"' in object_pattern:
                                object_pattern = object_pattern.replace('\"', '').strip()
                            if '\'' in object_pattern:
                                object_pattern = object_pattern.replace('\'', '').strip()
                            prompt_kwargs['object_pattern'] = object_pattern.strip()

                            # rel_pattern
                            pred_rel_pattern = remove_square_bracket_symbol(res[res.index('3.') + len('3.'):].strip())
                            # Find possible subject entity in relation description
                            (left, right), pred_rel_pattern = find_word_in_text(prompt_kwargs['subject_entity'],
                                                                                pred_rel_pattern)
                            if left is not None:
                                pred_rel_pattern = pred_rel_pattern[:left] + f' subject entity ' + pred_rel_pattern[
                                                                                                   right:]
                            # Find possible object entity in relation description
                            (left, right), pred_rel_pattern = find_word_in_text(prompt_kwargs['object_entity'],
                                                                                pred_rel_pattern.strip())
                            if left is not None:
                                pred_rel_pattern = pred_rel_pattern[:left] + f' object entity ' + pred_rel_pattern[
                                                                                                  right:]
                            if 'subject entity' not in pred_rel_pattern:
                                pred_rel_pattern = 'subject entity ' + pred_rel_pattern
                            if 'object entity' not in pred_rel_pattern:
                                pred_rel_pattern = pred_rel_pattern + ' object entity'

                            if '  ' in pred_rel_pattern:
                                pred_rel_pattern = pred_rel_pattern.replace('  ', ' ')
                            prompt_kwargs['rel_pattern'] = pred_rel_pattern.strip()
                            llm_gen_success = True
                        prem_valid = True
                    except:
                        logger.warning(
                            f'\nFail to parse the response for prompt:\n{verify_prem_prompt}\nThe response:\n{openai_generation}')
                        continue
                if not prem_valid:
                    pbar.update(1)
                    continue
                # ------------ judge rule label ------------
                llm_gen_success = False
                con_valid = False
                while not llm_gen_success:
                    # 1. LLM predicted patterns
                    unlabeled_subj_pattern_emb = model.get_sentence_embeddings(
                        texts=[prompt_kwargs['subject_pattern'].lower()])
                    unlabeled_obj_pattern_emb = model.get_sentence_embeddings(
                        texts=[prompt_kwargs['object_pattern'].lower()])
                    unlabeled_rel_pattern_emb = model.get_sentence_embeddings(
                        texts=[prompt_kwargs['rel_pattern'].lower()])
                    # # 2. unlabeled data-rule matching scores
                    new_sim_scores = 1.0 / 3.0 * (
                            compute_cos_sim(unlabeled_subj_pattern_emb, labeled_subj_pattern_embs) +
                            compute_cos_sim(unlabeled_obj_pattern_emb, labeled_obj_pattern_embs) +
                            compute_cos_sim(unlabeled_rel_pattern_emb, labeled_rel_pattern_embs))
                    top_k_similar_rule_ids = torch.sort(new_sim_scores[0],
                                                        descending=True).indices[:args.in_context_size].cpu().tolist()
                    demos = []
                    for rule_id in top_k_similar_rule_ids:
                        premise = labeled_rule_subj_patterns[rule_id] + ' ' + \
                                  labeled_rule_rel_patterns[rule_id] + ' ' + \
                                  labeled_rule_obj_patterns[rule_id]
                        if premise in labeled_rule2data:
                            demos.append(random.sample(labeled_rule2data[premise], k=1)[0])
                            # demos.extend(labeled_rule2data[premise])
                    demos = demos[:args.in_context_size]
                    demo_labels = [data['relation'] for data in demos]
                    if prompt_kwargs['pred_relation'] not in demo_labels:
                        if prompt_kwargs['pred_relation'] in labeled_data_w_rule_by_rel.keys():
                            demos.append(random.sample(labeled_data_w_rule_by_rel[prompt_kwargs['pred_relation']], k=1)[0])
                    # judge all labels
                    verify_label_prompt = f'''\
                                          Given a sentence, a pair of subject (enclosed by <Sub></Sub>) and object entities (enclosed by <Obj></Obj>) in the sentence,decide the most precise relationship between the subject and object entities. If not sure, choose label '{rel2labelword[id2rel[NOTA_LABEL_ID]]}'.
                                          Note that the relationship must be one of the defined relation from candidate relations:
                                          {', '.join(labelword2rel.keys())}.
                                          Provide the relationship label without any supplementary information, explanations, or notes.\n
                                          '''
                    verify_label_prompt = textwrap.dedent(verify_label_prompt)

                    ##############
                    # p -> r
                    verify_label_prompt += "Some labeling rules includes:\n"
                    for id, data in enumerate(demos):
                        verify_label_prompt += f"{id + 1}. If the type of subject entity is {data['subject_pattern']}, " + \
                                               f"the type of object entity is {data['object_pattern']} " + \
                                               f"and {data['rel_pattern']}, " + \
                                               f"then the relation is {rel2labelword[data['relation']]}.\n"
                    verify_label_prompt += f"{len(demos) + 1}. If subject entity has no relationship with object entity, then the relation is {rel2labelword[id2rel[NOTA_LABEL_ID]]}.\n"
                    verify_label_prompt += f"{len(demos) + 2}. If none of the above relation labels can be valid or correct between subject entity and object entity, then the relation is {rel2labelword[id2rel[NOTA_LABEL_ID]]}.\n"


                    query = f"Sentence: {prompt_kwargs['sentence_ent_tag']} Subject entity: {prompt_kwargs['subject_entity']}. Object entity: {prompt_kwargs['object_entity']}. " + \
                            f"We can infer that the type of {prompt_kwargs['subject_entity']} is {prompt_kwargs['subject_pattern']}, the type of {prompt_kwargs['object_entity']} is {prompt_kwargs['object_pattern']}, and {prompt_kwargs['rel_pattern'].replace('subject entity', prompt_kwargs['subject_entity']).replace('object entity', prompt_kwargs['object_entity'])}. " + \
                            f"The relation between {prompt_kwargs['subject_entity']} and {prompt_kwargs['object_entity']} in the sentence is"
                    verify_label_prompt += query

                    # call openai
                    response = call_openai_engine(engine=args.engine,
                                                  api_key=args.api_key,
                                                  prompt=verify_label_prompt,
                                                  temperature=0.0)
                    openai_generation, usage = get_openai_generation(args.engine, response)
                    new_cost = estimate_cost(usage, args.engine)
                    total_cost += new_cost

                    prompt_track_file_handler.writelines(json.dumps({'prompt_type': 'verify_label_prompt',
                                                                     'prompt': verify_label_prompt,
                                                                     'response': openai_generation,
                                                                     **{k: unlabeled_sample[k] for k in
                                                                        unlabeled_sample.keys() - {'sim_scores'}},
                                                                     }, ensure_ascii=False))
                    prompt_track_file_handler.write('\n')
                    # parse openai generation
                    try:
                        con_valid = False
                        pred = remove_dot_end_symbol(openai_generation).strip().lower()
                        if args.dataset == 'semeval':
                            pred = re.sub(r'\(.*?\)', '', pred)
                            if '-' in pred:
                                left = pred.index('-')
                                pred = pred[0:left] + '(subject entity)' + pred[left:] + '(object entity)'
                        llm_gen_success = True
                        pred_rel_id = -1
                        for labelword in labelword2rel.keys():
                            if labelword in pred:
                                pred_rel_id = rel2id[labelword2rel[labelword]]
                                break
                        if pred_rel_id == -1:
                            for labelword in rel2id.keys():
                                if labelword in pred:
                                    pred_rel_id = rel2id[labelword]
                                    break
                        # con_valid can be true only when LLM == SLM
                        if pred_rel_id == rel2id[prompt_kwargs['pred_relation']]:
                            prompt_kwargs['pred_relation'] = id2rel[pred_rel_id]
                            con_valid = True
                    except:
                        logger.warning(
                            f'\nFail to parse the response for prompt:\n{verify_label_prompt}\nThe response:\n{openai_generation}')
                        continue
                if con_valid:
                    passed_unlabeled_samples.append(unlabeled_sample['sample_id'])
                    new_data = new_passed_data(unlabeled_sample, prompt_kwargs)
                    all_passed_unlabeled_data.append(new_data)
                    iter_passed_unlabeled_data.append(new_data)
                    if pr_file_handler is not None:
                        pr_file_handler.writelines(json.dumps({**new_data}, ensure_ascii=False))
                        pr_file_handler.write('\n')
                    if iter_new_passed_save_handler is not None:
                        iter_new_passed_save_handler.writelines(json.dumps({**new_data}, ensure_ascii=False))
                        iter_new_passed_save_handler.write('\n')
                    if iter_all_passed_save_handler is not None:
                        iter_all_passed_save_handler.writelines(json.dumps({**new_data}, ensure_ascii=False))
                        iter_all_passed_save_handler.write('\n')
                pbar.update(1)
        unlabeled_data = [data for data in unlabeled_data if
                          data['sample_id'] not in passed_unlabeled_samples]
        logger.info(f'-------------Error Ratio on new passed unlabeled data high-confidence {iteration_id}---------------')
        print(f'-------------Error Ratio on new passed unlabeled data high-confidence {iteration_id}---------------')
        passed_data_evaluation(iter_passed_unlabeled_data, logger, rel2id)
        ##############
        # low confidence unlabeled data
        # using rule matching scores
        unlabeled_data.sort(key=lambda x: x['top1_matched_rule']['confidence'])
        passed_unlabeled_samples = []
        select_low_conf = True
        for _, unlabeled_sample in enumerate(unlabeled_data[:low_conf_passed_limit]):
            if not select_low_conf:
                break
            if len(passed_unlabeled_samples) >= low_conf_passed_limit:
                break
            prompt_kwargs = {
                'sentence': unlabeled_sample['sentence'] if 'sentence' in unlabeled_sample.keys() else ' '.join([convert_ptb_token(token) for token in unlabeled_sample['token']]),
                'subject_entity': unlabeled_sample['subject_entity'] if 'subject_entity' in unlabeled_sample.keys() else unlabeled_sample['h']['name'],
                'object_entity': unlabeled_sample['object_entity'] if 'object_entity' in unlabeled_sample.keys() else unlabeled_sample['t']['name'],
                'subject_pattern': unlabeled_sample['subject_pattern'],
                'object_pattern': unlabeled_sample['object_pattern'],
                'rel_pattern': unlabeled_sample['rel_pattern'],
                'pred_relation': unlabeled_sample['pred_relation'],
            }

            if prompt_kwargs['subject_pattern'] is None:
                logger.warning(
                    f"\n{unlabeled_sample['slm_pred_rule']} is not correctly post processed!")
                pbar.update(1)
                continue

            if prompt_kwargs['pred_relation'] is None:
                logger.warning(
                    f"\nFail to obtain labels from {unlabeled_sample['slm_pred_rule']}!")
                pbar.update(1)
                continue

            if 'sentence_ent_tag' in unlabeled_sample.keys():
                prompt_kwargs['sentence_ent_tag'] = unlabeled_sample['sentence_ent_tag']
            else:
                tokens_ent_tag = []
                for token_index, token in enumerate(unlabeled_sample['token']):
                    if token_index == unlabeled_sample['h']['pos'][0]:
                        tokens_ent_tag.append('<Sub>')
                    if token_index == unlabeled_sample['h']['pos'][1]:
                        tokens_ent_tag.append('</Sub>')
                    if token_index == unlabeled_sample['t']['pos'][0]:
                        tokens_ent_tag.append('<Obj>')
                    if token_index == unlabeled_sample['t']['pos'][1]:
                        tokens_ent_tag.append('</Obj>')
                    tokens_ent_tag.append(convert_ptb_token(token))
                prompt_kwargs['sentence_ent_tag'] = ' '.join(tokens_ent_tag)

            # remove possible space between character and symbol
            prompt_kwargs['subject_entity'] = prompt_kwargs['subject_entity'].strip()
            prompt_kwargs['subject_entity'] = re.sub(r'\s*(\W+)\s*', r'\1', prompt_kwargs['subject_entity'])
            prompt_kwargs['object_entity'] = prompt_kwargs['object_entity'].strip()
            prompt_kwargs['object_entity'] = re.sub(r'\s*(\W+)\s*', r'\1', prompt_kwargs['object_entity'])

            if verify_method == 'LLM_GEN':
                llm_gen_success = False
                top_k_similar_rule_ids = torch.sort(unlabeled_sample['sim_scores'],
                                                    descending=True).indices[:args.in_context_size].cpu().tolist()
                while not llm_gen_success:
                    verify_prem_prompt = ''
                    demos = []
                    for rule_id in top_k_similar_rule_ids:
                        premise = labeled_rule_subj_patterns[rule_id] + ' ' + \
                                  labeled_rule_rel_patterns[rule_id] + ' ' + \
                                  labeled_rule_obj_patterns[rule_id]
                        if premise in labeled_rule2data:
                            demos.append(random.sample(labeled_rule2data[premise], k=1)[0])
                    demos = demos[:args.in_context_size]
                    demo_labels = [data['relation'] for data in demos]
                    if prompt_kwargs['pred_relation'] not in demo_labels:
                        if prompt_kwargs['pred_relation'] in labeled_data_w_rule_by_rel.keys():
                            demos.append(random.sample(labeled_data_w_rule_by_rel[prompt_kwargs['pred_relation']], k=1)[0])
                    ############
                    # LLM generates premise for unlabeled data
                    for id, data in enumerate(demos):
                        verify_prem_prompt += f"Sentence: {data['sentence_ent_tag']} Subject entity: {data['subject_entity']}. Object entity: {data['object_entity']}.\n" + \
                                              f"1. The type of {data['subject_entity']} is {data['subject_pattern']}.\n" + \
                                              f"2. The type of {data['object_entity']} is {data['object_pattern']}.\n" + \
                                              f"3. {data['rel_pattern'].replace('subject entity', data['subject_entity']).replace('object entity', data['object_entity'])}.\n"
                    query = '''\
                            Sentence: {sentence_ent_tag} Subject entity: {subject_entity}. Object entity: {object_entity}.
                            '''.format(**prompt_kwargs)
                    query = textwrap.dedent(query)
                    query += f"Given that the sentence, along with subject entity '{prompt_kwargs['subject_entity']}' (enclosed by <Sub></Sub>) and object entity '{prompt_kwargs['object_entity']}' (enclosed by <Obj></Obj>), judge the types of subject and object entities, as well as the relationship between these two entities.\n"
                    query += f"The responses should adhere to the subsequent format without any supplementary information, explanations, or notes:\n" + \
                             f"1. The type of {prompt_kwargs['subject_entity']} is [Entity Type1]\n" + \
                             f"2. The type of {prompt_kwargs['object_entity']} is [Entity Type2]\n" + \
                             f"3. [Relationship Description between {prompt_kwargs['subject_entity']} and {prompt_kwargs['object_entity']}].\n"
                    query += "Note that the subject entity and object entity may possess a specific relationship or no relationship, which can be inferred from the provided sentence."
                    verify_prem_prompt += query
                    # call openai
                    response = call_openai_engine(engine=args.engine,
                                                  api_key=args.api_key,
                                                  prompt=verify_prem_prompt,
                                                  temperature=0.0)
                    openai_generation, usage = get_openai_generation(args.engine, response)
                    new_cost = estimate_cost(usage, args.engine)
                    total_cost += new_cost

                    prompt_track_file_handler.writelines(json.dumps({'prompt_type': 'verify_prem_prompt',
                                                                     'prompt': verify_prem_prompt,
                                                                     'response': openai_generation,
                                                                     # **prompt_kwargs,
                                                                     **{k: unlabeled_sample[k] for k in
                                                                        unlabeled_sample.keys() - {'sim_scores'}},
                                                                     }, ensure_ascii=False))
                    prompt_track_file_handler.write('\n')

                    try:
                        ############
                        # post-processing prompt
                        prem_valid = False
                        res = openai_generation.strip()
                        llm_gen_success = True
                        if prompt_kwargs['subject_entity'].lower() in openai_generation.lower() and \
                                prompt_kwargs['object_entity'].lower() in openai_generation.lower():
                            res = res[res.index('1.'):]
                            left = res.index(f"{prompt_kwargs['subject_entity']} is") + len(
                                f"{prompt_kwargs['subject_entity']} is")
                            right = res.index('2.')
                            subject_pattern = remove_square_bracket_symbol(
                                remove_dot_end_symbol(res[left:right].strip()))
                            if 'type ' in subject_pattern:  # remove possible redundant 'type' word
                                subject_pattern = subject_pattern.replace('type ', '').strip()
                            if ' type' in subject_pattern:  # remove possible redundant 'type' word
                                subject_pattern = subject_pattern.replace(' type', '').strip()
                            if '\"' in subject_pattern:
                                subject_pattern = subject_pattern.replace('\"', '').strip()
                            if '\'' in subject_pattern:
                                subject_pattern = subject_pattern.replace('\'', '').strip()
                            prompt_kwargs['subject_pattern'] = subject_pattern.strip()

                            res = res[res.index('2.'):]
                            left = res.index(f"{prompt_kwargs['object_entity']} is") + len(
                                f"{prompt_kwargs['object_entity']} is")
                            right = res.index('3.')
                            object_pattern = remove_square_bracket_symbol(
                                remove_dot_end_symbol(res[left:right].strip()))
                            if 'type ' in object_pattern:  # remove possible redundant 'type' word
                                object_pattern = object_pattern.replace('type ', '').strip()
                            if ' type' in object_pattern:  # remove possible redundant 'type' word
                                object_pattern = object_pattern.replace(' type', '').strip()
                            if '\"' in object_pattern:
                                object_pattern = object_pattern.replace('\"', '').strip()
                            if '\'' in object_pattern:
                                object_pattern = object_pattern.replace('\'', '').strip()
                            prompt_kwargs['object_pattern'] = object_pattern.strip()

                            # rel_pattern
                            pred_rel_pattern = remove_square_bracket_symbol(res[res.index('3.') + len('3.'):].strip())
                            # Find possible subject entity in relation description
                            (left, right), pred_rel_pattern = find_word_in_text(prompt_kwargs['subject_entity'],
                                                                                pred_rel_pattern)
                            if left is not None:
                                pred_rel_pattern = pred_rel_pattern[:left] + f' subject entity ' + pred_rel_pattern[
                                                                                                   right:]
                            # Find possible object entity in relation description
                            (left, right), pred_rel_pattern = find_word_in_text(prompt_kwargs['object_entity'],
                                                                                pred_rel_pattern.strip())
                            if left is not None:
                                pred_rel_pattern = pred_rel_pattern[:left] + f' object entity ' + pred_rel_pattern[
                                                                                                  right:]
                            if 'subject entity' not in pred_rel_pattern:
                                pred_rel_pattern = 'subject entity ' + pred_rel_pattern
                            if 'object entity' not in pred_rel_pattern:
                                pred_rel_pattern = pred_rel_pattern + ' object entity'

                            if '  ' in pred_rel_pattern:
                                pred_rel_pattern = pred_rel_pattern.replace('  ', ' ')
                            prompt_kwargs['rel_pattern'] = pred_rel_pattern.strip()
                            llm_gen_success = True
                        prem_valid = True
                    except:
                        logger.warning(
                            f'\nFail to parse the response for prompt:\n{verify_prem_prompt}\nThe response:\n{openai_generation}')
                        continue
                # ------------ judge rule label ------------
                if not prem_valid:
                    pbar.update(1)
                    continue
                llm_gen_success = False
                con_valid = False
                while not llm_gen_success:
                    verify_label_prompt = f'''\
                                          Given a sentence, a pair of subject (enclosed by <Sub></Sub>) and object entities (enclosed by <Obj></Obj>) in the sentence, decide the most precise relationship between the subject and object entities. If not sure, choose label '{rel2labelword[id2rel[NOTA_LABEL_ID]]}'.
                                          Note that the relationship must be one of the defined relation from candidate relations:
                                          {', '.join(labelword2rel.keys())}.
                                          Provide the relationship label without any supplementary information, explanations, or notes.\n
                                          '''
                    verify_label_prompt = textwrap.dedent(verify_label_prompt)

                    # # 1. LLM predicted patterns
                    unlabeled_subj_pattern_emb = model.get_sentence_embeddings(
                        texts=[prompt_kwargs['subject_pattern'].lower()])
                    unlabeled_obj_pattern_emb = model.get_sentence_embeddings(
                        texts=[prompt_kwargs['object_pattern'].lower()])
                    unlabeled_rel_pattern_emb = model.get_sentence_embeddings(
                        texts=[prompt_kwargs['rel_pattern'].lower()])
                    # # 2. unlabeled data-rule matching scores
                    new_sim_scores = 1.0 / 3.0 * (
                            compute_cos_sim(unlabeled_subj_pattern_emb, labeled_subj_pattern_embs) +
                            compute_cos_sim(unlabeled_obj_pattern_emb, labeled_obj_pattern_embs) +
                            compute_cos_sim(unlabeled_rel_pattern_emb, labeled_rel_pattern_embs))
                    top_k_similar_rule_ids = torch.sort(new_sim_scores[0],
                                                        descending=True).indices[:args.in_context_size].cpu().tolist()

                    demos = []
                    for rule_id in top_k_similar_rule_ids:
                        premise = labeled_rule_subj_patterns[rule_id] + ' ' + \
                                  labeled_rule_rel_patterns[rule_id] + ' ' + \
                                  labeled_rule_obj_patterns[rule_id]
                        if premise in labeled_rule2data:
                            demos.append(random.sample(labeled_rule2data[premise], k=1)[0])
                            # demos.extend(labeled_rule2data[premise])
                    demos = demos[:args.in_context_size]
                    demo_labels = [data['relation'] for data in demos]
                    if prompt_kwargs['pred_relation'] not in demo_labels:
                        if prompt_kwargs['pred_relation'] in labeled_data_w_rule_by_rel.keys():
                            demos.append(random.sample(labeled_data_w_rule_by_rel[prompt_kwargs['pred_relation']], k=1)[0])

                    # p -> r
                    verify_label_prompt += "Some labeling rules includes:\n"
                    for id, data in enumerate(demos):
                        verify_label_prompt += f"{id + 1}. If the type of subject entity is {data['subject_pattern']}, " + \
                                               f"the type of object entity is {data['object_pattern']} " + \
                                               f"and {data['rel_pattern']}, " + \
                                               f"then the relation is {rel2labelword[data['relation']]}.\n"
                    verify_label_prompt += f"{len(demos) + 1}. If subject entity has no relationship with object entity, then the relation is {rel2labelword[id2rel[NOTA_LABEL_ID]]}.\n"
                    verify_label_prompt += f"{len(demos) + 2}. If none of the above relation labels can be valid or correct between subject entity and object entity, then the relation is {rel2labelword[id2rel[NOTA_LABEL_ID]]}.\n"

                    query = f"Sentence: {prompt_kwargs['sentence_ent_tag']} Subject entity: {prompt_kwargs['subject_entity']}. Object entity: {prompt_kwargs['object_entity']}. " + \
                            f"We can infer that the type of subject entity '{prompt_kwargs['subject_entity']}' is {prompt_kwargs['subject_pattern']}, the type of object entity '{prompt_kwargs['object_entity']}' is {prompt_kwargs['object_pattern']}, and {prompt_kwargs['rel_pattern'].replace('subject entity', prompt_kwargs['subject_entity']).replace('object entity', prompt_kwargs['object_entity'])}. " + \
                            f"The relation between {prompt_kwargs['subject_entity']} and {prompt_kwargs['object_entity']} in the sentence is"
                    verify_label_prompt += query

                    # call openai
                    response = call_openai_engine(engine=args.engine,
                                                  api_key=args.api_key,
                                                  prompt=verify_label_prompt,
                                                  temperature=0.0)
                    openai_generation, usage = get_openai_generation(args.engine, response)
                    new_cost = estimate_cost(usage, args.engine)
                    total_cost += new_cost

                    prompt_track_file_handler.writelines(json.dumps({'prompt_type': 'verify_label_prompt',
                                                                     'prompt': verify_label_prompt,
                                                                     'response': openai_generation,
                                                                     **{k: unlabeled_sample[k] for k in
                                                                        unlabeled_sample.keys() - {'sim_scores'}},
                                                                     # **prompt_kwargs,
                                                                     }, ensure_ascii=False))
                    prompt_track_file_handler.write('\n')
                    # parse openai generation
                    try:
                        con_valid = False
                        pred = remove_dot_end_symbol(openai_generation).strip().lower()
                        if args.dataset == 'semeval':
                            pred = re.sub(r'\(.*?\)', '', pred)
                            if '-' in pred:
                                left = pred.index('-')
                                pred = pred[0:left] + '(subject entity)' + pred[left:] + '(object entity)'
                        llm_gen_success = True
                        pred_rel_id = -1
                        for labelword in labelword2rel.keys():
                            if labelword in pred:
                                pred_rel_id = rel2id[labelword2rel[labelword]]
                                break
                        for labelword in rel2id.keys():
                            if labelword in pred:
                                pred_rel_id = rel2id[labelword]
                                break
                        # con_valid can be true only when LLM == NOTA
                        if pred_rel_id == NOTA_LABEL_ID:
                            prompt_kwargs['pred_relation'] = id2rel[pred_rel_id]
                            con_valid = True
                    except:
                        logger.warning(
                            f'\nFail to parse the response for prompt:\n{verify_label_prompt}\nThe response:\n{openai_generation}')
                        continue
                if con_valid:
                    passed_unlabeled_samples.append(unlabeled_sample['sample_id'])
                    new_data = new_passed_data(unlabeled_sample, prompt_kwargs)
                    all_passed_unlabeled_data.append(new_data)
                    iter_passed_unlabeled_data.append(new_data)
                    if pr_file_handler is not None:
                        pr_file_handler.writelines(json.dumps({**new_data}, ensure_ascii=False))
                        pr_file_handler.write('\n')
                    if iter_new_passed_save_handler is not None:
                        iter_new_passed_save_handler.writelines(json.dumps({**new_data}, ensure_ascii=False))
                        iter_new_passed_save_handler.write('\n')
                    if iter_all_passed_save_handler is not None:
                        iter_all_passed_save_handler.writelines(json.dumps({**new_data}, ensure_ascii=False))
                        iter_all_passed_save_handler.write('\n')
                pbar.update(1)
        pr_file_handler.flush()
        unlabeled_data = [data for data in unlabeled_data if
                          data['sample_id'] not in passed_unlabeled_samples]

        end_time = time.time()
        logger.info(
            f'Iteration:{iteration_id}: Querying the LLM using: {end_time - start_time} seconds, {(end_time - start_time) / 60.0} minutes')
        logger.info(f'The total cost {total_cost}$, nearly {total_cost * 7.24}￥.')

        iter_new_passed_save_handler.flush()
        iter_all_passed_save_handler.flush()

        current_unlabeled_data_path = f"{output_dir}/pr/{args.exp_name}_iters/{iteration_id}_unlabeled.json"
        Path(current_unlabeled_data_path).parent.mkdir(exist_ok=True, parents=True)
        logger.info(f'-------------Remaining unlabeled data {len(unlabeled_data)} is saved to {current_unlabeled_data_path}---------------')
        unlabeled_save_handler = open(current_unlabeled_data_path, 'a')
        for data in unlabeled_data:
            # save the labeled data
            unlabeled_save_handler.writelines(json.dumps({**{k: data[k] for k in data.keys() - {'sim_scores'}}},
                                                         ensure_ascii=False))
            unlabeled_save_handler.write('\n')
        unlabeled_save_handler.flush()
        prompt_track_file_handler.flush()

        logger.info(f'-------------Error Ratio on new passed unlabeled data after iteration {iteration_id}---------------')
        print(f'-------------Error Ratio on new passed unlabeled data after iteration {iteration_id}---------------')
        passed_data_evaluation(iter_passed_unlabeled_data, logger, rel2id)
        logger.info(f'-------------Error Ratio on all passed unlabeled data after iteration {iteration_id}---------------')
        print(f'-------------Error Ratio on all passed unlabeled data after iteration {iteration_id}---------------')
        passed_data_evaluation(all_passed_unlabeled_data, logger, rel2id)

        # use new labeled data to train SLM
        del model
        slm_finetune_args = f"--exp_name {args.exp_name}_iter{iteration_id} " + \
                            f"--fix_exp_name True " + \
                            f"--dataset {args.dataset} " + \
                            f"--lre_setting {args.lre_setting} " + \
                            f"--k_shot_lre {args.k_shot_lre} --k_shot_id {args.k_shot_id} " + \
                            f"--train_file {args.exp_name}_pr.json"
        slm_finetune_args = slm_args_parser().parse_args(slm_finetune_args.split())
        post_processing_args(slm_finetune_args)
        logger.info(f'-------------Fine-tuning new SLM after iteration {iteration_id} ---------------')
        slm_finetune(slm_finetune_args, logger, eval_on_unlabeled_data=False)
        slm_save_dir = slm_finetune_args.model_save_dir + f"/{args.exp_name}_iter{iteration_id}"
        model = T5Model(resume_from_save_dir=True,
                        save_dir=slm_save_dir,
                        batch_size=64 if torch.cuda.is_available() else 16)
        # # using SLM to generate rules for current unlabeled data
        unlabeled_slm_pred_path = f"{output_dir}/pr/{args.exp_name}_iters/{iteration_id}_unlabeled_rules.json"
        Path(unlabeled_slm_pred_path).parent.mkdir(exist_ok=True, parents=True)
        model.predict(path_in=current_unlabeled_data_path, path_out=unlabeled_slm_pred_path, args=args, logger=logger)
        logger.info(f'-------------Performance of new SLM on unlabeled set after iteration {iteration_id} ---------------')
        slm_pred_file_evaluation(unlabeled_slm_pred_path, rel2id_file_path, logger)
