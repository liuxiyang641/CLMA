import json
import re
import textwrap
from tqdm import tqdm
import os
import random
import time

from src.utils.openai_utils import call_openai_engine, get_openai_generation, estimate_cost
from src.utils.utils import *
from src.slm.t5_model import T5Model

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def llm_inference(args, logger):
    """ extract relations using rules induced from labeled data  """
    # -------------- Reading data and output settings --------------
    with open(f"src/configs/data_config.json", "r", encoding="utf-8") as f:
        data_configs = json.load(f)
        rel2id_file_path = data_configs[args.dataset]['rel2id']
        test_file_path = data_configs[args.dataset]['test']  # test file

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
            rel2labelword[rel] = (rel.lower().replace("_", " ").
                                  replace("-", " ").
                                  replace("per", "person").
                                  replace("org", "organization").
                                  replace("stateor", "state or "))
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
    # load previous induced premises from labeled training samples
    with open(f"src/configs/rule_train_config.json", "r", encoding="utf-8") as f:
        train_rule_files = json.load(f)
    train_rule_dir = f'{output_dir}/pr/'
    if args.lre_setting == 'k-shot':
        train_rule_path = train_rule_dir + train_rule_files[args.dataset]['k-shot'][
            f'{args.k_shot_lre}-{args.k_shot_id}']
    train_data_w_rule, train_data_w_rule_by_rel = read_data_file(train_rule_path,
                                                                 max_count=args.debug_test_num if args.mode == 'debug' else None)
    logger.info(f'Loading induced rules from labeled data: {train_rule_path}, {len(train_data_w_rule)} data.')

    # load test samples
    test_data, test_data_by_rel = read_data_file(test_file_path,
                                                 max_count=args.debug_test_num - 1 if args.mode == 'debug' else None)
    logger.info(f'Loading test data from the file: {test_file_path}, {len(test_data)} testing samples.')


    output_dir = os.path.join(args.output_dir, f"{args.dataset}/{args.k_shot_lre}-{args.k_shot_id}")
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

    # -------------- Loading small pretrained model --------------
    with open(f"src/configs/slm_ckpt_config.json", "r", encoding="utf-8") as f:
        slm_save_dirs = json.load(f)
    if args.lre_setting == 'k-shot':
        slm_save_dir = f"src/slm/ckpt/{args.dataset}/k-shot/{args.k_shot_lre}-{args.k_shot_id}/" + \
                       slm_save_dirs[args.dataset]['k-shot'][f'{args.k_shot_lre}-{args.k_shot_id}']

    model = T5Model(resume_from_save_dir=True, save_dir=slm_save_dir, batch_size=128 if torch.cuda.is_available() else 16)

    path_in = test_file_path
    path_out = f"{output_dir}/slm/re_{args.exp_name}_pred_rule.json"
    model.predict(path_in=path_in, path_out=path_out, args=args, logger=logger)

    slm_pred_test_rules, _ = read_data_file(path_out, max_count=args.debug_test_num - 1 if args.mode == 'debug' else None)  # max_count=args.debug_test_num if args.mode == 'debug' else None

    # obtain rule embeddings of labeled data
    def rule_verbalization(ins):
        return f"Subject entity type: {ins['subject_pattern']}. Object entity type: {ins['object_pattern']}. " + \
            f"Relationship: subject entity {ins['pred_rel_pattern']} object entity."

    #############
    # ablation study, without rule
    match_degree_method = 'PATTERN2PATTERN'
    if match_degree_method == 'PREMISE2PERMISE':
        # obtain rule embeddings of unlabeled data
        prev_rule_embs = model.get_sentence_embeddings(
            texts=[x['premise'] if x['premise'] is not None else '' for x in train_data_w_rule])
        test_rule_embs = model.get_sentence_embeddings(texts=[rule_verbalization(ins) for ins in slm_pred_test_rules])
        rule_sim_scores = compute_cos_sim(test_rule_embs, prev_rule_embs)
    elif match_degree_method == 'PATTERN2PATTERN':
        labeled_rule2data = {}
        for ins in train_data_w_rule:
            if ins['subject_pattern'] is not None:
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
        # SLM predicted patterns
        test_subj_pattern_embs = model.get_sentence_embeddings(
            texts=[x['subject_pattern'].lower() if x['subject_pattern'] is not None else '' for x in slm_pred_test_rules])
        test_obj_pattern_embs = model.get_sentence_embeddings(
            texts=[x['object_pattern'].lower() if x['object_pattern'] is not None else '' for x in slm_pred_test_rules])
        test_rel_pattern_embs = model.get_sentence_embeddings(
            texts=[x['rel_pattern'].lower() if x['rel_pattern'] is not None else '' for x in slm_pred_test_rules])
        labeled_subj_pattern_embs = model.get_sentence_embeddings(
            texts=labeled_rule_subj_patterns)
        labeled_obj_pattern_embs = model.get_sentence_embeddings(
            texts=labeled_rule_obj_patterns)
        labeled_rel_pattern_embs = model.get_sentence_embeddings(
            texts=labeled_rule_rel_patterns)
        rule_sim_scores = 1.0 / 3.0 * (compute_cos_sim(test_subj_pattern_embs, labeled_subj_pattern_embs) +
                                       compute_cos_sim(test_obj_pattern_embs, labeled_obj_pattern_embs) +
                                       compute_cos_sim(test_rel_pattern_embs, labeled_rel_pattern_embs))
    elif match_degree_method == 'SEN2SEN':
        labeled_sens = [x['sentence_ent_tag'] for x in train_data_w_rule]
        labeled_sen_embs = model.get_sentence_embeddings(texts=labeled_sens)
        unlabeled_sens = [x['sentence_ent_tag'] for x in slm_pred_test_rules]
        unlabeled_sen_embs = model.get_sentence_embeddings(texts=unlabeled_sens)
        rule_sim_scores = compute_cos_sim(unlabeled_sen_embs, labeled_sen_embs)
    slm_rel2labelword = {}
    for rel in rel2id.keys():
        slm_rel2labelword[rel] = convert_label2words(rel)
    slm_labelword2rel = {}
    for k, v in slm_rel2labelword.items():
        slm_labelword2rel[v] = k

    for sample_id, test_sample in enumerate(test_data):
        test_sample['slm_pred_rule'] = slm_pred_test_rules[sample_id]['slm_pred_rule']
        test_sample['subject_pattern'] = slm_pred_test_rules[sample_id]['subject_pattern']
        test_sample['object_pattern'] = slm_pred_test_rules[sample_id]['object_pattern']
        test_sample['rel_pattern'] = slm_pred_test_rules[sample_id]['rel_pattern']
        # post-processing SLM predicted relations
        slm_pred_relation = slm_pred_test_rules[sample_id]['pred_relation']
        pred_rel_id = NOTA_LABEL_ID
        if slm_pred_relation:
            for labelword in slm_labelword2rel.keys():
                if labelword in slm_pred_relation:
                    pred_rel_id = rel2id[slm_labelword2rel[labelword]]
                    break
            slm_pred_relation = id2rel[pred_rel_id]
        else:
            slm_pred_relation = None
        test_sample['pred_relation'] = slm_pred_relation
        test_sample['sim_scores'] = rule_sim_scores[sample_id]
        top_k_similar_rule_ids = torch.sort(rule_sim_scores[sample_id],
                                            descending=True).indices[:args.in_context_size].cpu().tolist()
        for i, rule_id in enumerate(top_k_similar_rule_ids):
            if match_degree_method == 'SEN2SEN':
                test_sample[f'top{i + 1}_matched_rule'] = {
                    'confidence': rule_sim_scores[sample_id][rule_id].tolist(),
                }
            else:
                test_sample[f'top{i + 1}_matched_rule'] = {
                    'premise': labeled_rule_premise[rule_id],
                    'relation': labeled_rule_relation[rule_id],
                    'confidence': rule_sim_scores[sample_id][rule_id].tolist(),
                }

        test_sample['confidence'] = slm_pred_test_rules[sample_id]['confidence']

    rule_match_file = os.path.join(output_dir, f"slm/re_{args.exp_name}_rule_match.json")
    rule_match_file_handler = open(rule_match_file, 'a')
    for data in test_data:
        rule_match_file_handler.writelines(json.dumps({**{k: data[k] for k in
                                                            data.keys() - {'sim_scores'}}}, ensure_ascii=False))
        rule_match_file_handler.write('\n')

    test_data_by_rel = {}
    for data in test_data:
        rel = data['relation']
        if rel not in test_data_by_rel:
            test_data_by_rel[rel] = [data]
        else:
            test_data_by_rel[rel].append(data)

    # -------------- Querying the LLM --------------
    total_cost = 0  # total_cost is used to estimate money cost
    start_time = time.time()
    pbar = tqdm(total=len(test_data), desc=f'Identifying relations')

    pred_rel_ids, true_rel_ids = [], []
    for relation_label, test_samples in test_data_by_rel.items():
        for test_sample in test_samples:
            prompt_kwargs = {'sentence': ' '.join([convert_ptb_token(token) for token in test_sample['token']]),
                             'subject_entity': test_sample['h']['name'],
                             'object_entity': test_sample['t']['name'],
                             'relation': relation_label,
                             'subject_pattern': test_sample['subject_pattern'],
                             'object_pattern': test_sample['object_pattern'],
                             'rel_pattern': test_sample['rel_pattern'],
                             'pred_relation': test_sample['pred_relation'],
                             }
            tokens_ent_tag = []
            for token_index, token in enumerate(test_sample['token']):
                if token_index == test_sample['h']['pos'][0]:
                    tokens_ent_tag.append('<Sub>')
                if token_index == test_sample['h']['pos'][1]:
                    tokens_ent_tag.append('</Sub>')
                if token_index == test_sample['t']['pos'][0]:
                    tokens_ent_tag.append('<Obj>')
                if token_index == test_sample['t']['pos'][1]:
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
                top_k_similar_rule_ids = torch.sort(test_sample['sim_scores'],
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
                        if prompt_kwargs['pred_relation'] in train_data_w_rule_by_rel.keys():
                            demos.append(random.sample(train_data_w_rule_by_rel[prompt_kwargs['pred_relation']], k=1)[0])
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
                                                                     **{k: test_sample[k] for k in
                                                                        test_sample.keys() - {'sim_scores'}},
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
                                pred_rel_pattern = pred_rel_pattern[:left] + f' subject entity ' + pred_rel_pattern[right:]
                            # Find possible object entity in relation description
                            (left, right), pred_rel_pattern = find_word_in_text(prompt_kwargs['object_entity'],
                                                                                pred_rel_pattern.strip())
                            if left is not None:
                                pred_rel_pattern = pred_rel_pattern[:left] + f' object entity ' + pred_rel_pattern[right:]
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
                llm_gen_success = False
                con_valid = False
                while not llm_gen_success:
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
                        if prompt_kwargs['pred_relation'] in train_data_w_rule_by_rel.keys():
                            demos.append(random.sample(train_data_w_rule_by_rel[prompt_kwargs['pred_relation']], k=1)[0])
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
                                                                     **{k: test_sample[k] for k in
                                                                        test_sample.keys() - {'sim_scores'}},
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
                        random_res = rel2id[random.sample(rel2id.keys(), k=1)[0]]
                        pred_rel_id = random_res
                        for labelword in labelword2rel.keys():
                            if labelword in pred:
                                pred_rel_id = rel2id[labelword2rel[labelword]]
                                break
                        for labelword in rel2id.keys():
                            if labelword in pred:
                                pred_rel_id = rel2id[labelword]
                                break
                        prompt_kwargs['pred_relation'] = id2rel[pred_rel_id]
                        true_rel_ids.append(rel2id[prompt_kwargs['relation']])
                        pred_rel_ids.append(pred_rel_id)
                    except:
                        logger.warning(
                            f'\nFail to parse the response for prompt:\n{verify_label_prompt}\nThe response:\n{openai_generation}')
                        continue
                pbar.update(1)
                pr_file_handler.writelines(
                    json.dumps({'relation': prompt_kwargs['pred_relation'],
                                'subject_pattern': prompt_kwargs['subject_pattern'],
                                'object_pattern': prompt_kwargs['object_pattern'],
                                'rel_pattern': prompt_kwargs['rel_pattern'],
                                'subject_entity': prompt_kwargs['subject_entity'],
                                'object_entity': prompt_kwargs['object_entity'],
                                'sentence': prompt_kwargs['sentence'],
                                'token': test_sample['token'],
                                'h': test_sample['h'],
                                't': test_sample['t'],
                                'sentence_ent_tag': prompt_kwargs['sentence_ent_tag'],
                                'gold_relation': prompt_kwargs['relation'],
                                }, ensure_ascii=False))
                pr_file_handler.write('\n')

    end_time = time.time()
    evaluation(true_rel_ids, pred_rel_ids, rel2id, logger)
    logger.info(f'Querying the LLM using: {end_time - start_time} seconds, {(end_time - start_time) / 60.0} minutes')
    logger.info(f'The total cost {total_cost}$, nearly {total_cost * 7.24}￥.')
