import json
import re
import textwrap
from tqdm import tqdm
import os
import random
import time

from src.utils.openai_utils import call_openai_engine, get_openai_generation, estimate_cost
from src.utils.utils import *


def rule_induction(args, logger):
    """ extract induced rules from the real sentence and save  """
    with open(f"src/configs/data_config.json", "r", encoding="utf-8") as f:
        data_configs = json.load(f)
        rel2id_file_path = data_configs[args.dataset]['rel2id']
        if args.lre_setting == 'k-shot':
            # load k-shot train file
            data_set_path = data_configs[args.dataset]['k-shot'] + f'{args.k_shot_lre}-{args.k_shot_id}/{args.split}.json'
        else:
            raise ValueError('Unknown lre_setting hyperparameter!')


    rel2id = json.load(open(rel2id_file_path))
    id2rel = {v: k for k, v in rel2id.items()}
    NOTA_LABEL_ID = -1
    for name in ['NA', 'na', 'no_relation', 'Other', 'Others', 'other', 'false', 'unanswerable']:
        if name in rel2id:
            NOTA_LABEL_ID = rel2id[name]
            break

    # Label words
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

    train_data, train_data_by_rel = read_data_file(data_set_path,
                                                   max_count=args.debug_test_num if args.mode == 'debug' else None)

    random.seed(args.random_seed)  # fix random results

    if args.lre_setting == 'k-shot':
        saved_dir = os.path.join(args.output_dir, f"{args.dataset}/k-shot/{args.k_shot_lre}-{args.k_shot_id}")

    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)
        os.makedirs(os.path.join(saved_dir, 'pr'))
        os.makedirs(os.path.join(saved_dir, 'slm'))
        os.makedirs(os.path.join(saved_dir, 'track'))

    # the data file tracking prompt and response
    prompt_track_file = os.path.join(saved_dir, f'track/{args.exp_name}_prompt.json')
    prompt_track_file_handler = open(prompt_track_file, 'a')
    pr_file = os.path.join(saved_dir, f'pr/{args.exp_name}_premise.json')
    pr_file_handler = open(pr_file, 'a')

    total_cost = 0  # total_cost is used to estimate money cost
    start_time = time.time()
    pbar = tqdm(total=len(train_data), desc=f'Inducing premise of rule')
    logger.info(f'Loading data from the file: {data_set_path}, {len(train_data)} training samples.')

    # counter = 0
    for relation_label, training_samples in train_data_by_rel.items():
        # if counter >= 8:
        #     break
        # counter += 1
        for training_sample in training_samples:  # generating new samples for k training samples each relation
            prompt_kwargs = {'sentence': ' '.join([convert_ptb_token(token) for token in training_sample['token']]),
                             'subject_entity': training_sample['h']['name'],
                             'object_entity': training_sample['t']['name'],
                             'relation': relation_label,
                             'relation_labelword': rel2labelword[relation_label],
                             # 'label_desc': rel2guidelines[relation_label],
                             }
            tokens_ent_tag = []
            for token_index, token in enumerate(training_sample['token']):
                if token_index == training_sample['h']['pos'][0]:
                    tokens_ent_tag.append('<Sub>')
                if token_index == training_sample['h']['pos'][1]:
                    tokens_ent_tag.append('</Sub>')
                if token_index == training_sample['t']['pos'][0]:
                    tokens_ent_tag.append('<Obj>')
                if token_index == training_sample['t']['pos'][1]:
                    tokens_ent_tag.append('</Obj>')
                tokens_ent_tag.append(convert_ptb_token(token))
            prompt_kwargs['sentence_ent_tag'] = ' '.join(tokens_ent_tag)
            # remove possible space between character and symbol
            prompt_kwargs['subject_entity'] = prompt_kwargs['subject_entity'].strip()
            prompt_kwargs['subject_entity'] = re.sub(r'\s*(\W+)\s*', r'\1', prompt_kwargs['subject_entity'])
            prompt_kwargs['object_entity'] = prompt_kwargs['object_entity'].strip()
            prompt_kwargs['object_entity'] = re.sub(r'\s*(\W+)\s*', r'\1', prompt_kwargs['object_entity'])

            llm_gen_success = False
            max_tries = 5
            current_tries = 0
            while not llm_gen_success and current_tries < max_tries:
                current_tries += 1
                premise_induce_prompt = '''\
                                        Sentence: {sentence_ent_tag}
                                        Question: Given that {subject_entity} (enclosed by <Sub></Sub>) and {object_entity} (enclosed by <Obj></Obj>) are subject and object entities, what are the types of {subject_entity} and {object_entity} and what is the relationship between these two entities can you inferred from the given sentence?
                                        The responses should adhere to the subsequent format without any supplementary information, explanations, or notes:
                                        1. The mention {subject_entity} is an entity of [Entity Type1].
                                        2. The mention {object_entity} is an entity of [Entity Type2].
                                        3. {subject_entity}, [Relationship Description between {subject_entity} and {object_entity}], {object_entity}.
                                        Do not utilize the terms 'subject entity' or 'object entity' as [Entity Type1] or [Entity Type2].
                                        Do not repeat the given relation label {relation_labelword} in the [Relationship Description] which should be described in natural language.
                                        Note that [ ] marks the place that should be filled with the right description.
                                        '''.format(**prompt_kwargs)
                premise_induce_prompt = textwrap.dedent(premise_induce_prompt)

                # # call openai
                response = call_openai_engine(engine=args.engine,
                                              api_key=args.api_key,
                                              prompt=premise_induce_prompt,
                                              temperature=0.0)
                openai_generation, usage = get_openai_generation(args.engine, response)
                new_cost = estimate_cost(usage, args.engine)
                total_cost += new_cost
                logger.info(f'New cost {new_cost}$, the total cost {total_cost}$')

                prompt_track_file_handler.writelines(json.dumps({'prompt_type': 'premise_induce_prompt',
                                                                 'prompt': premise_induce_prompt,
                                                                 'response': openai_generation,
                                                                 **training_sample,
                                                                 }, ensure_ascii=False))
                prompt_track_file_handler.write('\n')

                # parse openai generation
                try:
                    if prompt_kwargs['relation'] in openai_generation:
                        continue
                    if prompt_kwargs['subject_entity'].lower() in openai_generation.lower() and \
                            prompt_kwargs['object_entity'].lower() in openai_generation.lower():
                        res = openai_generation[openai_generation.index('1.'):]
                        left = res.index(f"is an entity of") + len(f"is an entity of")
                        right = res.index('2.')

                        subject_pattern = remove_square_bracket_symbol(
                            remove_dot_end_symbol(res[left:right].strip())).lower()
                        if 'type ' in subject_pattern:  # remove possible redundant 'type' word
                            subject_pattern = subject_pattern.replace('type ', '').strip()
                        if ' type' in subject_pattern:  # remove possible redundant 'type' word
                            subject_pattern = subject_pattern.replace(' type', '').strip()
                        if 'subject entity' in subject_pattern.lower():
                            continue
                        if 'entity' in subject_pattern:
                            subject_pattern = subject_pattern.replace('entity', '').strip()
                        if 'Entity' in subject_pattern:
                            subject_pattern = subject_pattern.replace('Entity', '').strip()
                        if '\"' in subject_pattern:
                            subject_pattern = subject_pattern.replace('\"', '').strip()
                        if '\'' in subject_pattern:
                            subject_pattern = subject_pattern.replace('\'', '').strip()
                        prompt_kwargs['subject_pattern'] = subject_pattern.strip()

                        res = res[res.index('2.'):]
                        left = res.index(f"is an entity of") + len(f"is an entity of")
                        right = res.index('3.')
                        # remove possible [ ] symbol
                        object_pattern = remove_square_bracket_symbol(
                            remove_dot_end_symbol(res[left:right].strip())).lower()
                        if 'type ' in object_pattern:  # remove possible redundant 'type' word
                            object_pattern = object_pattern.replace('type ', '').strip()
                        if ' type' in object_pattern:  # remove possible redundant 'type' word
                            object_pattern = object_pattern.replace(' type', '').strip()
                        if 'object entity' in object_pattern.lower():
                            continue
                        if 'entity' in object_pattern:
                            object_pattern = object_pattern.replace('entity', '').strip()
                        if 'Entity' in object_pattern:
                            object_pattern = object_pattern.replace('Entity', '').strip()
                        if '\"' in object_pattern:
                            object_pattern = object_pattern.replace('\"', '').strip()
                        if '\'' in object_pattern:
                            object_pattern = object_pattern.replace('\'', '').strip()
                        prompt_kwargs['object_pattern'] = object_pattern.strip()

                        # find relation pattern
                        rel_pattern = remove_square_bracket_symbol(res[res.index('3.') + len('3.'):].strip())
                        if ',' in rel_pattern:
                            rel_pattern = rel_pattern.split(',')[1]
                        # Find possible subject entity in relation description
                        (left, right), rel_pattern = find_word_in_text(prompt_kwargs['subject_entity'],
                                                                       rel_pattern)
                        if left is not None:
                            rel_pattern = rel_pattern[:left] + f' subject entity ' + rel_pattern[right:]
                        # Find possible object entity in relation description
                        (left, right), rel_pattern = find_word_in_text(prompt_kwargs['object_entity'],
                                                                       rel_pattern.strip())
                        if left is not None:
                            rel_pattern = rel_pattern[:left] + f' object entity ' + rel_pattern[right:]
                        if 'subject entity' not in rel_pattern:
                            rel_pattern = 'subject entity ' + rel_pattern
                        if 'object entity' not in rel_pattern:
                            rel_pattern = rel_pattern + ' object entity'

                        if '  ' in rel_pattern:
                            rel_pattern = rel_pattern.replace('  ', ' ')
                        prompt_kwargs['rel_pattern'] = rel_pattern.strip().lower()
                        llm_gen_success = True
                except:
                    logger.warning(
                        f'\nFail to parse the response for prompt:\n{premise_induce_prompt}\nThe response:\n{openai_generation}')
                    continue
            if not llm_gen_success:
                pbar.update(1)
                continue

            # # iteration
            max_iter = 2
            conc_valid = False
            for iter_id in range(max_iter):
                if conc_valid:
                    break
                # ------------------ Step 2: judge label of rule premise ------------------
                llm_gen_success = False
                current_tries = 0
                while not llm_gen_success and current_tries < max_tries:
                    current_tries += 1
                    verify_label_prompt = f'''\
                                        One premise consists of the types of subject and object entities, as well as the relationship description between them.
                                        Select up to three most probable relation labels between the subject and object entities from candidate relation label list:
                                        {', '.join(labelword2rel.keys())}.
                                        Provide the relationship labels using a comma-separated list without any supplementary information, explanations, or notes.\n
                                        '''
                    verify_label_prompt = textwrap.dedent(verify_label_prompt)
                    query = '''\
                            Premise: The type of subject entity is {subject_pattern}, the type of object entity is {object_pattern} and subject entity {rel_pattern} object entity.
                            Possible relation labels:
                            '''.format(**prompt_kwargs)
                    verify_label_prompt += textwrap.dedent(query)
                    # call openai
                    response = call_openai_engine(engine=args.engine,
                                                  api_key=args.api_key,
                                                  prompt=verify_label_prompt,
                                                  temperature=0.0)
                    openai_generation, usage = get_openai_generation(args.engine, response)
                    new_cost = estimate_cost(usage, args.engine)
                    total_cost += new_cost
                    logger.info(f'New cost {new_cost}$, the total cost {total_cost}$')

                    prompt_track_file_handler.writelines(json.dumps({'prompt_type': 'verify_label_prompt',
                                                                     'prompt': verify_label_prompt,
                                                                     'response': openai_generation,
                                                                     **training_sample,
                                                                     }, ensure_ascii=False))
                    prompt_track_file_handler.write('\n')
                    # parse openai generation
                    try:
                        pred = remove_dot_end_symbol(openai_generation).strip().lower()
                        # multiple wrong labels
                        prompt_kwargs['pred_relation'] = []
                        prompt_kwargs['wrong_relation_labelword'] = []
                        pred_rel_id = -1
                        for labelword in labelword2rel.keys():
                            if labelword in pred:
                                pred_rel_id = rel2id[labelword2rel[labelword]]
                                prompt_kwargs['pred_relation'].append(id2rel[pred_rel_id])
                                if id2rel[pred_rel_id] != prompt_kwargs['relation']:
                                    prompt_kwargs['wrong_relation_labelword'].append(rel2labelword[id2rel[pred_rel_id]])
                        if len(prompt_kwargs['pred_relation']) != 0:
                            llm_gen_success = True
                            if len(prompt_kwargs['pred_relation']) == 1 and id2rel[pred_rel_id] == prompt_kwargs['relation']:
                                conc_valid = True
                            else:
                                conc_valid = False
                        else:
                            llm_gen_success = False
                    except:
                        logger.warning(
                            f'\nFail to parse the response for prompt:\n{verify_label_prompt}\nThe response:\n{openai_generation}')
                        continue
                if not llm_gen_success:
                    # if predicted label is not defined, use ground-truth label to refine rule patterns
                    new_current_tries = 0
                    while not llm_gen_success and new_current_tries < max_tries:
                        new_current_tries += 1
                        # with ground-truth label
                        premise_induce_by_label_prompt = '''\
                                                Sentence: {sentence_ent_tag}
                                                Question: Given that {subject_entity} (enclosed by <Sub></Sub>) and {object_entity} (enclosed by <Obj></Obj>) are subject and object entities and the relation label between {subject_entity} and {object_entity} entity is '{relation_labelword}', what are the types of {subject_entity} and {object_entity} and what is the relationship between these two entities can you inferred from the given sentence?
                                                The responses should adhere to the subsequent format without any supplementary information, explanations, or notes:
                                                1. The mention {subject_entity} is an entity of [Entity Type1].
                                                2. The mention {object_entity} is an entity of [Entity Type2].
                                                3. {subject_entity}, [Relationship Description between {subject_entity} and {object_entity}], {object_entity}.
                                                Do not utilize the terms 'subject entity' or 'object entity' as [Entity Type1] or [Entity Type2].
                                                Do not repeat the given relation label {relation_labelword} in the [Relationship Description] which should be described in natural language.
                                                Note that [ ] marks the place that should be filled with the right description.
                                                '''.format(**prompt_kwargs)
                        premise_induce_by_label_prompt = textwrap.dedent(premise_induce_by_label_prompt)

                        # # call openai
                        response = call_openai_engine(engine=args.engine,
                                                      api_key=args.api_key,
                                                      prompt=premise_induce_by_label_prompt,
                                                      temperature=0.0)
                        openai_generation, usage = get_openai_generation(args.engine, response)
                        new_cost = estimate_cost(usage, args.engine)
                        total_cost += new_cost
                        logger.info(f'New cost {new_cost}$, the total cost {total_cost}$')

                        prompt_track_file_handler.writelines(json.dumps({'prompt_type': 'premise_induce_by_label_prompt',
                                                                         'prompt': premise_induce_by_label_prompt,
                                                                         'response': openai_generation,
                                                                         **training_sample,
                                                                         }, ensure_ascii=False))
                        prompt_track_file_handler.write('\n')

                        # parse openai generation
                        try:
                            if prompt_kwargs['relation'] in openai_generation:
                                continue
                            if prompt_kwargs['subject_entity'].lower() in openai_generation.lower() and \
                                    prompt_kwargs['object_entity'].lower() in openai_generation.lower():
                                res = openai_generation[openai_generation.index('1.'):]
                                left = res.index(f"is an entity of") + len(f"is an entity of")
                                right = res.index('2.')

                                subject_pattern = remove_square_bracket_symbol(
                                    remove_dot_end_symbol(res[left:right].strip())).lower()
                                if 'type ' in subject_pattern:  # remove possible redundant 'type' word
                                    subject_pattern = subject_pattern.replace('type ', '').strip()
                                if ' type' in subject_pattern:  # remove possible redundant 'type' word
                                    subject_pattern = subject_pattern.replace(' type', '').strip()
                                if 'subject entity' in subject_pattern.lower():
                                    continue
                                if 'entity' in subject_pattern:
                                    subject_pattern = subject_pattern.replace('entity', '').strip()
                                if 'Entity' in subject_pattern:
                                    subject_pattern = subject_pattern.replace('Entity', '').strip()
                                if '\"' in subject_pattern:
                                    subject_pattern = subject_pattern.replace('\"', '').strip()
                                if '\'' in subject_pattern:
                                    subject_pattern = subject_pattern.replace('\'', '').strip()
                                prompt_kwargs['subject_pattern'] = subject_pattern.strip()

                                res = res[res.index('2.'):]
                                left = res.index(f"is an entity of") + len(f"is an entity of")
                                right = res.index('3.')
                                # remove possible [ ] symbol
                                object_pattern = remove_square_bracket_symbol(
                                    remove_dot_end_symbol(res[left:right].strip())).lower()
                                if 'type ' in object_pattern:  # remove possible redundant 'type' word
                                    object_pattern = object_pattern.replace('type ', '').strip()
                                if ' type' in object_pattern:  # remove possible redundant 'type' word
                                    object_pattern = object_pattern.replace(' type', '').strip()
                                if 'object entity' in object_pattern.lower():
                                    continue
                                if 'entity' in object_pattern:
                                    object_pattern = object_pattern.replace('entity', '').strip()
                                if 'Entity' in object_pattern:
                                    object_pattern = object_pattern.replace('Entity', '').strip()
                                if '\"' in object_pattern:
                                    object_pattern = object_pattern.replace('\"', '').strip()
                                if '\'' in object_pattern:
                                    object_pattern = object_pattern.replace('\'', '').strip()
                                prompt_kwargs['object_pattern'] = object_pattern.strip()

                                # find relation pattern
                                rel_pattern = remove_square_bracket_symbol(res[res.index('3.') + len('3.'):].strip())
                                if ',' in rel_pattern:
                                    rel_pattern = rel_pattern.split(',')[1]
                                # Find possible subject entity in relation description
                                (left, right), rel_pattern = find_word_in_text(prompt_kwargs['subject_entity'],
                                                                               rel_pattern)
                                if left is not None:
                                    rel_pattern = rel_pattern[:left] + f' subject entity ' + rel_pattern[right:]
                                # Find possible object entity in relation description
                                (left, right), rel_pattern = find_word_in_text(prompt_kwargs['object_entity'],
                                                                               rel_pattern.strip())
                                if left is not None:
                                    rel_pattern = rel_pattern[:left] + f' object entity ' + rel_pattern[right:]
                                if 'subject entity' not in rel_pattern:
                                    rel_pattern = 'subject entity ' + rel_pattern
                                if 'object entity' not in rel_pattern:
                                    rel_pattern = rel_pattern + ' object entity'

                                if '  ' in rel_pattern:
                                    rel_pattern = rel_pattern.replace('  ', ' ')
                                prompt_kwargs['rel_pattern'] = rel_pattern.strip().lower()
                                llm_gen_success = True
                        except:
                            logger.warning(
                                f'\nFail to parse the response for prompt:\n{premise_induce_prompt}\nThe response:\n{openai_generation}')
                            continue
                    pbar.update(1)
                    continue
                if not conc_valid:
                    # ------------------ Step 3: refine rule premise ------------------
                    llm_gen_success = False
                    current_tries = 0
                    while not llm_gen_success and current_tries < max_tries:
                        current_tries += 1

                        label_refine_prompt = ''
                        query = f"Sentence: {prompt_kwargs['sentence_ent_tag']} Subject entity {prompt_kwargs['subject_entity']}. Object entity: {prompt_kwargs['object_entity']}\n" + \
                                f"Question: Given that the correct relation label between '{prompt_kwargs['subject_entity']}' and '{prompt_kwargs['object_entity']}' is '{prompt_kwargs['relation_labelword']}', " + \
                                f"according to the sentence, try to revise and refine the following entity types of subject and object entities, and the relationship description between them to accurately reflect the semantics of true relation label '{prompt_kwargs['relation_labelword']}', rather than the false relation label: '{', '.join(prompt_kwargs['wrong_relation_labelword'])}'.\n"
                        query += f"1. The type of {prompt_kwargs['subject_entity']} is {prompt_kwargs['subject_pattern']}\n" + \
                                 f"2. The type of {prompt_kwargs['object_entity']} is {prompt_kwargs['object_pattern']}\n" + \
                                 f"3. {prompt_kwargs['rel_pattern'].replace('subject entity', prompt_kwargs['subject_entity']).replace('object entity', prompt_kwargs['object_entity'])}.\n"
                        query += f"The responses should adhere to the subsequent format without any supplementary information, explanations, or notes:\n" + \
                                 f"1. The type of {prompt_kwargs['subject_entity']} is [Correct Entity Type1]\n" + \
                                 f"2. The type of {prompt_kwargs['object_entity']} is [Correct Entity Type2]\n" + \
                                 f"3. [Correct Relationship Description between {prompt_kwargs['subject_entity']} and {prompt_kwargs['object_entity']}].\n" + \
                                 f"Note that [ ] marks the place that should be filled with the right description. Do not repeat the given relation label {prompt_kwargs['relation_labelword']} in the [Relationship Description] which should be described in natural language."
                        label_refine_prompt += query
                        # call openai
                        response = call_openai_engine(engine=args.engine,
                                                      api_key=args.api_key,
                                                      prompt=label_refine_prompt,
                                                      temperature=0.0)
                        openai_generation, usage = get_openai_generation(args.engine, response)
                        new_cost = estimate_cost(usage, args.engine)
                        total_cost += new_cost
                        logger.info(f'New cost {new_cost}$, the total cost {total_cost}$')

                        prompt_track_file_handler.writelines(json.dumps({'prompt_type': 'label_refine_prompt',
                                                                         'prompt': label_refine_prompt,
                                                                         'response': openai_generation,
                                                                         **training_sample,
                                                                         }, ensure_ascii=False))
                        prompt_track_file_handler.write('\n')
                        # parse openai generation
                        try:
                            if prompt_kwargs['relation'] in openai_generation:
                                continue
                            if prompt_kwargs['relation_labelword'] != rel2labelword[id2rel[NOTA_LABEL_ID]] and \
                                    prompt_kwargs['relation_labelword'].lower() in openai_generation.lower():
                                continue
                            if 'correct' in openai_generation.lower():
                                continue
                            if prompt_kwargs['subject_entity'].lower() in openai_generation.lower() and \
                                    prompt_kwargs['object_entity'].lower() in openai_generation.lower():
                                res = openai_generation[openai_generation.index('1.'):]
                                left = res.index(f"{prompt_kwargs['subject_entity']} is") + len(
                                    f"{prompt_kwargs['subject_entity']} is")
                                right = res.index('2.')

                                subject_pattern = remove_square_bracket_symbol(
                                    remove_dot_end_symbol(res[left:right].strip())).lower()
                                if 'type ' in subject_pattern:  # remove possible redundant 'type' word
                                    subject_pattern = subject_pattern.replace('type ', '').strip()
                                if ' type' in subject_pattern:  # remove possible redundant 'type' word
                                    subject_pattern = subject_pattern.replace(' type', '').strip()
                                if '\"' in subject_pattern:
                                    subject_pattern = subject_pattern.replace('\"', '').strip()
                                if '\'' in subject_pattern:
                                    subject_pattern = subject_pattern.replace('\'', '').strip()
                                if 'subject entity' in subject_pattern.lower():
                                    continue
                                prompt_kwargs['subject_pattern'] = subject_pattern.strip()

                                res = res[res.index('2.'):]
                                left = res.index(f"{prompt_kwargs['object_entity']} is") + len(
                                    f"{prompt_kwargs['object_entity']} is")
                                right = res.index('3.')
                                # remove possible [ ] symbol
                                object_pattern = remove_square_bracket_symbol(
                                    remove_dot_end_symbol(res[left:right].strip())).lower()
                                if 'type ' in object_pattern:  # remove possible redundant 'type' word
                                    object_pattern = object_pattern.replace('type ', '').strip()
                                if ' type' in object_pattern:  # remove possible redundant 'type' word
                                    object_pattern = object_pattern.replace(' type', '').strip()
                                if '\"' in object_pattern:
                                    object_pattern = object_pattern.replace('\"', '').strip()
                                if '\'' in object_pattern:
                                    object_pattern = object_pattern.replace('\'', '').strip()
                                if 'object entity' in object_pattern.lower():
                                    continue
                                prompt_kwargs['object_pattern'] = object_pattern.strip()

                                # find relation pattern
                                rel_pattern = remove_square_bracket_symbol(res[res.index('3.') + len('3.'):].strip())
                                # Find possible subject entity in relation description
                                (left, right), rel_pattern = find_word_in_text(prompt_kwargs['subject_entity'],
                                                                               rel_pattern)
                                if left is not None:
                                    rel_pattern = rel_pattern[:left] + f' subject entity ' + rel_pattern[right:]
                                # Find possible object entity in relation description
                                (left, right), rel_pattern = find_word_in_text(prompt_kwargs['object_entity'],
                                                                               rel_pattern.strip())
                                if left is not None:
                                    rel_pattern = rel_pattern[:left] + f' object entity ' + rel_pattern[right:]
                                if 'subject entity' not in rel_pattern:
                                    rel_pattern = 'subject entity ' + rel_pattern
                                if 'object entity' not in rel_pattern:
                                    rel_pattern = rel_pattern + ' object entity'

                                if '  ' in rel_pattern:
                                    rel_pattern = rel_pattern.replace('  ', ' ')
                                prompt_kwargs['rel_pattern'] = rel_pattern.strip().lower()
                                llm_gen_success = True
                        except:
                            logger.warning(
                                f'\nFail to parse the response for prompt:\n{label_refine_prompt}\nThe response:\n{openai_generation}')
                            continue

            # write the prediction results
            pr_file_handler.writelines(
                json.dumps({'relation': relation_label,
                            'subject_pattern': prompt_kwargs['subject_pattern'],
                            'object_pattern': prompt_kwargs['object_pattern'],
                            'rel_pattern': prompt_kwargs['rel_pattern'],
                            'subject_entity': prompt_kwargs['subject_entity'],
                            'object_entity': prompt_kwargs['object_entity'],
                            'sentence': prompt_kwargs['sentence'],
                            'sentence_ent_tag': prompt_kwargs['sentence_ent_tag'],
                            'token': training_sample['token'],
                            'h': training_sample['h'],
                            't': training_sample['t'],
                            }, ensure_ascii=False))
            pr_file_handler.write('\n')
            pbar.update(1)


    end_time = time.time()
    logger.info(f'Querying the LLM using: {end_time - start_time} seconds, {(end_time - start_time) / 60.0} minutes')
    logger.info(f'The total cost {total_cost}$, nearly {total_cost * 7.24}￥.')

