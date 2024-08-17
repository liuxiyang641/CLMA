import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import logging
import logging.handlers
import argparse
import time
import json
from src.slm.t5_model import T5Model
from pathlib import Path
from src.utils.utils import *


def get_logger(args: argparse.Namespace):
    post_processing_args(args)
    if args.lre_setting == 'k-shot':
        log_filename = "src/slm/logs/k-shot/" + args.exp_name
        if not os.path.exists("src/slm/logs/k-shot/"):
            os.makedirs("src/slm/logs/k-shot/")
    # Remove all handlers associated with the root logger object.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    rfh = logging.handlers.RotatingFileHandler(
        filename=log_filename,
        maxBytes=100 * 1024 * 1024,  # max 100M
        backupCount=0,
    )

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        # filename=log_filename,
                        level=logging.INFO,
                        handlers=[
                            rfh
                        ]
                        )

    return logging.getLogger(__name__)


def post_processing_args(args: argparse.Namespace):
    if args.lre_setting == 'k-shot':
        if args.mode == 'debug':
            if args.fix_exp_name:
                args.exp_name = f'debug_{args.exp_name}'
            else:
                args.exp_name = f'debug_{args.exp_name}_{args.dataset}_{args.k_shot_lre}-{args.k_shot_id}_{time.strftime("%m_%d_%H_%M_%S")}'
        else:
            if not args.fix_exp_name:
                args.exp_name = f'{args.exp_name}_{args.dataset}_{args.k_shot_lre}-{args.k_shot_id}_{time.strftime("%m_%d_%H_%M_%S")}'
        if not os.path.exists("src/slm/logs/"):
            os.makedirs("src/slm/logs/")
        args.output_dir = f"{args.output_dir}/{args.dataset}/k-shot/{args.k_shot_lre}-{args.k_shot_id}"
        args.model_save_dir = f"{args.model_save_dir}/{args.dataset}/k-shot/{args.k_shot_lre}-{args.k_shot_id}"
    print(args)


def slm_args_parser():
    parser = argparse.ArgumentParser(description='SLM for rule learning')
    # general args
    parser.add_argument('--exp_name', type=str, default='test')
    parser.add_argument('--fix_exp_name', type=bool, default=False)
    parser.add_argument('--method', type=str, default='vanilla')
    parser.add_argument('--random_seed', type=int, default=42)

    # dataset args
    parser.add_argument('--dataset', type=str, default='tacrev')
    parser.add_argument('--train_file', type=str, default=None,
                        help="The training file in outputs/{args.dataset}/{args.k_shot_lre}-{args.k_shot_id}/pr/")
    parser.add_argument('--valid_file', type=str, default=None,
                        help="The validation file in outputs/{args.dataset}/{args.k_shot_lre}-{args.k_shot_id}/pr/")
    parser.add_argument('--output_dir', type=str, default='src/slm/outputs',
                        help="The output directory of generated data.")
    parser.add_argument('--model_save_dir', type=str, default='src/slm/ckpt',
                        help="The output directory of model checkpoints and evaluation results.")
    parser.add_argument('--mode', type=str, choices=['run', 'debug'], default='run',
                        help="The mode of executing code")
    parser.add_argument('--debug_test_num', type=int, default=5,
                        help="The test samples of debug mode")

    parser.add_argument('--lre_setting', type=str, default='k-shot', choices=['k-shot'],
                        help="The low-resource setting of labeled data")
    parser.add_argument('--k_shot_lre', default=8, type=int, choices=[8, 16, 32],
                        help='K training samples for each relation, under the low resource relation extraction task')
    parser.add_argument('--k_shot_id', default=1, type=int, choices=[1, 2, 3, 4, 5],
                        help='The id number of k-shot training')
    # environment args
    return parser


def slm_finetune(args, logger, eval_on_unlabeled_data=True):
    model = T5Model()
    model.fit(args, logger)

    with open(f"src/configs/data_config.json", "r", encoding="utf-8") as f:
        data_configs = json.load(f)
        rel2id_file_path = data_configs[args.dataset]['rel2id']
        if args.lre_setting == 'k-shot':
            # load k-shot train file
            train_file_path = data_configs[args.dataset]['k-shot'] + f'{args.k_shot_lre}-{args.k_shot_id}/train.json'
            val_file_path = data_configs[args.dataset]['k-shot'] + f'{args.k_shot_lre}-{args.k_shot_id}/val.json'
        test_file_path = data_configs[args.dataset]['test']

    logger.info('-------------Train set---------------')
    path_out = Path(f"{args.output_dir}/{args.exp_name}/pred_train_rule_{args.exp_name}.json")
    path_out.parent.mkdir(exist_ok=True, parents=True)
    path_out = str(path_out)
    model.predict(path_in=train_file_path, path_out=path_out, args=args, logger=logger)
    logger.info('-------------Valid set---------------')
    path_out = f"{args.output_dir}/{args.exp_name}/pred_val_rule_{args.exp_name}.json"
    model.predict(path_in=val_file_path, path_out=path_out, args=args, logger=logger)
    logger.info('-------------Full test set---------------')
    path_out = f"{args.output_dir}/{args.exp_name}/pred_full_test_rule_{args.exp_name}.json"
    model.predict(path_in=f"data/{args.dataset}/test.json", path_out=path_out, args=args, logger=logger)
    slm_pred_file_evaluation(path_out, rel2id_file_path, logger)

    if not eval_on_unlabeled_data:
        return

    logger.info('-------------Unlabeled train set---------------')
    if args.lre_setting == 'k-shot':
        # load k-shot unlabeled file
        unlabeled_file_path = data_configs[args.dataset]['k-shot'] + \
                              f'{args.k_shot_lre}-{args.k_shot_id}/unlabeled_train.json'
    if args.lre_setting == 'k-shot':
        output_dir = os.path.join('outputs', f"{args.dataset}/k-shot/{args.k_shot_lre}-{args.k_shot_id}")
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
    labeled_data_w_rule, labeled_data_w_rule_by_rel = read_data_file(train_rule_path,
                                                                     max_count=args.debug_test_num if args.mode == 'debug' else None)
    logger.info(f'Loading rules from the file: {train_rule_path}, {len(labeled_data_w_rule)} induced rules.')

    # load unlabeled data
    unlabeled_data, _ = read_data_file(unlabeled_file_path,
                                       max_count=args.debug_test_num - 1 if args.mode == 'debug' else None)
    path_in = unlabeled_file_path
    saved_slm_pred_paths = f"{output_dir}/slm/unlabeled_{args.exp_name}_pred_rule.json"
    model.predict(path_in=path_in, path_out=saved_slm_pred_paths, args=args, logger=logger)
    slm_pred_file_evaluation(saved_slm_pred_paths, rel2id_file_path, logger)
    slm_pred_rules, _ = read_data_file(saved_slm_pred_paths,
                                       max_count=args.debug_test_num - 1 if args.mode == 'debug' else None)

    # obtain rule embeddings of labeled data
    def rule_verbalization(ins):
        return f"Subject entity type: {ins['subject_pattern']}. Object entity type: {ins['object_pattern']}. " + \
            f"Relationship: subject entity {ins['pred_rel_pattern']} object entity."
    match_degree_method = 'PATTERN2PATTERN'
    if match_degree_method == 'PREMISE2PERMISE':
        # obtain rule embeddings of unlabeled data
        unlabeled_rule_embs = model.get_sentence_embeddings(
            texts=[x['premise'] if x['premise'] is not None else '' for x in slm_pred_rules])
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
            texts=[x['subject_pattern'].lower() if x['subject_pattern'] is not None else '' for x in slm_pred_rules])
        unlabeled_obj_pattern_embs = model.get_sentence_embeddings(
            texts=[x['object_pattern'].lower() if x['object_pattern'] is not None else '' for x in slm_pred_rules])
        unlabeled_rel_pattern_embs = model.get_sentence_embeddings(
            texts=[x['rel_pattern'].lower() if x['rel_pattern'] is not None else '' for x in slm_pred_rules])
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
        unlabeled_sens = [x['sentence_ent_tag'] for x in slm_pred_rules]
        unlabeled_sen_embs = model.get_sentence_embeddings(texts=unlabeled_sens)
        rule_sim_scores = compute_cos_sim(unlabeled_sen_embs, labeled_sen_embs)

    #################
    # # output matched rules
    for sample_id, unlabeled_sample in enumerate(unlabeled_data):
        unlabeled_sample['sentence_ent_tag'] = slm_pred_rules[sample_id]['sentence_ent_tag']
        unlabeled_sample['slm_pred_rule'] = slm_pred_rules[sample_id]['slm_pred_rule']
        unlabeled_sample['confidence'] = slm_pred_rules[sample_id]['confidence']
        unlabeled_sample['subject_pattern'] = slm_pred_rules[sample_id]['subject_pattern']
        unlabeled_sample['object_pattern'] = slm_pred_rules[sample_id]['object_pattern']
        unlabeled_sample['rel_pattern'] = slm_pred_rules[sample_id]['rel_pattern']
        unlabeled_sample['pred_relation'] = slm_pred_rules[sample_id]['pred_relation']
        top_k_similar_rule_ids = torch.sort(rule_sim_scores[sample_id],
                                            descending=True).indices[:5].cpu().tolist()
        for i, rule_id in enumerate(top_k_similar_rule_ids):
            unlabeled_sample[f'top{i + 1}_matched_rule'] = {
                'premise': labeled_rule_premise[rule_id],
                'relation': labeled_rule_relation[rule_id],
                'confidence': rule_sim_scores[sample_id][rule_id].tolist(),
            }
    rule_match_file = os.path.join(output_dir, f"slm/{args.exp_name}_rule_match.json")
    rule_match_file_handler = open(rule_match_file, 'a')
    for data in unlabeled_data:
        # save the labeled and unlabeled data
        rule_match_file_handler.writelines(json.dumps({**data}, ensure_ascii=False))
        rule_match_file_handler.write('\n')
    del model


if __name__ == '__main__':
    # Some basic settings
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_path = os.path.dirname(os.path.dirname(current_dir))
    os.chdir(root_path)

    args = slm_args_parser().parse_args()
    logger = get_logger(args)
    logger.info(args)
    slm_finetune(args, logger)

