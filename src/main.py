import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import traceback
import logging
import logging.handlers
import warnings
import json
import argparse
from llm_re import *


def get_logger(args):
    if args.lre_setting == 'k-shot':
        if args.mode == 'debug':
            args.exp_name = f'debug_{args.exp_name}_{args.dataset}_{args.k_shot_lre}-{args.k_shot_id}_{args.remote_server}_{time.strftime("%m_%d_%H_%M_%S")}'
        elif args.mode == 'run':
            args.exp_name = f'{args.exp_name}_{args.dataset}_{args.k_shot_lre}-{args.k_shot_id}_{args.remote_server}_{time.strftime("%m_%d_%H_%M_%S")}'
        log_filename = "logs/k-shot/" + args.exp_name   # keep the rerun log file
        if not os.path.exists("logs/k-shot/"):
            os.makedirs("logs/k-shot/")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    print(args)

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


def parse_argument():
    parser = argparse.ArgumentParser(description='LLM for Relation Extraction')
    # general args
    parser.add_argument('--exp_name', type=str, default='test')
    parser.add_argument('--method', type=str, default='vanilla')
    parser.add_argument('--remote_server', type=str, default='openai',
                        choices=['openai'])  # default remote server, querying OpenAI
    parser.add_argument('--engine', type=str, default='gpt-3.5-turbo')
    parser.add_argument('--api_key', '-ak', type=str, required=True)
    parser.add_argument('--random_seed', type=int, default=42)

    # dataset args
    parser.add_argument('--dataset', type=str, default='tacrev')
    parser.add_argument('--output_dir', type=str, default='outputs', help="The output directory of generated data.")

    # experiment args
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'])
    parser.add_argument('--mode', type=str, choices=['run', 'debug', 'rerun'], default='debug',
                        help="The mode of executing code")
    parser.add_argument('--in_context_size', type=int, default=3)  # number of demonstrations, 0 for zero-shot setting
    parser.add_argument('--debug_test_num', type=int, default=5,
                        help="The test samples of debug mode")
    parser.add_argument('--lre_setting', type=str, default='k-shot', choices=['k-shot'],
                        help="The low-resource setting of labeled data")
    parser.add_argument('--k_shot_lre', default=8, type=int, choices=[8, 16, 32],
                        help='K training samples for each relation, under the low resource relation extraction task')
    parser.add_argument('--k_shot_id', default=1, type=int, choices=[1, 2, 3, 4, 5],
                        help='The id number of k-shot training')

    return parser.parse_args()


if __name__ == '__main__':
    # Some basic settings
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_path = os.path.dirname(current_dir)
    os.chdir(root_path)

    args = parse_argument()
    logger = get_logger(args)
    logger.info(args)
    if args.remote_server == 'openai':
        llm_re_openai(args, logger)
    else:
        raise Exception('Unknown remote server!')
