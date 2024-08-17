from re_exps import *


def llm_re_openai(args, logger):
    """
    Using OpenAI GPT to finish task
    """
    # loading original data
    if args.method == 'rule_induction':
        rule_induction(args, logger)
    elif args.method == 're_rule_slm_icl':
        llm_inference(args, logger)
    elif args.method == 'collaborative_da':
        collaborative_da(args, logger)
    else:
        raise ValueError('Unknown method!')

