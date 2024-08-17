import json

import torch
from torch import Tensor
from pydantic.main import BaseModel
from transformers import (PreTrainedModel, PreTrainedTokenizerFast,
                          AutoModelForSeq2SeqLM, AutoTokenizer,
                          IntervalStrategy, Pipeline, TrainingArguments,
                          Seq2SeqTrainingArguments,
                          pipeline, set_seed)
from typing import Dict, List, Optional, Set, Tuple, Union
import re
import random
from pathlib import Path
from argparse import Namespace
from logging import Logger
from tqdm import tqdm

import src.slm.run_summarization as run_summarization
from src.slm.dataset import Sample, Dataset
from src.utils.utils import *


class DynamicModel(BaseModel):
    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True


class TextGenerator(DynamicModel):
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerFast
    # scores: Optional[List[Tensor]] = None
    max_target_length: int
    max_source_length: int

    def tokenize(self, texts: List[str], **kwargs):
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_source_length,
            return_tensors="pt",
            **kwargs,
        ).to(self.model.device)

    def run(
            self,
            texts: List[str],
            do_sample=True,
            top_k=50,
            temperature=1.0,
            num_return: int = 4,
            prompt: Optional[str] = None,
            prompt_ids: Optional[List[int]] = None,
            multi_prompt_ids: Optional[List[List[int]]] = None,
            decoder_input_ids: Optional[Tensor] = None,
            output_scores: bool = False,
            **kwargs,
    ) -> Dict:
        # https://huggingface.co/transformers/v4.7.0/main_classes/model.html#generation
        tok = self.tokenizer
        eos, bos = tok.eos_token_id, tok.bos_token_id

        if prompt is not None:
            prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
        if prompt_ids is not None:
            prompt_ids = [eos, bos] + prompt_ids
            decoder_input_ids = torch.tensor([prompt_ids])
        if multi_prompt_ids is not None:
            assert len(texts) == len(multi_prompt_ids)
            multi_prompt_ids = [[eos, bos] + lst for lst in multi_prompt_ids]
            decoder_input_ids = torch.tensor(multi_prompt_ids)
        if decoder_input_ids is not None:
            kwargs.update(decoder_input_ids=decoder_input_ids.to(self.model.device))

        outputs = self.model.generate(
            **self.tokenize(texts),
            do_sample=do_sample,
            top_k=top_k,
            temperature=temperature,
            num_return_sequences=num_return,
            return_dict_in_generate=True,
            output_scores=output_scores,
            max_length=self.max_target_length,
            **kwargs,
        )

        return {
            'outputs': self.decode(outputs),
            'confidences': self.get_confidence(outputs) if output_scores else None
        }

    def decode(self, outputs) -> List[str]:
        tok = self.tokenizer
        texts = tok.batch_decode(
            outputs.sequences, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )

        # Manually remove <bos><eos><pad> in case we have custom special tokens
        to_remove_token_list = list()
        if tok.bos_token:
            to_remove_token_list += [tok.bos_token]
        if tok.eos_token:
            to_remove_token_list += [tok.eos_token]
        if tok.pad_token:
            to_remove_token_list += [tok.pad_token]
        for i, t in enumerate(texts):
            for token in to_remove_token_list:
                t = t.replace(token, "")
                texts[i] = t
        return texts

    def get_confidence(self, outputs) -> List[dict]:
        all_scores = [_ for _ in torch.stack(outputs.scores, 1).cpu()]
        tok = self.tokenizer

        ignore_token_ids = []
        if tok.bos_token:
            ignore_token_ids += [tok.bos_token_id]
        if tok.eos_token:
            ignore_token_ids += [tok.eos_token_id]
        if tok.pad_token:
            ignore_token_ids += [tok.pad_token_id]
        ignore_token_ids += [tok.convert_tokens_to_ids('.')]
        subj_ignore_ids = tok.encode('Subject entity type:')[:-1]
        obj_ignore_ids = tok.encode('Object entity type:')[:-1]
        rel_ignore_ids = tok.encode('Relationship:')[:-1]
        label_ignore_ids = tok.encode('Label:')[:-1]
        confidences = list()
        for sample_id, scores in enumerate(all_scores):
            total_probs, pattern_probs, label_probs = 0.0, 0.0, 0.0
            total_counter, pattern_counter, label_counter = 0, 0, 0
            output_token_ids = outputs.sequences[sample_id].tolist()[1:]
            pattern_label_split = len(output_token_ids)
            matches = list()  # fixed tokens that should be ignored
            for i in range(len(output_token_ids)):
                if output_token_ids[i] == subj_ignore_ids[0]:
                    if i + len(subj_ignore_ids) < len(output_token_ids) and output_token_ids[i:i + len(
                            subj_ignore_ids)] == subj_ignore_ids:
                        matches += list(range(i, i + len(subj_ignore_ids)))
                if output_token_ids[i] == obj_ignore_ids[0]:
                    if i + len(obj_ignore_ids) < len(output_token_ids) and output_token_ids[
                                                                           i:i + len(obj_ignore_ids)] == obj_ignore_ids:
                        matches += list(range(i, i + len(obj_ignore_ids)))
                if output_token_ids[i] == rel_ignore_ids[0]:
                    if i + len(rel_ignore_ids) < len(output_token_ids) and output_token_ids[
                                                                           i:i + len(rel_ignore_ids)] == rel_ignore_ids:
                        matches += list(range(i, i + len(rel_ignore_ids)))
                if output_token_ids[i] == label_ignore_ids[0]:
                    if i + len(label_ignore_ids) < len(output_token_ids) and output_token_ids[i:i + len(
                            label_ignore_ids)] == label_ignore_ids:
                        matches += list(range(i, i + len(label_ignore_ids)))
                        pattern_label_split = i + len(label_ignore_ids)
            for i, token_id in enumerate(output_token_ids):
                if token_id in ignore_token_ids or i in matches:
                    continue
                else:
                    probs = torch.softmax(scores[i], dim=0)
                    total_probs += probs[token_id].tolist()
                    total_counter += 1
                    if i < pattern_label_split:
                        pattern_probs += probs[token_id].tolist()
                        pattern_counter += 1
                    else:
                        label_probs += probs[token_id].tolist()
                        label_counter += 1
            confidences.append({
                'total_confidence': total_probs / total_counter if total_counter != 0 else 0,
                'pattern_confidence': pattern_probs / pattern_counter if pattern_counter != 0 else 0,
                'label_confidence': label_probs / label_counter if label_counter != 0 else 0,
            })
        return confidences

    def get_sentence_embeddings(self, texts) -> torch.Tensor:
        bsz = 128 if torch.cuda.is_available() else 16
        with torch.no_grad():
            all_sent_embs = None
            if len(texts) <= bsz:
                tok_res = self.tokenize(texts, return_special_tokens_mask=True)
                enc_output = self.model.encoder(
                    input_ids=tok_res["input_ids"],
                    attention_mask=tok_res["attention_mask"],
                    return_dict=True,
                )
                # get the final hidden states
                last_hidden_state = enc_output.last_hidden_state  # bsz * seq_len * emb_len
                all_sent_embs = last_hidden_state * tok_res['attention_mask'].unsqueeze(2)
                # Default, represent sentences with averaged last hidden states
                all_sent_embs = all_sent_embs.sum(dim=1) / tok_res['attention_mask'].sum(dim=1, keepdim=True)
            else:
                for i in tqdm(range(0, len(texts), bsz), desc='Encoding rules'):
                    batch = texts[i:i + bsz]
                    tok_res = self.tokenize(batch, return_special_tokens_mask=True)
                    enc_output = self.model.encoder(
                        input_ids=tok_res["input_ids"],
                        attention_mask=tok_res["attention_mask"],
                        return_dict=True,
                    )
                    # get the final hidden states
                    last_hidden_state = enc_output.last_hidden_state  # bsz * seq_len * emb_len
                    # sent_embs = last_hidden_state * (tok_res["special_tokens_mask"] + tok_res['attention_mask']).unsqueeze(2)
                    sent_embs = last_hidden_state * tok_res['attention_mask'].unsqueeze(2)
                    # Default, represent sentences with averaged last hidden states
                    sent_embs = sent_embs.sum(dim=1) / tok_res['attention_mask'].sum(dim=1, keepdim=True)
                    if all_sent_embs is None:
                        all_sent_embs = sent_embs
                    else:
                        all_sent_embs = torch.concat([all_sent_embs, sent_embs], dim=0)
        return all_sent_embs


class T5Model(DynamicModel):
    resume_from_save_dir: bool = False
    gen: Optional[TextGenerator] = None
    save_dir: str = ""  # argument save_dir, if not set in the initialization, will be assigned when performing fit()
    model_name: str = "flan-t5-large"  # path or name of initial pretrained model
    do_pretrain: bool = False  # default: False
    pipe_name: str = "summarization"
    batch_size: int = 8 if torch.cuda.is_available() else 2  # small batch size for running on CPU
    grad_accumulation: int = 2
    random_seed: int = 42
    warmup_ratio: float = 0.2
    lr_pretrain: float = 3e-4
    lr_finetune: float = 3e-5
    epochs_pretrain: int = 3
    epochs_finetune: int = 30
    train_fp16: bool = True  # should be false when running at CPU

    max_source_length: int = 256
    max_target_length: int = 128

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.resume_from_save_dir and self.save_dir is not None:
            self.gen = TextGenerator(
                model=AutoModelForSeq2SeqLM.from_pretrained(self.save_dir),
                tokenizer=AutoTokenizer.from_pretrained(self.save_dir),
                max_source_length=self.max_source_length,
                max_target_length=self.max_target_length,
            )
            self.gen.model.to(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        else:
            self.gen = None

    def get_lr(self) -> float:
        return self.lr_pretrain if self.do_pretrain else self.lr_finetune

    def get_epochs(self) -> int:
        return self.epochs_pretrain if self.do_pretrain else self.epochs_finetune

    def get_train_args(self, do_eval: bool) -> TrainingArguments:
        # default: using loss metric on validation set to select best LM
        return Seq2SeqTrainingArguments(
            seed=self.random_seed,
            do_train=True,
            do_eval=do_eval or None,  # False still becomes True after parsing
            overwrite_output_dir=True,
            per_device_train_batch_size=self.batch_size,
            # per_device_eval_batch_size=self.batch_size,
            per_device_eval_batch_size=96 if torch.cuda.is_available() else 2,
            # gradient_accumulation_steps=self.grad_accumulation,
            warmup_ratio=self.warmup_ratio,
            output_dir=self.save_dir,
            save_strategy=IntervalStrategy.EPOCH,  # NO = "no", STEPS = "steps", EPOCH = "epoch"
            evaluation_strategy=IntervalStrategy.EPOCH if do_eval else IntervalStrategy.NO,
            logging_strategy=IntervalStrategy.EPOCH,
            learning_rate=self.get_lr(),
            num_train_epochs=self.get_epochs(),
            load_best_model_at_end=True,
            # fp16=self.train_fp16 if torch.cuda.is_available() else False,
            fp16=False,
            save_total_limit=2,
            # report_to=['wandb'],
            report_to=['none'],
            predict_with_generate=True,
            metric_for_best_model='eval_micro_f1',
            greater_is_better=True,
            generation_max_length=self.max_target_length
        )

    # build a class of `transformers.Pipeline` which includes a tokenizer and a model
    def make_pipe(self, **kwargs) -> Pipeline:
        pipe = pipeline(
            self.pipe_name,
            model=self.save_dir,
            tokenizer=self.save_dir,
            device=0 if torch.cuda.is_available() else -1,
            **kwargs,
        )
        return pipe

    def write_data(self, args, data: Dataset, name: str) -> str:
        path_out = Path(f"src/slm/data/{args.dataset}/{args.k_shot_lre}-{args.k_shot_id}/{name}_{args.exp_name}.json")
        path_out.parent.mkdir(exist_ok=True, parents=True)
        lines = [s.encode_to_line() for s in data.samples]
        if name in ['train']:
            random.seed(args.random_seed)
            random.shuffle(lines)
        with open(path_out, "w") as f:
            f.write("".join(lines))
        return str(path_out)

    def fit(self, args: Namespace, logger: Logger):
        save_dir = Path(f"{args.model_save_dir}/{args.exp_name}")
        save_dir.parent.mkdir(exist_ok=True, parents=True)
        self.save_dir = str(save_dir)

        if args.train_file is None or args.train_file == '':
            # if no train_file, use default files
            with open(f"src/configs/rule_train_config.json", "r", encoding="utf-8") as f:
                train_paths = json.load(f)
            if args.lre_setting == 'k-shot':
                path_train = f"outputs/{args.dataset}/k-shot/{args.k_shot_lre}-{args.k_shot_id}/pr/" + \
                             train_paths[args.dataset]['k-shot'][f'{args.k_shot_lre}-{args.k_shot_id}']
        else:
            if args.lre_setting == 'k-shot':
                path_train = f"outputs/{args.dataset}/k-shot/{args.k_shot_lre}-{args.k_shot_id}/pr/" + \
                             args.train_file
        if args.valid_file is None or args.valid_file == '':
            with open(f"src/configs/rule_val_config.json", "r", encoding="utf-8") as f:
                valid_paths = json.load(f)
            if args.lre_setting == 'k-shot':
                path_valid = f"outputs/{args.dataset}/k-shot/{args.k_shot_lre}-{args.k_shot_id}/pr/" + \
                             valid_paths[args.dataset]['k-shot'][f'{args.k_shot_lre}-{args.k_shot_id}']
        else:
            if args.lre_setting == 'k-shot':
                path_valid = f"outputs/{args.dataset}/k-shot/{args.k_shot_lre}-{args.k_shot_id}/pr/" + \
                             args.valid_file

        data_train = Dataset.load(path_train, max_count=args.debug_test_num - 1 if args.mode == 'debug' else None)
        data_valid = Dataset.load(path_valid, max_count=args.debug_test_num - 1 if args.mode == 'debug' else None)
        # transform original data format into the text-to-text format
        path_train = self.write_data(args, data_train, "train")  # slm/data/{dataset}/{k_shot}-{k_shot_id}
        path_valid = self.write_data(args, data_valid, "valid")

        # kwargs = {'tokenizer_kwargs': {'additional_special_tokens': ['']}}
        kwargs = {}

        data_args = run_summarization.DataTrainingArguments(
            train_file=path_train,
            validation_file=path_valid,
            overwrite_cache=True,
            max_source_length=self.max_source_length,
            max_target_length=self.max_target_length,
            **kwargs,
        )
        train_args = self.get_train_args(do_eval=path_valid is not None)
        kwargs = {
            k: v for k, v in train_args.to_dict().items() if not k.startswith("_")
        }
        train_args = run_summarization.Seq2SeqTrainingArguments(**kwargs)
        model_args = run_summarization.ModelArguments(
            model_name_or_path=self.model_name
        )
        run_summarization.main(
            model_args=model_args,
            training_args=train_args,
            data_args=data_args,
            exp_args=args,
            logger=logger,
        )
        delete_checkpoints(self.save_dir)

    def predict(self, path_in: str, path_out: str, args: Namespace, logger: Logger) -> List:
        data = Dataset.load(path_in, max_count=args.debug_test_num - 1 if args.mode == 'debug' else None)
        logger.info(f"Inference samples within the file {path_in}.")
        logger.info(f"Loading pretrained model from {self.save_dir}")

        if self.gen is None:
            self.gen = TextGenerator(
                model=AutoModelForSeq2SeqLM.from_pretrained(self.save_dir),
                tokenizer=AutoTokenizer.from_pretrained(self.save_dir),
                max_source_length=self.max_source_length,
                max_target_length=self.max_target_length,
            )
            self.gen.model.to(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        pred_rules = []

        Path(path_out).parent.mkdir(exist_ok=True, parents=True)
        logger.info(f"Saving the generation results into the file {path_out}.")
        file_handler = open(path_out, 'a')
        bsz = 96 if torch.cuda.is_available() else 16
        for i in tqdm(range(0, len(data.samples), bsz), desc='Predicting rules via SLM'):
            batch = data.samples[i:i + bsz]
            texts = [s.encode_to_x() for s in batch]
            batch_res = self.gen.run(
                texts,
                do_sample=False,
                num_return=1,
                num_beams=1,
                output_scores=True
            )
            for j, raw in enumerate(batch_res['outputs']):
                raw = raw.strip()
                decode_res = Sample.decode(raw)
                file_handler.writelines(json.dumps({
                    **(batch[j].to_dict()),
                    'slm_input_text': texts[j],
                    'slm_pred_rule': raw,
                    'subject_pattern': decode_res['subject_pattern'],
                    'object_pattern': decode_res['object_pattern'],
                    'rel_pattern': decode_res['rel_pattern'],
                    'premise': decode_res['premise'],
                    'pred_relation': decode_res['pred_relation'],
                    'confidence': batch_res['confidences'][j],
                },
                    ensure_ascii=False))
                file_handler.write('\n')
                pred_rules.append(decode_res)
        return pred_rules

    def get_sentence_embeddings(self, texts) -> torch.Tensor:
        return self.gen.get_sentence_embeddings(texts=texts)
