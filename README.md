# Collaborative Language Models for Data Annotation

## Datasets

- TACRED: [site](https://catalog.ldc.upenn.edu/LDC2018T24).
- Re-TACRED: [site](https://github.com/gstoica27/Re-TACRED).
- TACREV: [site](https://github.com/DFKI-NLP/tacrev).
- SemEval: [site](https://huggingface.co/datasets/sem_eval_2010_task_8).

Download the following dataset files into the `data/` directory. Then, generate the K-shot samples for the training and validation sets using the script provided by the previous work.

```shell
python generate_k_shot.py --k 8 --dataset tacrev --data_file train.txt
python generate_k_shot.py --k 8 --dataset tacrev --data_file val.txt
python generate_k_shot.py --k 16 --dataset tacrev --data_file train.txt
python generate_k_shot.py --k 16 --dataset tacrev --data_file val.txt
```

## Dependency

```shell
conda create -n CLMA python=3.7.16
conda activate CLMA
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 -c pytorch
pip install transformers datasets tenacity openai wandb scikit-learn pandas
pip install accelerate -U
```

## Logical Rule Induction

1, Obtaining rules from training set

Change the `EXP_METHOD` to `run_induction` in `run.sh` and set your openai API Key, then execute the script.

```shell
DATASET_NAME="tacrev"
EXP_METHOD="run_induction"
LRE_SETTING="k-shot"
K_SHOT_LRE="8"
K_SHOT_ID="1"

python main.py ... --api_key Your_API_Key
```

```shell
bash run.sh
```

The induced rules will be saved at `outputs/${DATASET_NAME}/${K_SHOT_LRE}/k-shot/${K_SHOT_ID}/pr/` directory.

The rules are saved in `{Your experiment name}.json` file. Assign the name of generated rule file to settings in `src/configs/rule_trian_config.json`, e.g.

```json
{
    "tacrev": {
        "k-shot": {
            "8-1": "{Your experiment name}.json"
        }
    }
}
```

2, Obtaining rules from validation set

Set the `split` hyperparameter to `val`,

```shell
python main.py ... --split val 
```

Then, run the script.

```shell
bash run.sh
```

Similarly, assign the name of generated rule file to settings in `src/configs/rule_val_config.json`.

## Fine-tuning SLM for Rule Generation

Then, run the script for fine-tuning SLM.

```shell
cd slm/
bash run.sh
```

The checkpoint of SLM is saved in `src/slm/ckpt/${DATASET_NAME}/${K_SHOT_LRE}/k-shot/${K_SHOT_ID}/`.

Set the checkpoint of SLM to settings in `src/configs/slm_ckpt_config.json`. 

## Collaborative Data Annotation

Change the `EXP_METHOD` to `run_induction`, then run the `run.sh`

```shell
EXP_METHOD="collaborative_da"
```

```shell
bash run.sh
```

## Utilizing the LLM Prompted by Rules for Testing

```shell
EXP_METHOD="llm_inference"
```

```shell
bash run.sh
```