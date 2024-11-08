# Cochrane-auto

This branch contains code and resources for training and evaluating the simplification systems in our paper Cochrane-auto: An Aligned Dataset for the Simplification of Biomedical Abstracts.
It is based on this [repository](https://github.com/liamcripwell/plan_simp).

We changed the original code by implementing early stopping into the training procedure, adding support for the merge operator, adding Rouge-L to the evaluation metrics and fixing some minor errors. Moreover, we provide detailed documentation on how to train and evaluate each model below.

## Installation

```bash
git clone -b cochrane-auto https://github.com/JanB100/doc_simp.git
cd doc_simp
conda create -n doc_simp python=3.10
conda activate doc_simp
pip install -r requirements.txt
pip install -e .
```

## Pretrained models
We share all Cochrane-auto pretrained models and the baseline model on [HuggingFace](https://huggingface.co/janbakker). To leverage these models, simply follow the instructions below and set the model path to the HuggingFace model name, e.g. janbakker/bartsent-cochraneauto.

## Data
All preprocessed Cochrane-auto data was copied from [this repository](https://github.com/JanB100/cochrane-auto) and placed into the [data](data) directory.

## Training the planning model
The script below can be used to train the classifier on 2 GPUs.

```bash
python plan_simp/scripts/train_clf.py \
  --name=classifier \
  --project=planning_models \
  --train_file=data/cochraneauto_sents_train.csv \
  --val_file=data/cochraneauto_sents_val.csv \
  --x_col=complex \
  --y_col=label \
  --batch_size=32 \
  --learning_rate=1e-5 \
  --ckpt_metric=val_macro_f1 \
  --hidden_dropout_prob=0.1 \
  --max_epochs=10 \
  --devices=2 \
  --use_merge_labels \
```

## Evaluating the planning model
The command below can be used to generate a plan with a classifier and evaluate it.

```bash
mkdir results

python plan_simp/scripts/eval_clf.py \
    <path_to_planning_model> \
    data/cochraneauto_sents_test.csv \
    --out_file=results/<model_name>.csv \
    --use_merge_labels \
```

## Training the simplification models
The script below shows how to train a text-only BART model.

```bash
python plan_simp/scripts/train_bart.py \
  --name=<model_name> \ #can be any name
  --project=simplification_models \
  --train_file=data/cochrane(auto)_<docs/para/sents>_train.csv \
  --val_file=data/cochrane(auto)_<docs/para/sents>_val.csv \
  --x_col=complex \
  --y_col=simple \
  --batch_size=8 \
  --accumulate_grad_batches=2 \
  --lr=2e-5 \
  --devices=2 \
  --skip_val_gen \
  --sent_level \ #only if it is a sentence-level model
```

To train a plan-guided model, add the following argument:

```bash
  --op_col=label \
```

## Generating simplifications
Use the script below to perform inference with a text-only model.

```bash
python plan_simp/scripts/generate.py inference \
  --model_ckpt=<path_to_simplification_model> \
  --test_file=data/cochrane(auto)_<docs/para/sents>_test.csv \
  --out_file=results/<model_name>.csv \
```

To perform inference with a plan-guided model, adjust the script as follows.

If it is guided by an oracle plan:

```bash
  --op_col=label \
```

If it is guided by a generated plan:

```bash
  --test_file=results/<classifier_name>.csv \
  --op_col=pred_l \
```

## Evaluating the simplifications

Download the [BARTScore](https://github.com/neulab/BARTScore/tree/main) model [here](https://drive.google.com/file/d/1_7JfF7KOInb7ZrxKHIigTMR4ChVET01m/view).

Use the command below to evaluate document-level outputs.

```bash
python plan_simp/scripts/eval_simp.py \
  --input_data=results/<model_name>.csv \
  --x_col=complex \
  --r_col=simple \
  --y_col=pred \
  --prepro=True \
  --bartscore_path=bart_score.pth \
```

To evaluate sentence- and paragraph-level outputs, use this command instead.

```bash
python plan_simp/scripts/eval_simp.py \
  --input_data=data/cochrane(auto)_docs_test.csv \
  --output_data=results/<model_name>.csv \
  --x_col=complex \
  --r_col=simple \
  --y_col=pred \
  --doc_id_col=pair_id \
  --prepro=True \
  --sent_level=True \
  --bartscore_path=bart_score.pth \
```
