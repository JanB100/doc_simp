# doc_simp

This repo contains code and resources for the paper Beyond Sentence-Level Text Simplification (Reproducibility Study of Context-Aware Document Simplification).
It is based on the [repository](https://github.com/liamcripwell/plan_simp) for the original paper.

We changed the original code by implementing early stopping into the training procedure, adding Rouge-L to the evaluation metrics and fixing some minor errors. We also added our code for constructing the paragraph-level Wiki-auto datasets. Moreover, we provide detailed documentation on how to train and evaluate each model below.

## Installation

```bash
git clone https://github.com/JanB100/doc_simp.git
cd doc_simp
conda create -n doc_simp python=3.9
conda activate doc_simp
pip install -r requirements.txt
pip install -e .
```

## Pretrained models
We share all Wiki-auto pretrained models on [HuggingFace](https://huggingface.co/janbakker). The original authors also made some of their Newsela-auto pretrained models available on [HuggingFace](https://huggingface.co/liamcripwell). To leverage these models, simply follow the instructions below and set the model path to the HuggingFace model name, e.g. janbakker/conbart.

## Data
The preprocessed Wiki-auto datasets shared by the original authors can be downloaded [here](https://drive.google.com/file/d/1lU8htUIVBuuU24HrPErpV01hlA6tc-d1/view?usp=sharing).
The paragraph-level data was constructed using [this script](data/paragraph_alignment.py), and can be downloaded [here](https://drive.google.com/file/d/1ZeALAhdWBfVNsFlRnPsGGia4NQB1Dbjq/view?usp=sharing).
All files should be placed into the data directory.

## Preparing context representations
Start by generating the context encodings of all complex and simple sentences. These are used by the contextual classifiers and ConBART.

```bash
mkdir context
```
```python
from plan_simp.scripts.encode_contexts import encode

for split in ["train", "valid", "test"]:
    for x in ["complex", "simple"]:
        encode(data=f"data/wikiauto_docs_{split}.csv",
               save_dir=f"context/{x}", x_col=x)
```

## Training the planning models
The script below can be used to train the context-independent classifier on 2 GPUs.

```bash
python plan_simp/scripts/train_clf.py \
  --name=classifier \
  --project=planning_models \
  --train_file=data/wikiauto_sents_train.csv \
  --val_file=data/wikiauto_sents_valid.csv \
  --x_col=complex \
  --y_col=label \
  --batch_size=32 \
  --learning_rate=1e-5 \
  --ckpt_metric=val_macro_f1 \
  --hidden_dropout_prob=0.1 \
  --max_epochs=10 \
  --devices=2 \
```

To train the contextual classifier, adjust the script as follows.
1. Change the model name to pg-dyn.
2. Use weight initialization.

```bash
  --checkpoint=planning_models/<path_to_classifier> \
```

3. Use dynamic context and a context window of 13.

```bash
  --add_context \
  --context_window=13 \
  --context_doc_id=pair_id \
  --context_dir=context/complex \
  --simple_context_doc_id=pair_id \
  --simple_context_dir=context/simple \
```

To include document positional embeddings, also add the following line.

```bash
  --doc_pos_embeds \
```

## Evaluating the planning models
The command below can be used to evaluate the context-independent classifier.

```bash
python plan_simp/scripts/eval_clf.py \
    <path_to_planning_model> \
    data/wikiauto_sents_test.csv \
```

To evaluate a contextual classifier, add the following arguments.

```bash
  --add_context=True \
  --context_dir=context/complex \
  --simple_context_dir=context/simple \
```

To evaluate on Wiki-auto the contextual classifier pretrained on Newsela-auto,
set the model path to liamcripwell/pgdyn-plan and also specify the target reading level.

 ```bash
  --reading_lvl=3 \
```

## Training the simplification models
The script below shows how to train a text-only BART model.

```bash
python plan_simp/scripts/train_bart.py \
  --name=<model_name> \ #can be any name
  --project=simplification_models \
  --train_file=data/wikiauto_<docs/para/sents>_train.csv \
  --val_file=data/wikiauto_<docs/para/sents>_valid.csv \
  --x_col=complex \
  --y_col=simple \
  --batch_size=8 \
  --accumulate_grad_batches=2 \
  --lr=2e-5 \
  --devices=2 \
  --skip_val_gen \
```

To train any other model, add the following arguments.

If it is a context-aware model (ConBART):

```bash
  --add_context \
  --context_dir=context/complex \
  --context_doc_id=pair_id \
  --simple_context_dir=context/simple \
  --simple_context_doc_id=pair_id \
```

If it is a LED model:

```bash
  --longformer \
```

If it is a plan-guided model:

```bash
  --op_col=label \
```

## Generating simplifications
Use the script below to perform inference with a text-only model.

```bash
mkdir results

python plan_simp/scripts/generate.py inference \
  --model_ckpt=<path_to_simplification_model> \
  --test_file=data/wikiauto_<docs/para/sents>_test.csv \
  --out_file=results/<model_name>.csv \
```

To perform inference with any other model, adjust the script as follows.

If it is a context-aware model:

```bash
  --context_doc_id=pair_id \
  --context_dir=context/complex \
  --temp_dir=<model_name> \
```

If it is guided by an oracle plan:

```bash
  --op_col=label \
```

If it is guided by the contextual classifier:

1. Change the first argument from inference to dynamic

2. Add the following arguments:

```bash
  --clf_model_ckpt=<path_to_planning_model> \
  --context_doc_id=pair_id \
  --context_dir=context/complex \
  --temp_dir=<model_name> \
```

3. If the model operates at the paragraph-level, also add this line:

```bash
  --para_lvl=True \
```

Finally, if the system was pretrained on Newsela-auto:

```bash
  --reading_lvl=3 \
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
  --input_data=data/wikiauto_docs_test.csv \
  --output_data=results/<model_name>.csv \
  --x_col=complex \
  --r_col=simple \
  --y_col=pred \
  --doc_id_col=pair_id \
  --prepro=True \
  --sent_level=True \
  --bartscore_path=bart_score.pth \
```
