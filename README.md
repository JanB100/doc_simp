# Thesis

This branch contains code and resources for Chapter 3 of my [Master's thesis](https://scripties.uba.uva.nl/search?id=record_55015): Plan-Guided Simplification of Biomedical Documents.
It is based on this [repository](https://github.com/liamcripwell/plan_simp).

We changed the original code by implementing support for the merge operator and a two-stage classifier approach. We also provide our code for aligning and preprocessing the data, along with the updated Cochrane corpus, the resulting alignments and the preprocessed Cochrane-auto datasets. Moreover, we provide detailed documentation on how to train and evaluate each model below.

## Installation

```bash
git clone https://github.com/JanB100/doc_simp.git
cd doc_simp
conda create -n doc_simp python=3.10
conda activate doc_simp
pip install -r requirements.txt
pip install -e .
```

## Pretrained models
We share all Cochrane-auto pretrained models on [HuggingFace](https://huggingface.co/janbakker). To leverage these models, simply follow the instructions below and set the model path to the HuggingFace model name. All model names end on -thesis except for o-bartsent-cochraneauto. 

We also share the [checkpoint](https://drive.google.com/file/d/12FHcrrPdqKgE6R4G7uuTUasuAS9da018/view?usp=sharing) for the neural CRF alignment model which we pretrained on Wiki-manual.

## Data
We share the train/val/test splits of our updated Cochrane corpus under [data/corpus](data/corpus).

The script [load_data.py](load_data.py) contains our code for extracting the sentences and paragraphs from the technical abstracts and lay summaries in this corpus.

The script [alignment.py](alignment.py) contains our code for automatically aligning these sentences using the pretrained alignment model and for computing its performance on a manually annotated subset.

The resulting Cochrane-auto alignments can be found together with our manual alignments under [data/alignments](data/alignments).

The script [preprocessing.py](preprocessing.py) contains our code for constructing the preprocessed Cochrane-auto datasets based on these alignments.

The resulting sentence-, paragraph- and document-level datasets can be found in the [data](data) directory. 

Finally, the script [analysis.py](analysis.py) contains our code for aligning the output sentences back to the input sentences and subsequently plotting confusion matrices of simplification operation labels.

## Preparing context representations
Start by generating the context encodings of all complex and simple sentences. These are used by ConBART.

```bash
mkdir context
```
```python
from plan_simp.scripts.encode_contexts import encode

for split in ["train", "val", "test"]:
    for x in ["complex", "simple"]:
        encode(data=f"data/cochraneauto_docs_{split}.csv",
               save_dir=f"context/{x}", x_col=x)
```

## Training the planning models
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

To train the first-stage classifier, replace --use_merge_labels with this argument:

```bash
  --binary_clf \
```

To train the second-stage classifier, add the following argument to the script above:

```bash
  --second_stage \
```

## Evaluating the planning models
The command below can be used to generate a plan with a classifier and evaluate it.

```bash
mkdir results

python plan_simp/scripts/eval_clf.py \
    <path_to_planning_model> \
    data/cochraneauto_sents_test.csv \
    --out_file=results/<model_name>.csv \
    --use_merge_labels \
```

To evaluate the first-stage classifier, remove the last argument.

To leverage the two-stage approach, run the command below after evaluating the first-stage classifier:

```bash
python plan_simp/scripts/eval_clf.py \
    <path_to_planning_model> \
    results/<binary_clf_name>.csv \
    --out_file=results/<model_name>.csv
    --use_merge_labels \
    --second_stage \
```

## Training the simplification models
The script below shows how to train a text-only BART model.

```bash
python plan_simp/scripts/train_bart.py \
  --name=<model_name> \ #can be any name
  --project=simplification_models \
  --train_file=data/cochraneauto_<docs/para/sents>_train.csv \
  --val_file=data/cochraneauto_<docs/para/sents>_val.csv \
  --x_col=complex \
  --y_col=simple \
  --batch_size=8 \
  --accumulate_grad_batches=2 \
  --lr=2e-5 \
  --devices=2 \
  --skip_val_gen \
  --sent_level \ #only if it is a sentence-level model
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
python plan_simp/scripts/generate.py inference \
  --model_ckpt=<path_to_simplification_model> \
  --test_file=data/cochraneauto_<docs/para/sents>_test.csv \
  --out_file=results/<model_name>.csv \
```

To perform inference with any other model, adjust the script as follows.

If it is a context-aware model:

1. Change the first argument from inference to dynamic

2. Add the following arguments:

```bash
  --context_doc_id=pair_id \
  --context_dir=context/complex \
  --temp_dir=<model_name> \
```

If it is guided by an oracle plan:

```bash
  --op_col=label \
```

If it is guided by a generated plan:

```bash
  --test_file=results/<classifier><_docs/_para/>.csv \
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
  --input_data=data/cochraneauto_docs_test.csv \
  --output_data=results/<model_name>.csv \
  --x_col=complex \
  --r_col=simple \
  --y_col=pred \
  --doc_id_col=pair_id \
  --prepro=True \
  --sent_level=True \
  --bartscore_path=bart_score.pth \
```
