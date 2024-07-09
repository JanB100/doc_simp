import nltk
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay as cmd

from plan_simp.data.utils import convert_labels, CLASS_LABELS

from alignment import align
from preprocessing import label, get_merge_idx
from utils import load_pickle_file


def load_results(df):
    """
    Returns complex sentences along with the corresponding model outputs and labels.

    df: DataFrame with model predictions (obtained by running the generate function).
    """
    complex, output = [], []

    for (_, pdf) in df.groupby("pair_id", sort=False):
        complex.append((" <s> ".join(pdf.complex)).split(" <s> "))
        pred = nltk.sent_tokenize(" ".join(pdf.pred.dropna()))

        j = 0
        while j < len(pred):
            while pred[j][-4:] in ["e.g.", "i.e."]:
                pred = pred[:j] + [pred[j]+" "+pred[j+1]] + pred[j+2:]
            j += 1
        output.append(pred)

    labels = list(df.label) if "label" in df else list(df.labels)
    if labels[0][0] == "[":
        labels = [l for ls in labels for l in eval(ls)]

    return complex, output, labels


def plot_confusion_matrix(model_name):
    """
    Plots the confusion matrices for the oracle labels and the 
    labels of the aligned model outputs with respect to the inputs.
    """
    para_ids = pd.read_csv("data/cochrane_sents_test.csv").para_id

    df = pd.read_csv(f"results/{model_name}.csv")
    complex, simple, gts = load_results(df)
    #sc_alignments = align(simple, complex, aligner)
    #save_as_pickle_file(sc_alignments, f"alignments/sc_{model_name}.pkl")
    sc_alignments = load_pickle_file(f"alignments/sc_{model_name}.pkl")
    #cs_alignments = align(complex, simple, aligner)
    #save_as_pickle_file(cs_alignments, f"alignments/cs_{model_name}.pkl")
    cs_alignments = load_pickle_file(f"alignments/cs_{model_name}.pkl")

    pred_ls = []
    n = 0

    for i, doc in enumerate(complex):
        pred_labels = []

        for j, sent in enumerate(doc):
            simple_sents = [simple[i][k] for k, x in enumerate(sc_alignments[i]) if x-1==j]
            pred_labels.append(label(sent, simple_sents))

        sent_ids = list(range(len(pred_labels)))
        for idx in get_merge_idx(pred_labels, sent_ids, cs_alignments[i], sc_alignments[i], \
                                list(para_ids[n:n+len(doc)])):
            for j in idx:
                pred_labels[j] = "merge"

        pred_ls += pred_labels
        n += len(doc)

    pred_ls = convert_labels(pred_ls, CLASS_LABELS)
    gts = convert_labels(gts, CLASS_LABELS)

    disp = cmd.from_predictions(gts, pred_ls, display_labels=CLASS_LABELS.keys(), cmap='Blues')
    disp.im_.colorbar.remove()
    plt.show()
