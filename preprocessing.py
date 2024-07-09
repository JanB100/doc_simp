
import Levenshtein
import math
import pandas as pd
from utils import load_pickle_file

from load_data import load_cochrane_data
from transformers import BartTokenizerFast


def label(complex_sent, simple_sents, threshold=0.92):
    """ 
    Assigns a complex sentence a simplification operation label 
    based on the simple sentences to which it is aligned.
    """
    if simple_sents == []:
        return "delete"
    elif len(simple_sents) > 1:
        return "split"
    elif Levenshtein.ratio(complex_sent, simple_sents[0]) >= threshold:
        return "ignore"
    
    return "rephrase"


def get_merge_idx(labels, sent_ids, cs_alignments, sc_alignments):
    """ 
    Returns the indices of the complex sentences within a given paragraph
    whose label should be replaced with 'merge'. 
    
    labels: original labels of the complex sentences within the paragraph
    sent_ids: ids of the complex sentences within the paragraph
    cs_alignments: new alignments from the sentences in the corresponding complex document (c) 
                   to the sentences in the corresponding simple document (s)
    sc_alignments: original alignments from the sentences in the simple document (s) 
                   to the sentences in the complex document (c)
    """
    merge_idx = []

    for i, (label, sent_id) in enumerate(zip(labels, sent_ids)):
        # Consecutive complex sentences should be labelled 'merge' if:
        if label == "rephrase":
            # (1) one of them was originally labelled 'rephrase', 
            #     because a simple sentence s_j was aligned to it (s -> c),
            #     and it is newly aligned to s_j (c -> s)
            simp_sent_id = cs_alignments[sent_id] - 1

            if simp_sent_id > -1 and sc_alignments[simp_sent_id] - 1 == sent_id:
                idx = [i]

                # (2) the surrounding sentences were labelled 'delete',
                # (3) and are newly aligned to the same s_j (c -> s)
                j = 1
                while i-j > -1 and cs_alignments[sent_id-j]-1 == simp_sent_id \
                               and labels[i-j] == "delete":
                    idx = [i-j] + idx
                    j += 1

                j = 1
                while i+j < len(sent_ids) and cs_alignments[sent_id+j]-1 == simp_sent_id \
                                          and labels[i+j] == "delete":
                    idx = idx + [i+j]
                    j += 1

                if len(idx) > 1:
                    merge_idx.append(idx)

    return merge_idx


def add_merge_labels(df, cs_alignments, sc_alignments):
    """
    Given the part of a sentence-level dataset that belongs to a single paragraph,
    replace the labels of the complex sentences with 'merge' where it is needed.
    """
    sent_ids = [x["sent_id"] for x in df]
    labels = [x["label"] for x in df]

    for idx in get_merge_idx(labels, sent_ids, cs_alignments, sc_alignments):
        ssid =  df[idx[0]]["simp_sent_id"]

        for i in idx[:-1]:
            df[i] |= {"label": "merge", "simp_sent_id": ssid}

        # The last sentence involved in a merge operation is labelled 'none'
        df[idx[-1]] |= {"label": "none", "simp_sent_id": ssid}


def build_sentence_level_dataset(dois, complex, simple, para_ids, sc_alignments, tokenizer, 
                                 use_merge_labels=False, cs_alignments=None, threshold=0.5):
    """
    Build a sentence-level dataset based on the alignments between sentences.
    """
    sent_df = []

    for i, doc in enumerate(complex):
        if len(set(a for a in sc_alignments[i] if a))  / len(doc) < threshold:
            continue

        if len(tokenizer('<pad> ' * len(doc) + ' <s> '.join(doc)).input_ids) > 1024:
            continue

        simp_sent_id, len_para = 0, 0
        pair_id = dois[i].split(".")[2]

        for j, sent in enumerate(doc):
            doc_pos = (j+1)/len(doc)
            simple_sents = [simple[i][k] for k, x in enumerate(sc_alignments[i]) if x-1==j]

            sent_df.append({"pair_id": pair_id, "para_id": para_ids[i][j], "sent_id": j,
                            "complex": sent, "label": label(sent, simple_sents),
                            "simple": simple_sents, "simp_sent_id": simp_sent_id, 
                            "doc_pos": doc_pos, "doc_quint": math.ceil(doc_pos/0.2), 
                            "doc_len": len(doc)})

            simp_sent_id += len(simple_sents)
            len_para += 1

            if j+1 == len(doc) or para_ids[i][j+1] > para_ids[i][j]:
                if use_merge_labels:
                    add_merge_labels(sent_df[-len_para:], cs_alignments[i], sc_alignments[i])
                len_para = 0

    return pd.DataFrame(sent_df)


def build_higher_level_datasets(sent_df):
    """
    Build paragraph- and document-level datasets given a sentence-level dataset.
    """
    para_df, doc_df = [], []

    for (pair_id, para_id), df in sent_df.groupby(["pair_id", "para_id"], sort=False):
        simple_para = [x for y in df.simple for x in y]
        row = {"pair_id": pair_id, "para_id": para_id, 
               "complex": " <s> ".join(df.complex), "simple": " <s> ".join(simple_para)}
        row |= {col: list(df[col]) for col in df if col not in row.keys()}
        para_df.append(row)

    for pair_id, df in sent_df.groupby("pair_id", sort=False):
        simple_doc = [x for y in df.simple for x in y]
        row = {"pair_id": pair_id, "complex": " <s> ".join(df.complex), 
               "simple": " <s> ".join(simple_doc)}
        row |= {col: list(df[col]) for col in df if col not in row.keys()}
        doc_df.append(row)
        
    return pd.DataFrame(para_df), pd.DataFrame(doc_df)


def build_all_datasets():
    """ 
    Function demonstrating how we built our datasets. 
    """
    for split in ["train", "val", "test"]:
        dois, complex, complex_para_ids, simple, simple_para_ids = load_cochrane_data(split)

        filtered_sc_alignments = load_pickle_file(f"alignments/filtered_sc_{split}.pkl")
        cs_alignments = load_pickle_file(f"alignments/cs_{split}.pkl")

        tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-base', add_prefix_space=True)

        sent_df = build_sentence_level_dataset(dois, complex, simple, complex_para_ids, \
                                filtered_sc_alignments, tokenizer, use_merge_labels=True,
                                cs_alignments=cs_alignments)
        para_df, doc_df = build_higher_level_datasets(sent_df)

        sent_df.to_csv(f"cochrane_sents_{split}.csv", index=False)
        para_df.to_csv(f"cochrane_para_{split}.csv", index=False)
        doc_df.to_csv(f"cochrane_docs_{split}.csv", index=False)
