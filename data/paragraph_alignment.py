"""
How to create the preprocessed paragraph-level datasets:
Step 1: Download the preprocessed sentence- and document-level datasets,
        and place the files into this directory.
        (https://drive.google.com/file/d/1lU8htUIVBuuU24HrPErpV01hlA6tc-d1)
Step 2: Download part 1 of the original wiki-auto data, and also place it into this directory.
        (https://www.dropbox.com/sh/ohqaw41v48c7e5p/AAD77KWGOT9gytzmTno2PWbIa/wiki-auto-all-data)
Step 3: Run the script.

Alternatively, download the resulting datasets directly from
https://drive.google.com/file/d/1ZeALAhdWBfVNsFlRnPsGGia4NQB1Dbjq.
"""


import json
import pandas as pd
from collections import defaultdict


def get_paragraph_alignments(df, wiki_auto):
    """
    Use the original wiki-auto data to find out which sentences
    in the preprocessed dataset belong to the same paragraph.
    """
    alignments = {}

    for doc_id, aligned_doc in zip(df.pair_id, df.complex):
        aligned_sents = aligned_doc.split(" <s> ")
        sent_ids_per_paragraph = defaultdict(list)
        sent_id = 0

        doc = wiki_auto[str(doc_id)]["normal"]["content"]
        for id, sent in doc.items():
            if sent == aligned_sents[sent_id]:
                para_id = id.split("-")[2]
                sent_ids_per_paragraph[para_id] += [sent_id]
                sent_id += 1
            if sent_id == len(aligned_sents):
                break

        alignments[doc_id] = list(sent_ids_per_paragraph.values())

    return dict(sorted(alignments.items()))


def get_paragraph_level_data(df, alignments):
    """
    Create a paragraph-level dataset based on the paragraph alignments.
    """
    para_data = []

    for doc_id, sent_ids_per_paragraph in alignments.items():
        document = df[df.pair_id == int(doc_id)]
        first_sent = document.iloc[0]
        doc_dict = {"title": first_sent.title, "pair_id": first_sent.pair_id}

        for i, sent_ids in enumerate(sent_ids_per_paragraph):
            paragraph = document[document.sent_id.isin(sent_ids)]

            sent_id = paragraph.iloc[0].sent_id
            complex = " <s> ".join(list(paragraph.complex))
            simple = [s for list_ in paragraph.simple for s in eval(list_)]
            para_dict = doc_dict | {"para_id": i, "sent_id": sent_id, \
                                    "complex": complex, "simple": simple}

            for c in ["label", "simp_sent_id", "doc_pos" ,"doc_quint", "doc_len"]:
                para_dict[c] = list(paragraph[c])

            para_data.append(para_dict)

    return pd.DataFrame(para_data)


if __name__ == '__main__':
    # Load the original wiki-auto data.
    with open("wiki-auto-part-1-data.json") as f:
        wiki_auto = json.load(f)

    # Create the paragraph-level train, validation and test sets.
    for split in ["train", "valid", "test"]:
        doc_df = pd.read_csv(f"wikiauto_docs_{split}.csv")
        alignments = get_paragraph_alignments(doc_df, wiki_auto)
        sent_df = pd.read_csv(f"wikiauto_sents_{split}.csv")
        para_df = get_paragraph_level_data(sent_df, alignments)
        para_df.to_csv(f"wikiauto_para_{split}.csv", index=False)
