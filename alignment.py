from transformers.data.processors import InputExample, glue_convert_examples_to_features
import torch
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader
from sklearn.utils.extmath import softmax
import tqdm
import numpy as np

from utils import load_pickle_file, save_as_pickle_file


def get_similarity_score_from_sent_pair(sentA_list, sentB_list, model, tokenizer, max_length = 128, mode = 'simple_to_complex'):
    """ 
    Function copied from the sample notebook by Jiang et al. .
    (https://colab.research.google.com/drive/1-6hWzTIgrEMrcervG_ANqrf1o2CugnfS)
    """
    model.eval()
    fake_example = []

    if mode == 'complex_to_simple':
        sentA, sentB = sentB, sentA

    for i in range(len(sentA_list)):
        fake_example.append(InputExample(guid=i, text_a=sentA_list[i], text_b=sentB_list[i], label='good'))

    fake_example_features = glue_convert_examples_to_features(fake_example, tokenizer, max_length, label_list = ["good", 'bad'], output_mode = 'classification')

    all_input_ids = torch.tensor([f.input_ids for f in fake_example_features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in fake_example_features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in fake_example_features], dtype=torch.long)
    all_label = torch.tensor([f.label for f in fake_example_features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_label)
    
    output_tensor = []
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=8)
    for batch in tqdm.tqdm_notebook(eval_dataloader):
        my_device = torch.device('cuda:0')
        batch = tuple(t.to(my_device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels': batch[3]}
            outputs = model(input_ids=inputs["input_ids"], \
                            attention_mask=inputs["attention_mask"], \
                            token_type_ids=inputs["token_type_ids"], \
                            labels=None, \
                            )

            output_tensor.append(outputs['logits'].cpu().data)

    output_tensor = torch.cat(output_tensor)
    probabilities = softmax(output_tensor)
    probabilities = [i[0] for i in probabilities]

    return probabilities


def align(source_sents, target_sents, model):
    """ 
    Leverages the alignment model to automatically align each source 
    sentence in a text to one or zero target sentences in the parallel text.
    Returns, for each source sentence: 0 if it is not aligned, or
    the index of the target sentence to which it is aligned + 1.
     """
    return [model(s, t, None)[2] if t else [0 for _ in s] for s, t in zip(source_sents, target_sents)]


def filter_alignments(sc_alignments, complex, simple, simple_para_ids, bert_model, tokenizer):
    """
    Filters the alignments such that a simple sentence can only be aligned to
    multiple complex sentences if they occur within the same paragraph.
    """
    new_alignments = sc_alignments

    for i, doc in enumerate(complex):
        for j, sent in enumerate(doc):
            aligned_idx = [k for k, x in enumerate(sc_alignments[i]) if x-1==j]
            aligned_para_ids = [simple_para_ids[i][k] for k in aligned_idx]

            # If a complex sentence c_i is aligned to simple sentences from multiple paragraphs,
            if len(set(aligned_para_ids)) > 1:
                aligned_sents = {sid: [] for sid in set(aligned_para_ids)}
                for k, sid in zip(aligned_idx, aligned_para_ids):
                    aligned_sents[sid] += [simple[i][k]]

                # concatenate the simple sentences that are aligned to c_i per paragraph, and
                simple_sents_list = [" ".join(sents) for sents in aligned_sents.values()]
                complex_sent_list = [sent for _ in set(aligned_para_ids)]
                # compute the similarity between c_i and each concatenation.
                probabilities = get_similarity_score_from_sent_pair(simple_sents_list, \
                                            complex_sent_list, bert_model, tokenizer)

                # Only keep the alignments to the paragraph with the most similar aligned sentences.
                index = np.argmax(probabilities)
                best_para_id = list(aligned_sents.keys())[index]

                for k in aligned_idx:
                    if simple_para_ids[i][k] != best_para_id:
                        new_alignments[i][k] = 0

    return new_alignments


def obtain_alll_alignments(aligner_checkpoint, split="train"):
    """ 
    Function demonstrating how we obtained our alignments. 
    """
    from transformers import BertTokenizer, BertForSequenceClassification
    from load_data import load_cochrane_data

    _, complex, _, simple, simple_para_ids = load_cochrane_data(split)

    aligner = load_pickle_file(aligner_checkpoint)
    aligner.to("cuda").eval()
    sc_alignments = align(simple, complex, aligner)
    cs_alignments = align(complex, simple, aligner)

    checkpoint = 'chaojiang06/wiki-sentence-alignment'
    tokenizer = BertTokenizer.from_pretrained(checkpoint, do_lower_case=True)
    bert_model = BertForSequenceClassification.from_pretrained(checkpoint, output_hidden_states=True)
    bert_model.to("cuda").eval()
    filtered_sc_alignments = filter_alignments(sc_alignments, complex, simple, \
                                               simple_para_ids, bert_model, tokenizer)

    return sc_alignments, cs_alignments, filtered_sc_alignments


def alignment_performance(automatic_alignments, manual_alignments):
    """ 
    Computes the performance of the alignment model on our manually annotated subset. 
    """
    tp, fp, total, upp_bound = 0, 0, 0, 0

    for i, ground_truths in manual_alignments.items():
        total += len(ground_truths)
        upp_bound += len(set(g[0] for g in ground_truths)) 

        for j, pred in enumerate(automatic_alignments[i]):
            if pred:
                if [f"simple_{j}", f"complex_{pred - 1}"] in ground_truths:
                    tp += 1
                else:
                    fp += 1

    fn = total - tp

    print(f"tp {tp}, fp {fp}, fn {fn}, total positive {total}, upper bound {upp_bound}")

    precision = tp / (tp + fp)
    recall = tp / total
    f1 = 2 * precision * recall / (precision + recall) if tp else 0.0

    print("precision: %.6f  recall: %.6f  f1: %.6f" % (precision, recall, f1))
    return f1


if __name__ == "__main__":
    """ 
    Sample code for computing the performance of the alignment model 
    on our manually annotated subset
    """
    manual_alignments = load_pickle_file("alignments/manual.pkl")

    sc_alignments_train = load_pickle_file("alignments/sc_train.pkl")
    sc_alignments_val = load_pickle_file("alignments/sc_val.pkl")
    sc_alignments_test = load_pickle_file("alignments/sc_test.pkl")

    automatic_alignments = sc_alignments_train + sc_alignments_val + sc_alignments_test
    alignment_performance(automatic_alignments, manual_alignments)
