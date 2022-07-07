import copy

import torch
from torch.utils.data import Dataset, DataLoader
from data_loading.loader import DataLoader as ECDataset
#from torchtext.vocab import vocab
import argparse
import itertools
from collections import Counter, OrderedDict


class BILSTMDataset(Dataset):

    @staticmethod
    def _build_vocab(tokens):
        all_tokens = itertools.chain(*tokens)
        counted_tokens = Counter(all_tokens)
        sorted_by_freq_tuples = sorted(counted_tokens.items(), key=lambda x: x[1], reverse=True)
        ordered_dict = OrderedDict(sorted_by_freq_tuples)
        unk_token = '<unk>'
        #dataset_vocab = vocab(OrderedDict([(token, 1) for token in tokens]), specials=[unk_token])
        #return dataset_vocab

    @staticmethod
    def get_special_sep_char():

        SPECIAL_SEP_CHAR = "[SEP]" # https://datascience.stackexchange.com/questions/86566/whats-the-right-input-for-gpt-2-in-nlp
        return SPECIAL_SEP_CHAR

    @staticmethod
    def get_special_valence_token(valence, unify_pos_neg=False):
        valence_token_map = {
            0: "[NEUTRO]",
            1: "[POSITIVO]",
            -1: "[NEGATIVO]"
        }

        valence = 1 if unify_pos_neg and valence == 1 or valence == -1 else valence

        return valence_token_map.get(valence)

    @staticmethod
    def process_data_for_bilstm(tokens,iobs,valences, add_valence=False):
        # the objective of this function is to map tokens on their embedding ids
        # iobs should be mapped to their respective id (N, Y, O) - O eventually should be removed when the task is set as Machine translation
        # valences should be mapped either to -1, 0, 1 or 0, 1(-1)
        # tokens and iobs are shift to introduce the [sep]

        token_data = []
        iob_data = []

        for sentence_iob, sentence_tokens, sentence_valence in zip(iobs, tokens, valences):
            print(sentence_iob, sentence_tokens)

            new_iob_sentence = [x for x in sentence_iob]
            new_tokens_sentence = [x for x in sentence_tokens]
            insertion_counter = 0 # tracks the difference from original sentence to the original one
            started = False
            for idx, (token_iob, token_text) in enumerate(zip(sentence_iob, sentence_tokens)):
                if token_iob == 'O' and not started:
                    pass
                elif token_iob.startswith('B-') :
                    started = True
                    new_iob_sentence.insert(idx+insertion_counter,'O')
                    new_tokens_sentence.insert(idx+insertion_counter,BILSTMDataset.get_special_sep_char())
                    insertion_counter += 1
                elif token_iob == 'O' and started:
                    new_iob_sentence.insert(idx+insertion_counter,'O')
                    new_tokens_sentence.insert(idx+insertion_counter,BILSTMDataset.get_special_sep_char())
                    insertion_counter += 1
                    started = False
                else:
                    pass

            if add_valence:
                sentence_valence = BILSTMDataset.get_special_valence_token(sentence_valence)
                new_iob_sentence.insert(0, 'O')
                new_iob_sentence.append('O')
                new_tokens_sentence.insert(0, sentence_valence)
                new_tokens_sentence.append(sentence_valence)

            if len(new_iob_sentence) != len(new_tokens_sentence):
                raise Exception("IOB and token sentence length dont' match !!!!!!!!!!!")
            token_data.append(new_tokens_sentence)
            iob_data.append(new_iob_sentence)
            print(new_tokens_sentence, new_iob_sentence)
        return token_data, iob_data

    'Characterizes a dataset for PyTorch'
    def __init__(self, data_files):
        'Initialization'
        self.dataset = ECDataset(data_files)
        self.tokens = self.dataset.tokens
        self.iobs = self.dataset.iobs
        self.valences = self.dataset.valences
        self.masks = self.dataset.masks

        self.tokens, self.iobs = BILSTMDataset.process_data_for_bilstm(self.tokens, self.iobs, self.valences, add_valence=True)
        # the question is whether to do the transformation on the fly (fetch time) or at initialization
        # given the small size of data - initialization is best
        self.vocab = BILSTMDataset._build_vocab(self.dataset.tokens)
        # transformation to vectors and also mapping IOB, MASK to IDs, VALENCE

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.dataset.tokens)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X = torch.load('data/' + ID + '.pt')
        y = self.labels[ID]

        return X, y

# steps

# create vocab
# map voca to list

def create_argument_parser():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-d','--data-files',  action="store", dest="data_files",nargs='+',
                        help='the training file')

    return parser


def main():
    # Use a breakpoint in the code line below to debug your script.
    parser = create_argument_parser()
    args = parser.parse_args()

    data_object = BILSTMDataset(args.data_files)
    #for tokens, labels, mask in zip(data_object.tokens, data_object.iobs, data_object.masks):

    #    print(tokens,labels, mask)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
