import copy

import torch
from torch.utils.data import Dataset, DataLoader
from data_loading.loader import DataLoader as ECDataset
from torchtext.vocab import vocab
import argparse
import itertools
from torch.nn.utils.rnn import pad_sequence
from collections import Counter, OrderedDict


class BILSTMDataset(Dataset):

    @staticmethod
    def _build_vocab(tokens, valence=False):
        all_tokens = itertools.chain(*tokens)
        if valence:
            all_tokens = [BILSTMDataset.get_special_valence_token(x) for x in tokens]
        #print([type(x) for x in all_tokens])
        counted_tokens = Counter(all_tokens)
        sorted_by_freq_tuples = sorted(counted_tokens.items(), key=lambda x: x[1], reverse=True)
        ordered_dict = OrderedDict(sorted_by_freq_tuples)
        unk_token = '<unk>'
        dataset_vocab = vocab(ordered_dict=ordered_dict, specials=[unk_token])
        return dataset_vocab
        # fix vocab issue here - does it work on the lab machine? if that's the case it might be enough

    @staticmethod
    def get_special_sep_char():

        SPECIAL_SEP_CHAR = "[SEP]" # https://datascience.stackexchange.com/questions/86566/whats-the-right-input-for-gpt-2-in-nlp
        return SPECIAL_SEP_CHAR

    @staticmethod
    def get_special_valence_token(valence:int, unify_pos_neg=False) -> str:
        valence_token_map = {
            0: "[NEUTRO]",
            1: "[POSITIVO]",
            2: "[POSITIVO]",
            -1: "[NEGATIVO]",
            -2: "[NEGATIVO]"
        }

        if unify_pos_neg:
            valence = valence if valence == 0 else 1

        result = valence_token_map.get(valence)
        if result is None:
            raise Exception(f"No mapping found for {result}")
        return result

    @staticmethod
    def process_data_for_bilstm(tokens,iobs,valences, add_valence=False):
        # the objective of this function is to map tokens on their embedding ids
        # iobs should be mapped to their respective id (N, Y, O) - O eventually should be removed when the task is set as Machine translation
        # valences should be mapped either to -1, 0, 1 or 0, 1(-1)
        # tokens and iobs are shift to introduce the [sep]

        token_data = []
        iob_data = []

        for sentence_iob, sentence_tokens, sentence_valence in zip(iobs, tokens, valences):
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

            if not all([isinstance(x,str) for x in new_tokens_sentence]):
                raise Exception(f"A None value was introduced here: {new_tokens_sentence}  vs {sentence_tokens} and sentence valence {sentence_valence}" )

            if len(new_iob_sentence) != len(new_tokens_sentence):
                raise Exception("IOB and token sentence length dont' match !!!!!!!!!!!")
            token_data.append(new_tokens_sentence)
            iob_data.append(new_iob_sentence)
            #print(new_tokens_sentence, new_iob_sentence)
        return token_data, iob_data

    @staticmethod
    def add_padding(tokens):
        pad_token_id = 0
        padded_sequence = pad_sequence(tokens, padding_value=pad_token_id, batch_first=True)
        pad_mask = ~(padded_sequence == pad_token_id)
        return padded_sequence, pad_mask.int()

    def show_first_example(self):
        print(self.token_ids[0])
        print(self.tokens_mask[0])
        print(self.iob_ids[0])
        print(self.iob_mask[0])
        print(self.valence_ids[0])
        print(self.tokens[0])
        print(self.iobs[0])

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

        self.vocab = BILSTMDataset._build_vocab(self.tokens)
        self.iob_mapping = BILSTMDataset._build_vocab(self.iobs)
        self.valence_mapping = BILSTMDataset._build_vocab(self.valences, valence=True)
        self.token_ids, self.tokens_mask = BILSTMDataset.add_padding([torch.tensor(self.vocab(sent)) for sent in self.tokens])
        self.iob_ids, self.iob_mask = BILSTMDataset.add_padding([torch.tensor(self.iob_mapping(sent)) for sent in self.iobs])
        self.valence_ids = [torch.tensor(self.valence_mapping([BILSTMDataset.get_special_valence_token(sent)])) for sent in self.valences]
        self.seq_len = len(self.token_ids[0])
        # transformation to vectors and also mapping IOB, MASK to IDs, VALENCE

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.dataset.tokens)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample

        # Load data and get label
        dict_x = {
            "source": self.token_ids[index],
            "mask": self.tokens_mask[index],
            "target": self.iob_ids[index],
            "seq_len": self.seq_len
        }

        y = self.iob_ids[index]

        return dict_x, y

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
