from typing import List, Union
import argparse
import json


def create_argument_parser():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-d','--data-files',  action="store", dest="data_files",nargs='+',
                        help='the training file')

    return parser


class DataLoader:

    @staticmethod
    def strip_iob_tags(iobs):
        return [x.replace("B-", "").replace("I-","") for x in iobs]

    @staticmethod
    def parse_single_fu(fu, keep_iob=True):
        doc_tokens = fu["text"].lower().split()
        iobs = fu["iob"] if keep_iob else DataLoader.strip_iob_tags(fu["iob"])
        valence = fu["valence"]

        return doc_tokens, iobs, valence

    @staticmethod
    def build_mask(iobs):
        final_mask = []
        for iob in iobs:
            final_mask.append(
                0 if iob == "O" else 1
            )
        return final_mask

    @staticmethod
    def generate_samples(filename, keep_iob=True):
        """
        read input file and generate samples. every EC candidate is considered as one sample.
        EC candidate is represented using a mask over the input Functional unit. A [B C] D E -> mask:[1,3]
        Thus the mask and the FU forms the input. while label (EC or Non-EC) is the output
        """
        with open(filename, "r") as in_file:
            data = json.load(in_file)

        tokens = []
        iobs = []
        valences = []
        mask = []
        for key, note in data.items():
            for key, turn in note.get("turns").items():
                for id, fu in turn.items():
                    tokenized_text, iob, valence = DataLoader.parse_single_fu(fu, keep_iob=keep_iob)
                    tokens.append(tokenized_text)
                    iobs.append(iob)
                    mask.append(DataLoader.build_mask(iob))
                    valences.append(valence)

        return tokens, iobs, valences, mask

    @staticmethod
    def map_mask_tokenizer(data_tokens, data_mask_old, tokenizer):
        """
        the mask that consists of the begin and end indices of each EC-candidate has to be mapped to new indices
        that we get after the tokenization (which uses BPE to break the words into poarts).
        Original: A [B C] D E -> mask:[1,3]
        New: A A1 B C C1 C2 D D1 E E1 -> mask:[2,6]
        """
        data_mask_new = []
        for words, mask_old in zip(data_tokens, data_mask_old):
            encodings = tokenizer(words, is_split_into_words=True)
            old_new_mapping = {}
            """print(words)
            for x in encodings.word_ids(0): print(x, end=" ")
            print("")"""
            for i, a in enumerate(encodings.word_ids(0)):
                if a == None: continue
                if a in old_new_mapping: continue
                old_new_mapping[a] = i
            old_new_mapping[max(old_new_mapping) + 1] = i
            """print(old_new_mapping)
            print(mask_old)"""
            mask_new = [old_new_mapping[x] for x in mask_old]
            data_mask_new.append(mask_new)
        return data_mask_new

    @staticmethod
    def load_data_files(data_files: Union[List[str], str], keep_iob=True):
        if isinstance(data_files, str):
            data_files = [data_files]

        all_tokens, all_iobs, all_valences, all_masks = [], [], [], []
        for file in data_files:
            data_tokens, iobs, valences, mask = DataLoader.generate_samples(file, keep_iob=keep_iob)
            all_tokens.extend(data_tokens)
            all_iobs.extend(iobs)
            all_valences.extend(valences)
            all_masks.extend(mask)
        return all_tokens, all_iobs, all_valences, all_masks

    def __init__(self, data_files, keep_iob=True):
        self.tokens, self.iobs, self.valences, self.masks = DataLoader.load_data_files(data_files, keep_iob=keep_iob)


def main():
    # Use a breakpoint in the code line below to debug your script.
    parser = create_argument_parser()
    args = parser.parse_args()

    data_object = DataLoader(args.data_files)
    for tokens, labels, mask in zip(data_object.tokens, data_object.iobs, data_object.masks):

        print(tokens,labels, mask)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
