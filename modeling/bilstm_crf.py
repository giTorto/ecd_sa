import torch.nn as nn
from torchcrf import CRF
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from pytorch_lightning import LightningModule
import argparse
import torch

# It loads the external Word Embeddings (eg FastText, word2vect etc.) into the embedding layer of pytorch
def custom_emb_layer(weights_matrix, non_trainable=False):
    num_embeddings, embedding_dim = weights_matrix.size()
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim


class CRF_LSTM(nn.Module):
    def __init__(self, weights_matrix, n_targets):
        super(CRF_LSTM, self).__init__()
        self.hidden = 300
        self.embedding, num_embeddings, embedding_dim = custom_emb_layer(weights_matrix, True)
        self.utt_encoder = nn.LSTM(embedding_dim, self.hidden // 2, bidirectional=True, num_layers=1)
        self.hid2tag = nn.Linear(self.hidden, n_targets)
        self.crf = CRF(n_targets)  # Number of output labels
        self.dropout = nn.Dropout(0.2)

    def forward(self, inp):
        embs = self.dropout(self.embedding(inp['source']).permute(1, 0, 2))  # sequence len, batch_size, embedding size
        packed_input = pack_padded_sequence(embs, inp['seq_lengths'].cpu())
        packed_output, (_, _) = self.utt_encoder(packed_input)
        utt_encoded, input_sizes = pad_packed_sequence(packed_output)
        targets = inp['target'].permute(1, 0)
        mask_pad = inp['mask'].permute(1, 0)  # 0 for pad tokens 1 for the rest
        utt_encoded = self.hid2tag(utt_encoded)
        loss = self.crf(utt_encoded, targets, mask=mask_pad) * -1  # Make the loss positive
        return loss

    def decode(self, seq):
        embs = self.embedding(seq).permute(1, 0, 2)  # sequence len, batch_size, embedding size
        utt_encoded, (_, _) = self.utt_encoder(embs)
        utt_encoded = self.hid2tag(utt_encoded)
        return self.crf.decode(utt_encoded)


def create_argument_parser():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--train','--training-files',  action="store", dest="training_files",nargs='+',
                        help='the training file')
    parser.add_argument('--test','--test-files',  action="store", dest="test_files",nargs='+',
                        help='the test file')

    return parser


def main():
    # Use a breakpoint in the code line below to debug your script.
    parser = create_argument_parser()
    args = parser.parse_args()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
