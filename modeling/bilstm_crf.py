from typing import Optional
import torch.nn as nn
from data_loading.torch_loader import BILSTMDataset
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor
from torch.optim import Optimizer
from torchcrf import CRF
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader

import argparse
import torch
import fasttext.util


# It loads the external Word Embeddings (eg FastText, word2vect etc.) into the embedding layer of pytorch
def custom_emb_layer(weights_matrix, non_trainable=False):
    num_embeddings, embedding_dim =  weights_matrix.get_input_matrix().shape
    emb_layer = nn.Embedding(num_embeddings, embedding_dim,padding_idx=0)
    emb_layer.load_state_dict({'weight': torch.tensor(weights_matrix.get_input_matrix())})
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim


class CRF_LSTM(LightningModule):
    def __init__(self, weights_matrix, n_targets, seq_len, num_embeddings):
        super(CRF_LSTM, self).__init__()
        #self.truncated_bptt_steps = 5

        #self.embedding, num_embeddings, self.embedding_dim = custom_emb_layer(weights_matrix, True)
        self.embedding_dim = 256
        self.hidden = self.embedding_dim
        self.embedding = nn.Embedding(num_embeddings,self.embedding_dim,padding_idx=0)
        self.utt_encoder = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden // 2, bidirectional=True, num_layers=1)
        self.hid2tag = nn.Linear(self.hidden, n_targets)
        self.crf = CRF(n_targets)  # Number of output labels
        self.dropout = nn.Dropout(0.2)
        self.seq_len = seq_len


    def training_step(self, batch, batch_idx,hiddens) -> STEP_OUTPUT:
        x = batch[0]
        mask = batch[1]
        seq_len = batch[2]
        target = batch[3]

        embs = self.dropout(self.embedding(x).permute(1, 0, 2))  # sequence len, batch_size, embedding size
        packed_input = pack_padded_sequence(embs, seq_len)
        packed_output, (_, _) = self.utt_encoder(packed_input)
        utt_encoded, input_sizes = pad_packed_sequence(packed_output)
        targets = target.permute(1, 0)
        mask_pad = mask.permute(1, 0)  # 0 for pad tokens 1 for the rest
        utt_encoded = self.hid2tag(utt_encoded)
        loss = self.crf(utt_encoded, targets, mask=mask_pad) * -1  # Make the loss positive
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:

        source = batch[0]
        mask = batch[1]
        seq_len = batch[2]
        target = batch[3]

        embedded = self.embedding(source)
        embs = self.dropout(embedded.permute(1,0,2))  # sequence len, batch_size, embedding size
        packed_input = pack_padded_sequence(embs, seq_len.cpu().numpy())
        self.log("The packed input is ", embs.shape)
        packed_output, (ht, ct) = self.utt_encoder(packed_input)
        utt_encoded, input_sizes = pad_packed_sequence(packed_output)
        targets = target.permute(1, 0)
        mask_pad = mask.permute(1, 0)  # 0 for pad tokens 1 for the rest
        utt_encoded = self.hid2tag(utt_encoded)
        loss = self.crf(utt_encoded, targets, mask=mask_pad) * -1  # Make the loss positive
        self.log("val_loss", loss)
        return loss

    def forward(self, batch, batch_idx):
        source = batch[0]
        mask = batch[1]
        seq_len = batch[2]
        target = batch[3]

        embs = self.embedding(source)
        embs = embs.permute(source).permute(1,0,2)  # sequence len, batch_size, embedding size
        utt_encoded, (_, _) = self.utt_encoder(embs)
        utt_encoded = self.hid2tag(utt_encoded)
        return self.crf.decode(utt_encoded)

    def backward(
        self, loss: Tensor, optimizer: Optional[Optimizer], optimizer_idx: Optional[int], *args, **kwargs
    ) -> None:
        loss.backward()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=1e-3)
        return optimizer


def create_argument_parser():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--train','--training-files',  action="store", dest="training_files",nargs='+',
                        help='the training file')
    parser.add_argument('--val','--val-files',  action="store", dest="val_files",nargs='+',
                        help='the test file')

    return parser


def main():
    # Use a breakpoint in the code line below to debug your script.
    parser = create_argument_parser()
    args = parser.parse_args()

    fasttext.util.download_model('it', if_exists='ignore')
    ft = fasttext.load_model('cc.it.300.bin')

    training_data = BILSTMDataset(args.training_files) # data loader required here
    val_data = BILSTMDataset(args.val_files)

    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=64, shuffle=True)

    print(training_data.token_ids[0], training_data.seq_len) # the sequence length value is simply wrong
    """
    tensor([  2,  17,   1, 249,   2,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       dtype=torch.int32) tensor(3417)
       """

    model = CRF_LSTM(ft, n_targets=3, seq_len=training_data.seq_len, num_embeddings=len(training_data.vocab))

    trainer = Trainer()

    trainer.fit(model, train_dataloader, val_dataloader)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
