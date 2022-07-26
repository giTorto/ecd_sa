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
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score
from evaluation.span_prediction import unified_evaluation


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
    def __init__(self, weights_matrix, n_targets, seq_len, num_embeddings, iob_mapping=None):
        super(CRF_LSTM, self).__init__()
        #self.truncated_bptt_steps = 10

        #self.embedding, num_embeddings, self.embedding_dim = custom_emb_layer(weights_matrix, True)
        self.embedding_dim = 256
        self.hidden = self.embedding_dim
        self.embedding = nn.Embedding(num_embeddings,self.embedding_dim,padding_idx=0)
        self.utt_encoder = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden // 2, bidirectional=True, num_layers=1)
        self.hid2tag = nn.Linear(self.hidden, n_targets)
        self.crf = CRF(n_targets)  # Number of output labels
        self.dropout = nn.Dropout(0.2)
        self.seq_len = seq_len
        self.iob_mapping = iob_mapping

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
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
        packed_output, (ht, ct) = self.utt_encoder(packed_input)
        utt_encoded, input_sizes = pad_packed_sequence(packed_output)
        targets = target.permute(1, 0)
        mask_pad = mask.permute(1, 0)  # 0 for pad tokens 1 for the rest
        utt_encoded = self.hid2tag(utt_encoded)
        loss = self.crf(utt_encoded, targets, mask=mask_pad) * -1  # Make the loss positive
        self.log("val_loss", loss)

        class_report = self.evaluate_it(targets, utt_encoded, mask, val_f1=True)

        return loss

    def forward(self, X):
        source = X[0]
        mask = X[1]
        seq_len = X[2]
        target = X[3]

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

    @staticmethod
    def truncate_it(targets, mask):
        mask = mask.tolist()
        final_pred = []
        for pred, mask in zip(targets, mask):

            truncate_here = mask.index(0)
            final_pred.append(pred[truncate_here:])
        return final_pred

    def evaluate_it(self, targets, utt_encoded, mask, val_f1=False):

        targets_reversed = [self.iob_mapping.lookup_tokens(x) for x in targets.permute(1,0).tolist()]
        predictions_reversed = [self.iob_mapping.lookup_tokens(x)  for x in self.crf.decode(utt_encoded)]

        targets_reversed = self.truncate_it(targets_reversed, mask)
        predictions_reversed = self.truncate_it(predictions_reversed, mask)

        class_report = classification_report(targets_reversed, predictions_reversed, output_dict=True)
        f1_score_result = f1_score(targets_reversed, predictions_reversed)
        self.log_dict(class_report)
        if val_f1:
            self.log("val_f1", f1_score_result)
        return class_report

    def test_step(self, batch, batch_idx) -> STEP_OUTPUT:

        source = batch[0]
        mask = batch[1]
        seq_len = batch[2]
        target = batch[3]

        embedded = self.embedding(source)
        embs = self.dropout(embedded.permute(1,0,2))  # sequence len, batch_size, embedding size
        packed_input = pack_padded_sequence(embs, seq_len.cpu().numpy())
        packed_output, (ht, ct) = self.utt_encoder(packed_input)
        utt_encoded, input_sizes = pad_packed_sequence(packed_output)
        targets = target.permute(1, 0)
        mask_pad = mask.permute(1, 0)  # 0 for pad tokens 1 for the rest
        utt_encoded = self.hid2tag(utt_encoded)
        loss = self.crf(utt_encoded, targets, mask=mask_pad) * -1  # Make the loss positive
        self.log("test_loss", loss)
        class_report = self.evaluate_it(targets, utt_encoded, mask)
        return class_report

    def predict_step(self, batch, batch_idx, **kwargs):
        source = batch[0]
        mask = batch[1]

        embs = self.embedding(source)
        embs = embs.permute(1,0,2)  # sequence len, batch_size, embedding size
        utt_encoded, (_, _) = self.utt_encoder(embs)
        utt_encoded = self.hid2tag(utt_encoded)
        crf_decoded = self.crf.decode(utt_encoded)
        crf_decoded = self.truncate_it(crf_decoded,mask)
        predictions_reversed = [self.iob_mapping.lookup_tokens(x) for x in crf_decoded]

        return predictions_reversed


def create_argument_parser():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--train','--training-files',  action="store", dest="training_files",nargs='+',
                        help='the training file')
    parser.add_argument('--val','--val-files',  action="store", dest="val_files",nargs='+',
                        help='the val file')
    parser.add_argument('--test','--test-files',  action="store", dest="test_files",nargs='+',
                        help='the test file')


    return parser


def main():
    # Use a breakpoint in the code line below to debug your script.
    parser = create_argument_parser()
    args = parser.parse_args()

    fasttext.util.download_model('it', if_exists='ignore')
    ft = fasttext.load_model('cc.it.300.bin')

    training_data = BILSTMDataset(args.training_files) # data loader required here
    val_data = BILSTMDataset(args.val_files, token_vocab=training_data.vocab, iob_mapping_vocab=training_data.iob_mapping)
    test_data = BILSTMDataset(args.test_files, token_vocab=training_data.vocab, iob_mapping_vocab=training_data.iob_mapping)

    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

    #print(training_data.token_ids[0], training_data.seq_len) # the sequence length value is simply wrong

    model = CRF_LSTM(ft, n_targets=len(training_data.iob_mapping), seq_len=training_data.seq_len,
                     num_embeddings=len(training_data.vocab), iob_mapping=training_data.iob_mapping)

    trainer = Trainer(max_epochs=20, callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=5)])

    trainer.fit(model, train_dataloader, val_dataloader)

    trainer.save_checkpoint("./checkpoints")

    results = trainer.test(model, ckpt_path='best', dataloaders=test_dataloader)
    print(results)

    predictions = trainer.predict(model, ckpt_path='best', dataloaders=test_dataloader)
    for i,batch in enumerate(predictions):
        for pred in batch:
            print(len(pred))



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
