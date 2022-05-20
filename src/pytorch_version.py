from lib2to3.pgen2.tokenize import tokenize
import math
import joblib

import pytorch_lightning as pl
from spacy import Vocab
import torch
from sklearn.utils import shuffle
from torch import nn, utils
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torchtext
from positional_encodings import Summer, PositionalEncoding1D
from pytorch_lightning.loggers import TensorBoardLogger

logger = TensorBoardLogger("logs", name="my_model")
# tokenizer = get_tokenizer("basic_english")
# tokenizer.__call__ = lambda s: [char for char in s]


class CharTokenizer:
    def __init__(self):
        pass

    def __call__(self, s: str):
        return [c for c in s]

tokenizer = CharTokenizer()


class StarTrekDataset(utils.data.Dataset):
    def __init__(self, context_size: int) -> None:
        self.context_size = context_size
        self.x = []
        self.y = []

        with open("data/clean_lines.txt") as f:
            self.lines = [l.lower().strip() for l in f.readlines()]
            self.lines = [tokenizer(l) for l in self.lines]

        for line in self.lines:
            for idx in range(len(line) - self.context_size):
                self.x.append(line[idx : idx + self.context_size])
                self.y.append(line[idx + self.context_size])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def yield_tokens(data_iter):
    for text, _ in data_iter:
        yield tokenizer(" ".join(text))


def collate_batch(batch):
    x_processed = []
    y_processed = []
    for x, y in batch:
        x_processed.append(vocab(x))
        y_processed.append(vocab([y]))
    return torch.Tensor(x_processed).long(), torch.Tensor(y_processed).long().flatten()


# pytorch lightning module
class StarTrekModel(pl.LightningModule):
    def __init__(self, context_size: int, emb_dim: int = 16) -> None:
        super().__init__()
        self.context_size = context_size
        self.emb_dim = emb_dim
        self.add_pos_encoding = Summer(PositionalEncoding1D(emb_dim))
        self.embedding = nn.Embedding(len(vocab), emb_dim)
        self.transformer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=2, dim_feedforward=128
        )
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(
            in_features=self.context_size * self.emb_dim, out_features=len(vocab)
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.add_pos_encoding(x)
        x = self.transformer(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log("train_loss", loss)
        return {"loss": loss}

    # def validation_step(self, batch, batch_idx):
    #     x, y = batch
    #     y_hat = self.forward(x)
    #     loss = nn.CrossEntropyLoss()(y_hat, y)
    #     return {"val_loss": loss}

    # def validation_epoch_end(self, outputs):
    #     avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
    #     return {"val_loss": avg_loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

CONTEXT_SIZE = 32
ds = StarTrekDataset(context_size=CONTEXT_SIZE)

REBUILD = False
if REBUILD:
    vocab = build_vocab_from_iterator(yield_tokens(ds), specials=["<unk>"], min_freq=25)
    vocab.set_default_index(vocab["<unk>"])
    joblib.dump(vocab, "data/vocab.joblib")
    print("Saved vocab.")
else:
    vocab: torchtext.vocab.Vocab = joblib.load("data/vocab.joblib")
    print("Loaded vocab.")

print(f"vocab size = {len(vocab)}")


train_loader = utils.data.DataLoader(
    ds, batch_size=512, shuffle=True, collate_fn=collate_batch, num_workers=12
)

model = StarTrekModel(context_size=CONTEXT_SIZE)
trainer = pl.Trainer(logger=logger, max_epochs=10)
trainer.fit(model, train_loader)


print("end")
