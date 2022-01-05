import tensorflow as tf
from tensorflow import keras
import tensorflow_text as text
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab
import boto3
from model_definition import SequenceModel

bucket = boto3.resource("s3").Bucket("deep-text-generation")

with open("data/clean_lines.txt") as f:
    data: list[str] = [line.strip() for line in f.readlines()]

RECREATE_VOCAB = False

if RECREATE_VOCAB:
    vocab = bert_vocab.bert_vocab_from_dataset(
        data,
        vocab_size=5000,
        reserved_tokens=["[PAD]", "[UNK]"],
        bert_tokenizer_params={"lower_case": True},
    )

    with open("data/vocab.txt", "w") as f:
        f.write("\n".join(vocab))

    bucket.upload_file("data/vocab.txt", "StarTrek/vocab.txt")

else:
    bucket.download_file("StarTrek/vocab.txt", "data/vocab.txt")

tokenizer = text.BertTokenizer("data/vocab.txt", lower_case=True)

with open("data/vocab.txt") as f:
    vocab_size = len(f.readlines())


def process_and_split_dataset(data_list: list[str], tokenizer):
    ds_len = len(data_list)
    train_size = int(ds_len * 0.8)
    val_size = int(ds_len * 0.2)

    # to dataset & processing
    data = tokenizer.tokenize(data_list)
    data = tf.data.Dataset.from_tensor_slices((data, data))
    data = data.map(lambda x, y: (x.flat_values[:-1], y.flat_values[1:]))

    # splitting
    data = data.shuffle(buffer_size=4096, seed=42)
    train = data.take(train_size)
    val = data.skip(train_size).take(val_size)

    return train, val


BATCH_SIZE = 64
train, val = process_and_split_dataset(data, tokenizer=tokenizer)
train = train.padded_batch(BATCH_SIZE).prefetch(32)
val = val.padded_batch(BATCH_SIZE).prefetch(32)

model = SequenceModel(
    vocab_size=vocab_size, embedding_dim=128, recurrent_size=128, hidden_size=128
)

model.compile("adam", "sparse_categorical_crossentropy")
model.fit(train, validation_data=val, epochs=10)

print("done")