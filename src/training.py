import tensorflow as tf
from tensorflow import keras
import tensorflow_text as text
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab
import boto3
from model_definition import SequenceModel
import yaml
from rich.console import Console
from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy("mixed_float16")


c = Console()

with open("src/modelconfig.yaml", "r") as f:
    config = yaml.safe_load(f)["default"]

c.print(f"Using config = {config}")


bucket = boto3.resource("s3").Bucket("deep-text-generation")

with open("data/clean_lines.txt") as f:
    data: list[str] = [line.strip() for line in f.readlines()]

RECREATE_VOCAB = False

if RECREATE_VOCAB:
    vocab = bert_vocab.bert_vocab_from_dataset(
        tf.data.TextLineDataset("data/clean_lines.txt"),
        vocab_size=1000,
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


# -------------------------- Data Setup ---------------------------------
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


BATCH_SIZE = config.get("batch_size")
train, val = process_and_split_dataset(data, tokenizer=tokenizer)
train = train.padded_batch(BATCH_SIZE).prefetch(32)
val = val.padded_batch(BATCH_SIZE).prefetch(32)


# -------------------------- Model Setup ---------------------------------
model = SequenceModel(
    vocab_size=vocab_size,
    embedding_dim=config.get("embedding_dim"),
    recurrent_size=config.get("recurrent_size"),
    hidden_size=config.get("hidden_size"),
)

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
opt = tf.keras.optimizers.Adam(learning_rate=config.get("learning_rate"))

model.compile(opt, loss, metrics=["accuracy"])

# -------------------------- Model Training ---------------------------------


def scheduler(epoch, lr):
    progress = epoch / config.get("epochs")

    if progress < 0.5:
        print("Learning rate:", lr)
        return lr
    else:
        print("Learning rate:", lr / 10)
        return lr / 10

callbacks = [keras.callbacks.LearningRateScheduler(scheduler)] if config.get("lr_scheduler", False) else []

model.fit(
    train,
    epochs=config.get("epochs"),
    validation_data=val,
    callbacks=callbacks,
    verbose=1,
)

model.fit(train, validation_data=val, epochs=config.get("epochs"), callbacks=callbacks)


def generate_from_model(
    model: tf.keras.Model, seed: str, temperature: float = 1, total_tokens: int = 100
):
    tokens = tokenizer.tokenize([seed]).flat_values

    while len(tokens) < total_tokens:
        # generate next token
        prediction = model(tokens[None, :], training=False)[:, -1, :]
        prediction = tf.nn.softmax(prediction / temperature)
        token = tf.random.categorical(tf.math.log(prediction), num_samples=1)[0]
        token = token.numpy()[0]
        tokens = tf.concat([tokens, tf.constant([token], dtype=tf.int64)], 0)

    final_words = tokenizer.detokenize(tokens[None, :])
    return " ".join([x.decode("utf-8") for x in final_words.numpy()[0]])


print(generate_from_model(model, "[BRIDGE]", temperature=0.8))

