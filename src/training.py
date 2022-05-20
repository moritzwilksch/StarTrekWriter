import boto3
import tensorflow as tf
import tensorflow_text as text
import yaml
from rich.console import Console
from tensorflow import keras
from tensorflow.keras import mixed_precision
from tensorflow_text.tools.wordpiece_vocab import \
    bert_vocab_from_dataset as bert_vocab

from model_definition import SequenceModel, SequenceModelWithAttention

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

    return train, val, data


BATCH_SIZE = config.get("batch_size")
train, val, total_data = process_and_split_dataset(data, tokenizer=tokenizer)
train = train.padded_batch(BATCH_SIZE).prefetch(32)
val = val.padded_batch(BATCH_SIZE).prefetch(32)
total_data = total_data.padded_batch(BATCH_SIZE).prefetch(32)


# -------------------------- Model Setup ---------------------------------
# model = SequenceModel(
model = SequenceModelWithAttention(
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

    if progress < 0.75:
        return config.get("learning_rate")
    else:
        return config.get("learning_rate") / 10


callbacks = (
    [keras.callbacks.LearningRateScheduler(scheduler)]
    if config.get("use_scheduler", False)
    else []
)

model.fit(
    train,
    epochs=config.get("epochs"),
    validation_data=val,
    callbacks=callbacks,
    verbose=1,
)


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

    final_text = " ".join([x.decode("utf-8") for x in final_words.numpy()[0]])

    for punct in ",.!?;:'":
        final_text = final_text.replace(f" {punct}", f"{punct}")

    for open_brackets in "([{":
        final_text = final_text.replace(f"{open_brackets} ", f"{open_brackets}")

    for closing_brackets in "}])":
        final_text = final_text.replace(f" {closing_brackets}", f"{closing_brackets}")

    for name in ["picard", "riker", "laforge", "worf", "data", "crusher"]:
        final_text = final_text.replace(f"{name}:", f"{name.upper()}:")

    return final_text.replace("' ", "'")


print(generate_from_model(model, "[BRIDGE]", temperature=0.8))
