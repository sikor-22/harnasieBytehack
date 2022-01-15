import tensorflow as tf
import keras

MAX_FEATURES = 20000
EMBEDDING_DIM = 128
SEQUENCE_LENGTH = 500

def prepare_text(input_data):
    lowercase = tf.strings.lower(input_data)
    strip_username = tf.strings.regex_replace(lowercase, "\B\@\w+", "")
    return strip_username

vectorize_layer = keras.layers.TextVectorization(
    standardize = prepare_text,
    max_tokens = MAX_FEATURES,
    output_mode = "int",
    output_sequence_length = SEQUENCE_LENGTH
)

def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label

def def_model():
    inputs = keras.Input(shape=(None,), dtype="int64")
    x = keras.layers.Embedding(MAX_FEATURES, EMBEDDING_DIM)(inputs)
    x = keras.layers.Dropout(0.5)(x)
    #Conv1d
    x = keras.layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
    x = keras.layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
    x = keras.layers.GlobalMaxPooling1D()(x)
    #Dense
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.5)(x)
    predictions = keras.layers.Dense(1, activation="sigmoid", name="predictions")(x)
    return keras.Model(input, predictions)