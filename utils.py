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

def vectorize_text(text):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text)