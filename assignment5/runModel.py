# Harry Chong
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from collections import Counter

# Load IMDB Reviews
datasets, info = tfds.load("imdb_reviews", as_supervised=True, with_info=True)

# Setup vocabulary table
def preprocess(X_batch, y_batch):
    X_batch = tf.strings.substr(X_batch, 0, 300)
    X_batch = tf.strings.regex_replace(X_batch, rb"<br\s*/?>", b" ")
    X_batch = tf.strings.regex_replace(X_batch, b"[^a-zA-Z']", b" ")
    X_batch = tf.strings.regex_replace(X_batch, "<[^>]+>",  " ")
    X_batch = tf.strings.split(X_batch)
    return X_batch.to_tensor(default_value=b"<pad>"), y_batch

# Print some reviews and labels from dataset
for X_batch, y_batch in datasets["train"].batch(2).take(1):
    for review, label in zip(X_batch.numpy(), y_batch.numpy()):
        #print("Review:", review.decode("utf-8")[:200], "...")
        #print("Label:", label, "= Positive" if label else "= Negative")
        #print()
        pass
    
preprocess(X_batch, y_batch)

vocabulary = Counter()
for X_batch, y_batch in datasets["train"].batch(32).map(preprocess):
    for review in X_batch:
        vocabulary.update(list(review.numpy()))

# Tokenize each word to ID
vocab_size = 10000
truncated_vocabulary = [
    word for word, count in vocabulary.most_common()[:vocab_size]]

word_to_id = {word: index for index, word in enumerate(truncated_vocabulary)}
words = tf.constant(truncated_vocabulary)
word_ids = tf.range(len(truncated_vocabulary), dtype=tf.int64)
vocab_init = tf.lookup.KeyValueTensorInitializer(words, word_ids)
num_oov_buckets = 1000
table = tf.lookup.StaticVocabularyTable(vocab_init, num_oov_buckets)

LABELS = ["NEGATIVE", "POSITIVE"]
def preprocess_input(X_batch):
    X_batch = tf.strings.substr(X_batch, 0, 300)
    X_batch = tf.strings.regex_replace(X_batch, rb"<br\s*/?>", b" ")
    X_batch = tf.strings.regex_replace(X_batch, b"[^a-zA-Z']", b" ")
    X_batch = tf.strings.regex_replace(X_batch, "<[^>]+>",  " ")
    X_batch = tf.strings.split(X_batch)
    return X_batch.to_tensor(default_value=b"<pad>")

def get_prediction(review):
    # Preprocessing
    review_array = table.lookup(tf.constant([review.split()]))
    
    # Prediction score that the item is encoded as 1 (Positive)
    threshold_confidence = 0.5
    score = float(model.predict(review_array)[0][0])
    
    if score > threshold_confidence:
        actual_predict, actual_proba = "POSITIVE", round(score, 5)
        other_predict, other_proba = "NEGATIVE", round(1 - score, 5)
    else:
        actual_predict, actual_proba = "NEGATIVE", round(1 - score, 5)
        other_predict, other_proba = "POSITIVE", round(score, 5)
    
    print('Review:', review, '\nPrediction:', actual_predict, 
          '\nPredicted probability that the review is {}: {}'.format(actual_predict, actual_proba),
          '\nPredicted probabiltiy that the review is {}: {}\n'.format(other_predict, other_proba))
    
def encode_words(X_batch, y_batch):
    return table.lookup(X_batch), y_batch

# Load model
model = keras.models.load_model('model.h5')

# [USER PARAMETER] Number of IMDB reviews to pass into model for prediction.
num_review = 10

# Pull and shuffle the number of reviews from the imdb dataset, preprocess it, and
# evaluate on it.
data = tfds.load(name="imdb_reviews", split=('test'), as_supervised=True)
review, label = next(iter(data.shuffle(num_review).batch(num_review)))
preprocess(review, label)

# Predict and output result. Print out the prediction and predicted probability for each review.
for idx in range(num_review):
    get_prediction(review[idx].numpy())
