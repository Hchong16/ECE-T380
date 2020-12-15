import tensorflow as tf

def preprocess(x_batch):
    x_batch['text'] = tf.strings.substr(x_batch['text'], 0, 300)
    x_batch['text'] = tf.strings.regex_replace(x_batch['text'], rb"<br\s*/?>", b" ")
    x_batch['text'] = tf.strings.regex_replace(x_batch['text'], b"[^a-zA-Z']", b" ")
    x_batch['text'] = tf.strings.split(x_batch['text'])
    
    x_batch['title'] = tf.strings.regex_replace(x_batch['title'], rb"<br\s*/?>", b" ")
    x_batch['title'] = tf.strings.regex_replace(x_batch['title'], b"[^a-zA-Z']", b" ")
    x_batch['title'] = '<start> '+x_batch['title']+' <end>'
    x_batch['title'] = tf.strings.split(x_batch['title'])

    return x_batch['text'], x_batch["title"]