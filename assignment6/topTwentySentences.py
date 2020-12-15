# Harry Chong
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import tensorflow_addons as tfa
import seaborn as sn
import random
from tensorflow import keras
from preprocessDefinition import preprocess
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras import backend as K

# README: I could not load the most recent dataset (20200301.en) with tensorflow-dataset version 4.1.0 as I kept getting an error message
# regarding Apache Beam and how the dataset is too large. In order for me to get around this issue, I had to downgrade my tensorflow_datasets 
# package to version 2.0.0. Doing so allowed me to load the dataset into jupyter notebook.
dataset, info = tfds.load("wikipedia/20190301.en", split='train' ,with_info=True)
#dataset_size = info.splits["train"].num_examples # Total size = 5,824,596

train_set = dataset.take(100000)

# Preprocess article content and title
raw_x_train = [] # Article Content 
raw_y_train = [] # Article Title

for x in train_set.batch(1).map(preprocess):
    xnew = raw_x_train.extend(x[0].to_list())
    ynew = raw_y_train.extend(x[1].to_list())

max_source_len=75
max_target_len=10

# Tokenize article content
input_tokenizer = Tokenizer()
input_tokenizer.fit_on_texts(list(raw_x_train))
input_integer_seq = input_tokenizer.texts_to_sequences(raw_x_train)
    
word2idx_inputs = input_tokenizer.word_index
print("Total unique words in the input: {}".format(len(word2idx_inputs)))

max_input_len = max(len(content) for content in input_integer_seq)
print("Length of longest sentence in input: {}\n".format(max_input_len))
      
# Tokenize article title
output_tokenizer = Tokenizer(filters='')
output_tokenizer.fit_on_texts(list(raw_y_train))
output_integer_seq = output_tokenizer.texts_to_sequences(raw_y_train)

word2idx_outputs = output_tokenizer.word_index
print("Total unique words in the output: {}".format(len(word2idx_outputs)))

num_words_output = len(word2idx_outputs) + 1
max_out_len = max(len(sen) for sen in output_integer_seq)
print("Length of longest sentence in the output: {}".format(max_out_len))

encoder_input_sequences = pad_sequences(input_integer_seq, maxlen=max_source_len, padding='post')
decoder_input_sequences = pad_sequences(output_integer_seq, maxlen=max_target_len, padding='post')

x_vocab = len(input_tokenizer.word_index) + 1
y_vocab = len(output_tokenizer.word_index) + 1

# Attention Layer pulled from here: https://github.com/thushv89/attention_keras/blob/master/src/layers/attention.py
class AttentionLayer(Layer):
    """
    This class implements Bahdanau attention (https://arxiv.org/pdf/1409.0473.pdf).
    There are three sets of weights introduced W_a, U_a, and V_a
     """

    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # Create a trainable weight variable for this layer.

        self.W_a = self.add_weight(name='W_a',
                                   shape=tf.TensorShape((input_shape[0][2], input_shape[0][2])),
                                   initializer='uniform',
                                   trainable=True)
        self.U_a = self.add_weight(name='U_a',
                                   shape=tf.TensorShape((input_shape[1][2], input_shape[0][2])),
                                   initializer='uniform',
                                   trainable=True)
        self.V_a = self.add_weight(name='V_a',
                                   shape=tf.TensorShape((input_shape[0][2], 1)),
                                   initializer='uniform',
                                   trainable=True)

        super(AttentionLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs, verbose=False):
        """
        inputs: [encoder_output_sequence, decoder_output_sequence]
        """
        assert type(inputs) == list
        encoder_out_seq, decoder_out_seq = inputs
        if verbose:
            print('encoder_out_seq>', encoder_out_seq.shape)
            print('decoder_out_seq>', decoder_out_seq.shape)

        def energy_step(inputs, states):
            """ Step function for computing energy for a single decoder state
            inputs: (batchsize * 1 * de_in_dim)
            states: (batchsize * 1 * de_latent_dim)
            """

            assert_msg = "States must be an iterable. Got {} of type {}".format(states, type(states))
            assert isinstance(states, list) or isinstance(states, tuple), assert_msg

            """ Some parameters required for shaping tensors"""
            en_seq_len, en_hidden = encoder_out_seq.shape[1], encoder_out_seq.shape[2]
            de_hidden = inputs.shape[-1]

            """ Computing S.Wa where S=[s0, s1, ..., si]"""
            # <= batch size * en_seq_len * latent_dim
            W_a_dot_s = K.dot(encoder_out_seq, self.W_a)

            """ Computing hj.Ua """
            U_a_dot_h = K.expand_dims(K.dot(inputs, self.U_a), 1)  # <= batch_size, 1, latent_dim
            if verbose:
                print('Ua.h>', U_a_dot_h.shape)

            """ tanh(S.Wa + hj.Ua) """
            # <= batch_size*en_seq_len, latent_dim
            Ws_plus_Uh = K.tanh(W_a_dot_s + U_a_dot_h)
            if verbose:
                print('Ws+Uh>', Ws_plus_Uh.shape)

            """ softmax(va.tanh(S.Wa + hj.Ua)) """
            # <= batch_size, en_seq_len
            e_i = K.squeeze(K.dot(Ws_plus_Uh, self.V_a), axis=-1)
            # <= batch_size, en_seq_len
            e_i = K.softmax(e_i)

            if verbose:
                print('ei>', e_i.shape)

            return e_i, [e_i]

        def context_step(inputs, states):
            """ Step function for computing ci using ei """

            assert_msg = "States must be an iterable. Got {} of type {}".format(states, type(states))
            assert isinstance(states, list) or isinstance(states, tuple), assert_msg

            # <= batch_size, hidden_size
            c_i = K.sum(encoder_out_seq * K.expand_dims(inputs, -1), axis=1)
            if verbose:
                print('ci>', c_i.shape)
            return c_i, [c_i]

        fake_state_c = K.sum(encoder_out_seq, axis=1)
        fake_state_e = K.sum(encoder_out_seq, axis=2)  # <= (batch_size, enc_seq_len, latent_dim

        """ Computing energy outputs """
        # e_outputs => (batch_size, de_seq_len, en_seq_len)
        last_out, e_outputs, _ = K.rnn(
            energy_step, decoder_out_seq, [fake_state_e],
        )

        """ Computing context vectors """
        last_out, c_outputs, _ = K.rnn(
            context_step, e_outputs, [fake_state_c],
        )

        return c_outputs, e_outputs

    def compute_output_shape(self, input_shape):
        """ Outputs produced by the layer """
        return [
            tf.TensorShape((input_shape[1][0], input_shape[1][1], input_shape[1][2])),
            tf.TensorShape((input_shape[1][0], input_shape[1][1], input_shape[0][1]))
        ]

# Load model
model = keras.models.load_model('wikipediaModel.h5', custom_objects={'AttentionLayer': AttentionLayer})

# Build up Model - use three stacked LSTM with attention layer (This is referenced for the Encoder/Decoder model)
latent_dim = max_source_len
embedding_dim = 100

# Encoder
encoder_inputs = keras.layers.Input(shape=(max_source_len))

# Embedding Layer
enc_emb =  Embedding(x_vocab,embedding_dim,trainable=True)(encoder_inputs)

#encoder lstm 1
encoder_lstm1 = LSTM(latent_dim,return_sequences=True,return_state=True,dropout=0.3,recurrent_dropout=0.3)
encoder_output1, state_h1, state_c1 = encoder_lstm1(enc_emb)

#encoder lstm 2
encoder_lstm2 = LSTM(latent_dim,return_sequences=True,return_state=True,dropout=0.3,recurrent_dropout=0.3)
encoder_output2, state_h2, state_c2 = encoder_lstm2(encoder_output1)

#encoder lstm 3
encoder_lstm3=LSTM(latent_dim, return_state=True, return_sequences=True,dropout=0.3,recurrent_dropout=0.3)
encoder_outputs, state_h, state_c= encoder_lstm3(encoder_output2)

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None,))

#embedding layer
dec_emb_layer = Embedding(y_vocab, embedding_dim,trainable=True)
dec_emb = dec_emb_layer(decoder_inputs)

decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True,dropout=0.2,recurrent_dropout=0.2)
decoder_outputs,decoder_fwd_state, decoder_back_state = decoder_lstm(dec_emb,initial_state=[state_h, state_c])

# Attention layer
attn_layer = AttentionLayer(name='attention_layer')
attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs])

# Concat attention input and decoder LSTM output
decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_out])

# Dense layer
decoder_dense =  TimeDistributed(Dense(y_vocab, activation='softmax'))
decoder_outputs = decoder_dense(decoder_concat_input)

# Define the model 
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.summary()

# Encode the input sequence to get the feature vector
encoder_model = Model(inputs=encoder_inputs,outputs=[encoder_outputs, state_h, state_c])

# Decoder setup
# Below tensors will hold the states of the previous time step
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_hidden_state_input = Input(shape=(max_source_len,latent_dim))

# Get the embeddings of the decoder sequence
dec_emb2= dec_emb_layer(decoder_inputs) 
# To predict the next word in the sequence, set the initial states to the states from the previous time step
decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=[decoder_state_input_h, decoder_state_input_c])

# Attention inference
attn_out_inf, attn_states_inf = attn_layer([decoder_hidden_state_input, decoder_outputs2])
decoder_inf_concat = Concatenate(axis=-1, name='concat')([decoder_outputs2, attn_out_inf])

# A dense softmax layer to generate prob dist. over the target vocabulary
decoder_outputs2 = decoder_dense(decoder_inf_concat) 

# Final decoder model
decoder_model = Model(
    [decoder_inputs] + [decoder_hidden_state_input,decoder_state_input_h, decoder_state_input_c],
    [decoder_outputs2] + [state_h2, state_c2])

# Build the dictionary to convert the index to word for target and source vocabulary:
reverse_target_word_index=output_tokenizer.index_word
reverse_source_word_index=input_tokenizer.index_word
target_word_index=output_tokenizer.word_index

# Define function below is the inference process
def decode_sequence(input_seq):
    # Encode the input as state vectors.
    e_out, e_h, e_c = encoder_model.predict(input_seq)
    
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))
    
    # Populate the first word of target sequence with the start word.
    target_seq[0, 0] = target_word_index[b'<start>']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
      
        output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_word_index[sampled_token_index]
        
        if(sampled_token!=b'<end>'):
            decoded_sentence += ' '+ str(sampled_token)

        # Exit condition: either hit max length or find stop word.
        if (sampled_token == b'<end>'  or len(decoded_sentence.split()) >= (max_target_len-1)):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index

        # Update internal states
        e_h, e_c = h, c

    return decoded_sentence

# Number of articles to generate a title
num_articles  = 20

def seq_to_target(input_seq):
    newString=''
    for i in input_seq:
        if((i!=0 and i!=target_word_index[b'<start>']) and i!=target_word_index[b'<end>']):
            newString=newString+str(reverse_target_word_index[i])+' '
    return newString

def seq_to_source(input_seq):
    newString=''
    for i in input_seq:
        if(i!=0):
            newString=newString+str(reverse_source_word_index[i])+' '
    return newString

for i in range(0,num_articles):
    print("Source Sentence:",seq_to_source(encoder_input_sequences[i]))
    print("Original Target:",seq_to_target(decoder_input_sequences[i]))
    print("Predicted Target:",decode_sequence(encoder_input_sequences[i].reshape(1,max_source_len)))
    print("\n")
