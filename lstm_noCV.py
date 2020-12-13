from data_handler import get_data
import argparse
import os
os.environ['KERAS_BACKEND'] = 'theano'
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Input, LSTM
from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Input, Merge, Convolution1D, MaxPooling1D, GlobalMaxPooling1D
import numpy as np
import pdb
from sklearn.metrics import make_scorer, f1_score, accuracy_score, recall_score, precision_score, classification_report, precision_recall_fscore_support
from sklearn.ensemble  import GradientBoostingClassifier, RandomForestClassifier
from gensim.parsing.preprocessing import STOPWORDS
from sklearn.model_selection import KFold
from keras.utils import np_utils, to_categorical
import codecs
import operator
import gensim, sklearn
from string import punctuation
from collections import defaultdict
from batch_gen import batch_gen
import sys
import matplotlib.pyplot as plt

from nltk import tokenize as tokenize_nltk
from my_tokenizer import glove_tokenize



### Preparing the text data
texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids

# vocab generation
vocab, reverse_vocab = {}, {}
freq = defaultdict(int)
tweets = {}



EMBEDDING_DIM = None
GLOVE_MODEL_FILE = None
SEED = 42
NO_OF_FOLDS = 10
CLASS_WEIGHT = None
LOSS_FUN = None
OPTIMIZER = None
KERNEL = None
TOKENIZER = None
MAX_SEQUENCE_LENGTH = None
INITIALIZE_WEIGHTS_WITH = None
LEARN_EMBEDDINGS = None
EPOCHS = 10
BATCH_SIZE = 512
SCALE_LOSS_FUN = None

word2vec_model = None



def get_embedding(word):
    #return
    try:
        return word2vec_model[word]
    except Exception as e:
        print('Encoding not found: %s' %(word))
        return np.zeros(EMBEDDING_DIM)

def get_embedding_weights():
    embedding = np.zeros((len(vocab) + 1, EMBEDDING_DIM))
    n = 0
    for k, v in vocab.items():
        try:
            embedding[v] = word2vec_model[k]
        except:
            n += 1
            pass
    print("%d embedding missed"%n)
    return embedding


def select_tweets():
    # selects the tweets as in mean_glove_embedding method
    # Processing
    tweets = get_data()
    X, Y = [], []
    tweet_return = []
    for tweet in tweets:
        _emb = 0
        words = TOKENIZER(tweet['text'].lower())
        for w in words:
            if w in word2vec_model:  # Check if embeeding there in GLove model
                _emb+=1
        if _emb:   # Not a blank tweet
            tweet_return.append(tweet)
    print('Tweets selected:', len(tweet_return))
    #pdb.set_trace()
    return tweet_return


def gen_vocab():
    # Processing
    vocab_index = 1
    #print('-------------------------')
    #print('GEN VOCAB')
    #print('-------------------------')
    for tweet in tweets:
        #print(tweet)
        text = TOKENIZER(tweet['text'].lower())
        text = ' '.join([c for c in text if c not in punctuation])
        words = text.split()
        words = [word for word in words if word not in STOPWORDS]
        #print(words)
        for word in words:
            #print(word)
            if word not in vocab:
                vocab[word] = vocab_index
                reverse_vocab[vocab_index] = word       # generate reverse vocab as well
                vocab_index += 1
            freq[word] += 1
    vocab['UNK'] = len(vocab) + 1
    reverse_vocab[len(vocab)] = 'UNK'


def filter_vocab(k):
    global freq, vocab
    pdb.set_trace()
    freq_sorted = sorted(list(freq.items()), key=operator.itemgetter(1))
    tokens = freq_sorted[:k]
    vocab = dict(list(zip(tokens, list(range(1, len(tokens) + 1)))))
    vocab['UNK'] = len(vocab) + 1


def gen_sequence():
    y_map = {
            'none': 0,
            'racism': 1,
            'sexism': 2
            }
    #print('-------------------------')
    #print('GEN SEQUENCE')
    #print('-------------------------')
    X, y = [], []
    for tweet in tweets:
        #print(tweet)
        text = TOKENIZER(tweet['text'].lower())
        text = ' '.join([c for c in text if c not in punctuation])
        #print(text)
        words = text.split()
        words = [word for word in words if word not in STOPWORDS]
        #print(words)
        seq, _emb = [], []
        #print(words)
        for word in words:
            seq.append(vocab.get(word, vocab['UNK']))
        X.append(seq)
        #print(seq)
        y.append(y_map[tweet['label']])
    
    return X, y


def shuffle_weights(model):
    weights = model.get_weights()
    weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
    model.set_weights(weights)

def lstm_model(sequence_length, embedding_dim):
    model_variation = 'LSTM'
    print(('Model variation is %s' % model_variation))
    model = Sequential()
    model.add(Embedding(len(vocab)+1, embedding_dim, input_length=sequence_length, trainable=LEARN_EMBEDDINGS))
    model.add(Dropout(0.25))#, input_shape=(sequence_length, embedding_dim)))
    model.add(LSTM(50))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss=LOSS_FUN, optimizer=OPTIMIZER, metrics=['accuracy'])

    """
    model.add(Embedding(len(vocab)+1, embedding_dim, input_length=sequence_length, trainable=LEARN_EMBEDDINGS))
    model.add(Dropout(0.25))#, input_shape=(sequence_length, embedding_dim)))
    model.add(LSTM(50))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))
    model.add(Activation('softmax'))
    model.compile(loss=LOSS_FUN, optimizer=OPTIMIZER, metrics=['accuracy'])
    """
    print(model.summary())
    return model


def train_LSTM(X, y, model, inp_dim, weights, epochs=EPOCHS, batch_size=BATCH_SIZE):
    ## 
    ## THIS DOESN"T HAVE TRAIN/TEST SPLIT YET?
    ##
    
    # This way ignore the k-folds method and tracks the loss/accuracy over time
    #print(y.shape)
    y = to_categorical(y)
    print(y.shape)
    history = model.fit(X, y, validation_split=0.1, epochs=10, batch_size=batch_size)
    print(history.history.keys())
    # Plot
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('logging/acc.png')
    plt.clf()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('logging/loss.png')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LSTM based models for twitter Hate speech detection')
    parser.add_argument('-f', '--embeddingfile', required=True)
    parser.add_argument('-d', '--dimension', required=True)
    parser.add_argument('--tokenizer', choices=['glove', 'nltk'], required=True)
    parser.add_argument('--loss', default=LOSS_FUN, required=True)
    parser.add_argument('--optimizer', default=OPTIMIZER, required=True)
    parser.add_argument('--epochs', default=EPOCHS, required=True)
    parser.add_argument('--batch-size', default=BATCH_SIZE, required=True)
    parser.add_argument('-s', '--seed', default=SEED)
    parser.add_argument('--folds', default=NO_OF_FOLDS)
    parser.add_argument('--kernel', default=KERNEL)
    parser.add_argument('--class_weight')
    parser.add_argument('--initialize-weights', choices=['random', 'glove'], required=True)
    parser.add_argument('--learn-embeddings', action='store_true', default=False)
    parser.add_argument('--scale-loss-function', action='store_true', default=False)


    args = parser.parse_args()
    GLOVE_MODEL_FILE = args.embeddingfile
    EMBEDDING_DIM = int(args.dimension)
    SEED = int(args.seed)
    NO_OF_FOLDS = int(args.folds)
    CLASS_WEIGHT = args.class_weight
    LOSS_FUN = args.loss
    OPTIMIZER = args.optimizer
    KERNEL = args.kernel
    if args.tokenizer == "glove":
        TOKENIZER = glove_tokenize
    elif args.tokenizer == "nltk":
        TOKENIZER = tokenize_nltk.casual.TweetTokenizer(strip_handles=True, reduce_len=True).tokenize
    INITIALIZE_WEIGHTS_WITH = args.initialize_weights    
    LEARN_EMBEDDINGS = args.learn_embeddings
    EPOCHS = int(args.epochs)
    BATCH_SIZE = int(args.batch_size)
    SCALE_LOSS_FUN = args.scale_loss_function



    np.random.seed(SEED)
    print('GLOVE embedding: %s' %(GLOVE_MODEL_FILE))
    print('Embedding Dimension: %d' %(EMBEDDING_DIM))
    print('Allowing embedding learning: %s' %(str(LEARN_EMBEDDINGS)))

    #word2vec_model = gensim.models.Word2Vec.load_word2vec_format(GLOVE_MODEL_FILE)
    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(GLOVE_MODEL_FILE)

    tweets = select_tweets()
    gen_vocab()
    #filter_vocab(20000)
    X, y = gen_sequence()
    #print(X)
    #print(y)
    #Y = y.reshape((len(y), 1))
    MAX_SEQUENCE_LENGTH = max([len(x) for x in X])
    
    print("max seq length is %d"%(MAX_SEQUENCE_LENGTH))

    data = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
    y = np.array(y)
    data, y = sklearn.utils.shuffle(data, y)
    W = get_embedding_weights()

    model = lstm_model(data.shape[1], EMBEDDING_DIM)
    #model = lstm_model(data.shape[1], 25, get_embedding_weights())
    train_LSTM(data, y, model, EMBEDDING_DIM, W)
    
    # Save model
    model.save('models/lstm_noCV_random_dim200_epoch10_batch128.h5')
    
    # save embeddings
    embeddings = model.layers[0].get_weights()[0]
    np.save('models/EMBEDDINGS_lstm_noCV_random_dim200_epoch10_batch128.npy', embeddings)

    pdb.set_trace()
