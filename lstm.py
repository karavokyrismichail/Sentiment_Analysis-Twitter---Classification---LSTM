import numpy as np
import gensim
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding
from keras.models import Sequential
from keras.layers import Dense,LSTM,Dropout
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

def lstm(x_train, x_test, y_train,  y_test):
    #split the tweets
    texts = [text.split() for text in x_train]
    #building vocab and training word2vec model with the tweets
    w2v_model = gensim.models.word2vec.Word2Vec(vector_size=300, window=7, min_count=10, workers=8)
    w2v_model.build_vocab(texts)
    w2v_model.train(texts, total_examples=len(texts), epochs=10)
    #tokenize all the tweets
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(x_train)
    vocab_size=len(tokenizer.word_index)+1
    #uniform all the sequences in same length
    X_train = pad_sequences(tokenizer.texts_to_sequences(x_train), maxlen=300)
    X_test = pad_sequences(tokenizer.texts_to_sequences(x_test), maxlen=300)
    #creating embedding matrix
    embedding_matrix = np.zeros((vocab_size, 300))
    for word, i in tokenizer.word_index.items():
        if word in w2v_model.wv:
            embedding_matrix[i] = w2v_model.wv[word]
    embedding_layer = Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=300, trainable=False)
    #building model 
    model = Sequential()
    model.add(embedding_layer)
    model.add(Dropout(0.5))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='relu'))
    print(model.summary())
    model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
    #train the model
    lstm_model = model.fit(X_train, y_train,batch_size=1024,epochs=10,validation_split=0.1,verbose=1)
    y_pred = lstm_model.predict(X_test)
    print(y_pred)