import pandas as pd
from tensorflow.keras.preprocessing import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Embedding, LSTM
from sklearn.model_selection import train_test_split 
import re 

data = pd.read_csv('cloudy_predictions_sentiment.csv')

data = data[['Review', 'Sentiment']]


def preProcessData(text):
    text = text.lower()
    new_text = re.sub('[^a-zA-Z0-9()\s]',' ', text)
    return new_text

data['Review'] = data['Review'].apply(preProcessData)

max_features = 2000

tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(data['Review'].values)  #split up into words


X = tokenizer.texts_to_sequences(data['Review'].values)
X = pad_sequences(X, 28) 

Y = pd.get_dummies(data['Sentiment']).values

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2,random_state = 0)

model = Sequential()
model.add(Embedding(max_features,embed_dim = 64, input_length = X.shape[1]))
model.add(LSTM(32, return_sequences = True))
model.add(Dense(2, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

batch_size = 16

model.fit(X_train, Y_train, epochs = 10, batch_size = batch_size, validation_data = (X_test, Y_test))

model.save('sentiment.h5')