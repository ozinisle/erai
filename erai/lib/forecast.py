
import os, errno, traceback, sys
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, GlobalMaxPooling1D, SpatialDropout1D
from keras import optimizers

from keras.callbacks import CSVLogger
from folderPathManipulations import getParentFolder,createFolder

from pathlib import Path

traceback_template = '''Traceback (most recent call last):
  File "%(filename)s", line %(lineno)s, in %(name)s
%(type)s: %(message)s\n''' # Skipping the "actual line" item

def trainLSTMModel(trainingDataFilePath, TIME_STEPS_CONFIG=60, BATCH_SIZE_CONFIG = 512, LEARNING_RATE_CONFIG=0.001, EPOCHS_CONFIG = 100) :

    success = False
    trainingData = None
    min_max_scaler = None
    x_t = None
    y_t = None
    x_test_t = None
    y_test_t = None
    lstm_model = None
    history = None

    # Hyper parameters
    TIME_STEPS=TIME_STEPS_CONFIG
    BATCH_SIZE = BATCH_SIZE_CONFIG
    LEARNING_RATE=LEARNING_RATE_CONFIG
    EPOCHS = EPOCHS_CONFIG

    try:
        plt.style.use('dark_background')

        trainingData = pd.read_csv(trainingDataFilePath)
        min_max_scaler = MinMaxScaler()

        numberOfFeatures = trainingData.shape[1]
        numberOfOutputs = 5

        # check if there are any null/Nan values to worry about
        print("checking if any null values are present\n", trainingData.isna().sum())

        df_train, df_test = train_test_split(trainingData, train_size=0.6, test_size=0.4, shuffle=False)
        
        print("Train and Test size", len(df_train), len(df_test))
        # scale the feature MinMax, build array
        x = df_train.values
        min_max_scaler = MinMaxScaler()
        x_train = min_max_scaler.fit_transform(x)

        x_test = min_max_scaler.transform(df_test)

        x_t, y_t = build_timeseries(x_train, 3)
        x_t = trim_dataset(x_t, BATCH_SIZE)
        y_t = trim_dataset(y_t, BATCH_SIZE)
        x_temp, y_temp = build_timeseries(x_test, 3)
        x_val, x_test_t = np.split(trim_dataset(x_temp, BATCH_SIZE),2)
        y_val, y_test_t = np.split(trim_dataset(y_temp, BATCH_SIZE),2)

        # Creating model
        lstm_model = Sequential()

        #input layer
        lstm_model.add(LSTM(numberOfFeatures, batch_input_shape=(BATCH_SIZE, TIME_STEPS, x_t.shape[2]), dropout=0.0, recurrent_dropout=0.0, stateful=True,     kernel_initializer='random_uniform'))
        
        # hidden layers
        # lstm_model.add(Dropout(0.5))
        lstm_model.add(Dense(numberOfFeatures*8,activation='relu'))
        lstm_model.add(Dense(numberOfFeatures*8,activation='relu'))
        lstm_model.add(Dense(numberOfFeatures*8,activation='relu'))

        # output layers
        lstm_model.add(Dense(numberOfOutputs,activation='sigmoid'))

        optimizer = optimizers.RMSprop(lr=LEARNING_RATE)
        lstm_model.compile(loss='mean_squared_error', optimizer=optimizer,metrics=['accuracy'])

        logFilePath = getParentFolder(trainingDataFilePath)
        logFilePath = logFilePath +"/logs"
        createFolder(logFilePath)
        csv_logger = CSVLogger(logFilePath+'model_training_feedback_log.log', append=True)

        history = lstm_model.fit(x_t, y_t, epochs=EPOCHS, verbose=2, batch_size=BATCH_SIZE,
                     shuffle=False, validation_data=(trim_dataset(x_val, BATCH_SIZE),
                     trim_dataset(y_val, BATCH_SIZE)), callbacks=[csv_logger])

        plt.clf()
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(loss) + 1)
        plt.plot(epochs, loss, 'g', label='Training loss')
        plt.plot(epochs, val_loss, 'y', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        plt.clf()
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        plt.plot(epochs, acc, 'g', label='Training acc')
        plt.plot(epochs, val_acc, 'y', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

        success = True
    except:
        print("Error executing method >>> ")
        # exc_type, exc_obj, exc_tb = sys.exc_info()
        # fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        # print("Unexpected error:", sys.exc_info())
        # print(exc_type, fname, exc_tb.tb_lineno)
        
        # http://docs.python.org/2/library/sys.html#sys.exc_info
        exc_type, exc_value, exc_traceback = sys.exc_info() # most recent (if any) by default
        
        '''
        Reason this _can_ be bad: If an (unhandled) exception happens AFTER this,
        or if we do not delete the labels on (not much) older versions of Py, the
        reference we created can linger.

        traceback.format_exc/print_exc do this very thing, BUT note this creates a
        temp scope within the function.
        '''

        traceback_details = {
                            'filename': exc_traceback.tb_frame.f_code.co_filename,
                            'lineno'  : exc_traceback.tb_lineno,
                            'name'    : exc_traceback.tb_frame.f_code.co_name,
                            'type'    : exc_type.__name__,
                            'message' : traceback.extract_tb(exc_traceback)
                            }
        
        del(exc_type, exc_value, exc_traceback) # So we don't leave our local labels/objects dangling
        # This still isn't "completely safe", though!
        # "Best (recommended) practice: replace all exc_type, exc_value, exc_traceback
        # with sys.exc_info()[0], sys.exc_info()[1], sys.exc_info()[2]
        
        print
        print(traceback.format_exc())
        print
        print(traceback_template % traceback_details)
        print

        #traceback.print_exception()
        raise

    finally:    
        return success, trainingData, min_max_scaler, x_t, y_t, x_test_t, y_test_t, lstm_model, history

def build_timeseries(mat, y_col_index, TIME_STEPS=60):  
    
    # y_col_index is the index of column that would act as output column
    # total number of time-series samples would be len(mat) - TIME_STEPS
    dim_0 = mat.shape[0] - TIME_STEPS
    dim_1 = mat.shape[1]
    x = np.zeros((dim_0, TIME_STEPS, dim_1))
    y = np.zeros((dim_0,5))
    
    for i in range(dim_0):
        x[i] = mat[i:TIME_STEPS+i]
        # fetch first 5 column values of current row as output
        y[i] = mat[TIME_STEPS+i][0:5] #[TIME_STEPS+i, y_col_index]
    print("length of time-series i/o",x.shape,y.shape)
    return x, y

def trim_dataset(mat, batch_size=128):
    """
    trims dataset to a size that's divisible by BATCH_SIZE
    """
    no_of_rows_drop = mat.shape[0]%batch_size
    if(no_of_rows_drop > 0):
        return mat[:-no_of_rows_drop]
    else:
        return mat