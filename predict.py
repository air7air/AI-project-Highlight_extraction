import os
import math
import numpy as np
import pandas as pd
from trainer import Trainer
from tensorflow.keras.optimizers import Adam
from keras.applications.xception import preprocess_input
from sklearn.metrics import confusion_matrix, classification_report
from trainer import Trainer
from data_loader import DataLoader


import os
import math
import numpy as np
import pandas as pd
from trainer import Trainer
from tensorflow.keras.optimizers import Adam
from keras.applications.xception import preprocess_input
import sys



# dataset_dir = r'dataset14_sl3_vsr2_vw64_vh64_asr22050_mfcc'
# dataset_dir = './data3/dataset1' #바꿈
# ckpt_dir = 'checkpoints'

dataset_dir = sys.argv[1] #바꿈
ckpt_dir = sys.argv[2]


learning_rate = 1e-4
epochs = 2
batch_size = 1 #원래 256
class_weights = (1, 1)
x_includes = ['video', 'audio']

# for basic model
#x_expand = 0
# for sequence model
x_expand = 2    # 앞 2개, 뒤 2개 segment 포함한다는 뜻

from data_loader import DataLoader

data_loader = DataLoader(dataset_dir, x_includes=x_includes, x_expand=x_expand)  # 데이터 로더에서 앞 2개, 뒤 2개 segment를 더 가져온다

data_config = data_loader.get_metadata()['config']
input_shape_dict = data_loader.get_metadata()['data_shape']


#######################모델 생성
from tensorflow.keras.layers import Dense, Dropout, Conv3D, Conv2D, Input, MaxPool3D, MaxPool2D, Flatten, concatenate, \
    Reshape
from tensorflow.keras.layers import TimeDistributed, LSTM
from tensorflow.keras.backend import expand_dims
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import plot_model



def build_sequence_model():  # 3D CNN + LSTM
    video_input_shape = [None] + input_shape_dict['video']
    audio_input_shape = [None] + input_shape_dict['audio']
    weight_decay = 0.005

    # Video 3D Conv layers
    video_input = Input(video_input_shape)
    x = video_input
    x = TimeDistributed(
        Conv3D(8, (3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu', kernel_initializer='he_uniform',
               kernel_regularizer=l2(weight_decay)))(x)
    x = TimeDistributed(MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same'))(x)
    video_output = TimeDistributed(Flatten())(x)

    # Audio 2D Conv layers
    audio_input = Input(audio_input_shape)
    x = expand_dims(audio_input)  # add channel dim
    x = TimeDistributed(
        Conv2D(4, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='he_uniform',
               kernel_regularizer=l2(weight_decay)))(x)
    x = TimeDistributed(MaxPool2D((2, 2), strides=(2, 2), padding='same'))(x)
    audio_output = TimeDistributed(Flatten())(x)

    # LSTM layers
    x = concatenate([video_output, audio_output])
    x = LSTM(16, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l2(weight_decay))(x)

    # Fully-connected layers
    x = Dense(16, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l2(weight_decay))(x)
    #     x = Dropout(0.2)(x)
    fc_output = Dense(1, activation='sigmoid', kernel_initializer='he_uniform', kernel_regularizer=l2(weight_decay))(x)

    model = Model(inputs=[video_input, audio_input], outputs=fc_output)

    return model


model = build_sequence_model()

checkpoint_name = 'ckpt-20201223-012124-0006-0.0705'
model.load_weights(os.path.join(ckpt_dir, checkpoint_name + '.h5'))

trainer = Trainer(model, data_loader, ckpt_dir)
trainer.test_prediction(batch_size)


