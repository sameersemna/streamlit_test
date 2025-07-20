## KERAS
from tensorflow.keras.utils import set_random_seed
set_random_seed(66)  # Set random seed for reproducibility
from tensorflow.config.threading import set_intra_op_parallelism_threads, set_inter_op_parallelism_threads
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Dropout, BatchNormalization, Flatten
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.metrics import F1Score
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

N_EPOCHS          = 3
BATCH_SIZE        = 64
LR                = 0.01
DATASET_PERC      = 0.9
IMG_SIZE          = 224## 224 minimum
CONV_FILTERS_1, CONV_FILTERS_2, CONV_FILTERS_3, CONV_FILTERS_4, CONV_FILTERS_5 = 16,16,32,64,128
CONV_K_REG        = 0.0001
DENSE_K_REG       = 0.001
FILE_MODEL_WEIGHTS = f"./data/processed/image_epochs_{N_EPOCHS}.h5"
FILE_MODEL_HISTORY = f"./data/processed/image_epochs_{N_EPOCHS}.pkl"

f1_metric = F1Score(average='macro', name='f1_score')

def get_model_img_custom(df):
    ### MODEL SETUP:
    input_layer = Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    # C1
    conv1 = Conv2D(filters=CONV_FILTERS_1, kernel_size=(3, 3), 
                    activation='relu', kernel_regularizer=l2(CONV_K_REG))(input_layer)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(filters=CONV_FILTERS_1, kernel_size=(3, 3), 
                    activation='relu', kernel_regularizer=l2(CONV_K_REG))(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = Dropout(0.25)(pool1)
    # C2
    conv2 = Conv2D(filters=CONV_FILTERS_2, kernel_size=(3, 3), 
                    activation='relu', kernel_regularizer=l2(CONV_K_REG))(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(filters=CONV_FILTERS_2, kernel_size=(3, 3), 
                    activation='relu', kernel_regularizer=l2(CONV_K_REG))(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = Dropout(0.25)(pool2)
    # C3
    conv3 = Conv2D(filters=CONV_FILTERS_3, kernel_size=(3, 3), 
                    activation='relu', kernel_regularizer=l2(CONV_K_REG))(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(filters=CONV_FILTERS_3, kernel_size=(3, 3), 
                    activation='relu', kernel_regularizer=l2(CONV_K_REG))(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = Dropout(0.25)(pool3)
    # C4
    conv4 = Conv2D(filters=CONV_FILTERS_4, kernel_size=(3, 3), 
                    activation='relu', kernel_regularizer=l2(CONV_K_REG))(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(filters=CONV_FILTERS_4, kernel_size=(3, 3), 
                    activation='relu', kernel_regularizer=l2(CONV_K_REG))(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4 = Dropout(0.25)(pool4)
    # # C5
    conv5 = Conv2D(filters=CONV_FILTERS_5, kernel_size=(3, 3), 
                    activation='relu', kernel_regularizer=l2(CONV_K_REG))(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(filters=CONV_FILTERS_5, kernel_size=(3, 3), 
                    activation='relu', kernel_regularizer=l2(CONV_K_REG))(conv5)
    conv5 = BatchNormalization()(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
    pool5 = Dropout(0.25)(pool5)

    # Global Average Pooling
    gap = GlobalAveragePooling2D()(pool5) ## change this to pool4 or pool5 if you want to use deeper layers

    # Dense layers
    # D1
    dense1 = Dense(512, activation='relu', kernel_regularizer=l2(DENSE_K_REG))(gap)
    dense1 = BatchNormalization()(dense1)
    dense1 = Dropout(0.4)(dense1)
    # D2
    dense2 = Dense(256, activation='relu', kernel_regularizer=l2(DENSE_K_REG))(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Dropout(0.3)(dense2)

    # D3
    dense3 = Dense(64, activation='relu', kernel_regularizer=l2(DENSE_K_REG))(dense2)
    dense3 = BatchNormalization()(dense3)
    dense3 = Dropout(0.3)(dense3)

    # Output layer
    output = Dense(len(df['prdtypecode'].unique()), activation='softmax', kernel_regularizer=l2(0.001))(dense3)
    model = Model(inputs=input_layer, outputs=output)

    model.compile(optimizer=AdamW(learning_rate=LR, weight_decay=0.01), loss='categorical_crossentropy',
            metrics=['accuracy', f1_metric])

    return model