import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorboard.plugins.hparams import api as hp
from sklearn.preprocessing import LabelEncoder
import datetime
import numpy as np
import pandas as pd

class RGBStreamBuilder:
    def build_rgb_stream(self, input_shape=(640, 640, 3), num_frames=20):
        model = models.Sequential()
        model.add(layers.Conv3D(64, (3, 3, 3), activation='relu', input_shape=(num_frames,) + input_shape))
        model.add(layers.MaxPooling3D((1, 2, 2)))
        model.add(layers.Conv3D(128, (3, 3, 3), activation='relu'))
        model.add(layers.MaxPooling3D((1, 2, 2)))
        model.add(layers.Conv3D(256, (3, 3, 3), activation='relu'))
        model.add(layers.MaxPooling3D((1, 2, 2)))
        model.add(layers.TimeDistributed(layers.Flatten()))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dropout(0.5))
        return model

class PoseStreamBuilder:
    def build_pose_stream(self, input_shape=(20, 17*2), lstm_activation_fn = 'relu'):
        model = models.Sequential()
        model.add(layers.LSTM(64, input_shape=input_shape, return_sequences=True))
        model.add(layers.LSTM(128, return_sequences=True))
        model.add(layers.LSTM(256, return_sequences=True))
        model.add(layers.Dense(256, activation=lstm_activation_fn))
        return model

class FusionModelBuilder:
    def build_fusion_model(self, rgb_stream, pose_stream, num_classes, activation='softmax'):
        rgb_stream_flattened = layers.Flatten()(rgb_stream.output)
        pose_stream_flattened = layers.Flatten()(pose_stream.output)
        concat__layer = layers.Concatenate()([rgb_stream_flattened, pose_stream_flattened])
        fusion_dense = layers.Dense(512, activation='relu')(concat__layer)
        fusion_output = layers.Dense(num_classes, activation=activation)(fusion_dense)
        fusion_model = models.Model(inputs=[pose_stream.input, rgb_stream.input], outputs=fusion_output)
        return fusion_model

class ActionRecognitionTrainer:
    def __init__(self, buffer_length, activation_fn= 'sigmoid', loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'binary_accuracy']):
        self.rgb_stream_builder = RGBStreamBuilder()
        self.pose_stream_builder = PoseStreamBuilder()
        self.fusion_model_builder = FusionModelBuilder()
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics
        self.activation_fn = activation_fn
        self.buffer_length = buffer_length 
    def train_model(self, num_classes, rgb_input_shape=(640, 640, 3), pose_input_shape=(10, 34)):
        # Build the RGB stream
        print("Building the RGB net...")
        rgb_stream = self.rgb_stream_builder.build_rgb_stream(input_shape=rgb_input_shape, num_frames=10)

        # Build the Pose stream
        print("Building the Pose net...")
        pose_stream = self.pose_stream_builder.build_pose_stream(input_shape=pose_input_shape)

        # Build the Fusion model
        print("Building the fussion net...")
        fusion_model = self.fusion_model_builder.build_fusion_model(rgb_stream, pose_stream, num_classes, activation=self.activation_fn)

        print(fusion_model.summary())

        # Compile the model
        print("Compiling model...")
        fusion_model.compile(loss= self.loss, optimizer= self.optimizer, metrics= self.metrics)
        return fusion_model

    def train_with_hyperparameters(self, model, X, y, epochs):
        history = model.fit(
                X, y, epochs=epochs,
                validation_split = 0.3,
                # batch_size = batch_size
            )
        return history

# # Set the number of classes for your action recognition task
# num_classes = 6

# # Initialize trainer
# trainer = ActionRecognitionTrainer()

# # Train the model
# fusion_model = trainer.train_model(num_classes)


# df = pd.read_csv('keypoint_dataset/KTH/keypoints_df_20240208_200954.csv') 

# df.columns = ['keypoints', 'target']
# df['keypoints'] = df['keypoints'].apply(lambda x: np.asarray(eval(x)) if type(x)==str else np.zeros((10, 34)))
# X = np.asarray(df['keypoints'].tolist())
# y = np.asarray(df['target'])

# label_encoder = LabelEncoder()
# y = label_encoder.fit_transform(df['target'])

# y = to_categorical(y)

# trainer.train_with_hyperparameters(fusion_model, X, y, epochs=100, )