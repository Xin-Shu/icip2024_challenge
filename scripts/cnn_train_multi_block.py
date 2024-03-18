import os
import sys
import json 
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import tensorflow as tf
from keras import layers
from tensorflow import keras
from tensorflow import data as data

from data_selector import csv_data_path, target_block_num
from byte_extractor import home, in_folder_path, byte_folder_path, header_size, batch_size

# os.environ["CUDA_VISIBLE_DEVICES"]= "1"
model_name = 'compress_the_one'
model_save_path = 'model/'
num_epochs = 500
split_ratio = 0.95

class DataLoader:

    def __init__(
            self,
            byte_folder_path,
            csv_data_path,
            num_blocks,
            batch_size,
            train_set_ratio,
            if_shuffle=False,
            random_seed=24,
        ) -> None:
        assert os.path.isdir(byte_folder_path), f'Folder not exists: {byte_folder_path}'
        assert os.path.exists(csv_data_path), f'File not exists: {csv_data_path}'
        self.byte_folder_path = byte_folder_path
        self.batch_size = batch_size
        self.num_blocks = num_blocks
        self.train_set_ratio = train_set_ratio
        self.if_shuffle = if_shuffle
        self.random_seed = random_seed
        self.df = pd.read_csv(csv_data_path, index_col=False)
    
    def loader(self):
        if self.if_shuffle:
            _df_ = self.df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        else:
            _df_ = self.df
        xtrain, ytrain, xvalid, yvalid = [], [], [], []
        mid_court = int(len(_df_) * self.train_set_ratio)
        for i in tqdm(range(0, mid_court), ascii=True, desc='Loading train_set'):
            vid = _df_.loc[i, 'id']
            target_bitrate = _df_.loc[i, 'compressed_bitrate'] / _df_.loc[i, 'raw_bitrate']
            byte_ids = _df_.loc[i, 'byte_selection'].split(',')
            assert len(byte_ids)== self.num_blocks, f'Wrong number of byte_ids: {vid}'
            byte_img_tensor = self.img_loader(vid, byte_ids)
            assert len(byte_img_tensor)== self.num_blocks, f'Wrong number of byte_img_tensor: {vid}'
            xtrain.append(byte_img_tensor)
            ytrain.append(target_bitrate)
        for i in tqdm(range(mid_court, len(_df_)), ascii=True, desc='Loading valid_set'):
            vid = _df_.loc[i, 'id']
            target_bitrate = _df_.loc[i, 'compressed_bitrate'] / _df_.loc[i, 'raw_bitrate']
            byte_ids = _df_.loc[i, 'byte_selection'].split(',')
            assert len(byte_ids)== self.num_blocks, f'Wrong number of byte_ids: {vid}'
            byte_img_tensor = self.img_loader(vid, byte_ids)
            assert len(byte_img_tensor)== self.num_blocks, f'Wrong number of byte_img_tensor: {vid}'
            xvalid.append(byte_img_tensor)
            yvalid.append(target_bitrate)
        assert len(xtrain) + len(xvalid) == len(_df_), f'Error while spliting data: {len(xtrain)}, {len(xvalid)}'
        assert len(ytrain) + len(yvalid) == len(_df_), f'Error while spliting data: {len(ytrain)}, {len(yvalid)}'
        
        return (
            np.array(xtrain), np.array(ytrain), np.array(xvalid), np.array(yvalid)
        )
    
    def img_loader(self, vid, list_byte_ids):
        byte_blocks = []
        for id in list_byte_ids:
            byte_file_path = os.path.join(
                self.byte_folder_path, str(vid), f'{vid}-b{int(id):04d}.npy'
            )
            assert os.path.exists(byte_file_path), f'Byte file not found: {byte_file_path}'
            byte_blocks.append(np.load(byte_file_path))
        return np.array(byte_blocks)


class DataTrainer:
    def __init__(
            self,
            data_set,
            out_folder_path,
            out_model_name,
            batch_size,
            num_blocks,
        ) -> None:
        self.xtrain, self.ytrain, self.xvalid, self.yvalid = data_set
        self.out_folder_path = out_folder_path
        self.batch_size = batch_size
        self.num_blocks = num_blocks
        self.out_model_name = out_model_name
        self.model_save_folder = os.path.join(out_folder_path, out_model_name)
        if not os.path.isdir(self.model_save_folder):
            os.mkdir(self.model_save_folder)
    
    def get_model_structure(self, block_width):

        block_enc_layers = []
        inputs = keras.Input(shape=(self.num_blocks, block_width, block_width, 1))
        for block in range(self.num_blocks):
            input_slice = inputs[:, block, :, :, :]
            
            x = layers.Conv2D(4, (5, 5), padding='same', activation='relu')(input_slice)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D(pool_size=(4, 4))(x)

            x = layers.Conv2D(8, (5, 5), padding='same', activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D(pool_size=(4, 4))(x)

            x = layers.Conv2D(16, (5, 5), padding='same', activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D(pool_size=(4, 4))(x)

            x = layers.Conv2D(32, (5, 5), padding='same', activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x_out = layers.MaxPooling2D(pool_size=(2, 2))(x)
            block_enc_layers.append(x_out)
        y = layers.concatenate(block_enc_layers)
        y = layers.Flatten()(y)

        y = layers.Dense(256, activation='relu')(y)
        y = layers.BatchNormalization()(y)
        y = layers.Dropout(0.2)(y)

        y = layers.Dense(128, activation='relu')(y)
        y = layers.BatchNormalization()(y)
        y = layers.Dropout(0.2)(y)

        outputs = layers.Dense(1)(y)
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model

    def compile_train(self, epochs):
        block_width = int(self.batch_size ** 0.5)
        train_set = data.Dataset.from_tensor_slices((self.xtrain, self.ytrain)).batch(block_width)
        valid_set = data.Dataset.from_tensor_slices((self.xvalid, self.yvalid)).batch(block_width)

        # strategy = tf.distribute.MirroredStrategy()
        # print('INFO: Number of devices: {}'.format(strategy.num_replicas_in_sync))
        # with strategy.scope():
        model = self.get_model_structure(block_width)
        model.summary()
        optimizer = tf.keras.optimizers.Adam()
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.MeanAbsoluteError()
        )
        save_best_model = SaveBestModel()
        history = model.fit(
            train_set, epochs=epochs, validation_data=valid_set, callbacks=[save_best_model]
        )
        model.set_weights(save_best_model.best_weights)
        self.save_train_history_to_json(history)
        self.save_model_to_disk(model)

    def save_train_history_to_json(self, history):
        json_history_path = os.path.join(self.model_save_folder, 'history.json')
        with open(json_history_path, 'w') as json_file:
            json.dump(history.history, json_file, indent=2)
        print(f"INFO: Saved trainig history to history.json.")
    
    def save_model_to_disk(self, model):
        model_json = model.to_json()
        json_model_path = os.path.join(self.model_save_folder, f'{self.out_model_name}.json')
        keras_model_path = os.path.join(self.model_save_folder, f'{self.out_model_name}.keras')
        h5_model_path = os.path.join(self.model_save_folder, f'{self.out_model_name}.h5')

        with open(json_model_path, "w") as json_file:
            json.dump(model_json, json_file, indent=2)
        model.save(keras_model_path)
        model.save_weights(h5_model_path)
        print(f'INFO: Saved model_struct to {json_model_path} and entir_model to {keras_model_path}')


class SaveBestModel(tf.keras.callbacks.Callback):
    def __init__(self, save_best_metric='val_loss', this_max=False):
        self.save_best_metric = save_best_metric
        self.max = this_max
        if this_max:
            self.best = float('-inf')
        else:
            self.best = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        metric_value = logs[self.save_best_metric]
        if self.max:
            if metric_value > self.best:
                self.best = metric_value
                self.best_weights = self.model.get_weights()

        else:
            if metric_value < self.best:
                self.best = metric_value
                self.best_weights= self.model.get_weights()


def main():

    print('\n\n\nINFO: Initiating programm ...')
    loader = DataLoader(
        byte_folder_path=byte_folder_path,
        csv_data_path=csv_data_path,
        num_blocks=target_block_num,
        batch_size=batch_size,
        train_set_ratio=split_ratio,
        if_shuffle=True,
        random_seed=24
    )
    data_set = loader.loader()
    trainer = DataTrainer(
        data_set,
        'model/',
        model_name,
        batch_size,
        target_block_num,
    )
    trainer.compile_train(epochs=num_epochs)

if __name__ == '__main__':
    main()
