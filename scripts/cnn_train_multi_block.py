import os
import json 
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers
from sklearn import utils
from sklearn import metrics
from sklearn.model_selection import train_test_split
os.environ["CUDA_VISIBLE_DEVICES"]= "1"

from data_selector import csv_data_path, target_block_num
from byte_extractor import home, in_folder_path, byte_folder_path, header_size, batch_size


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
            target_bitrate = _df_.loc[i, 'compressed_bitrate']
            byte_ids = _df_.loc[i, 'byte_selection'].split(',')
            assert len(byte_ids)== self.num_blocks, f'Wrong number of byte_ids: {vid}'
            byte_img_tensor = self.img_loader(vid, byte_ids)
            assert len(byte_img_tensor)== self.num_blocks, f'Wrong number of byte_img_tensor: {vid}'
            xtrain.append(byte_img_tensor)
            ytrain.append(target_bitrate)
        for i in tqdm(range(mid_court, len(_df_)), ascii=True, desc='Loading valid_set'):
            vid = _df_.loc[i, 'id']
            target_bitrate = _df_.loc[i, 'compressed_bitrate']
            byte_ids = _df_.loc[i, 'byte_selection'].split(',')
            assert len(byte_ids)== self.num_blocks, f'Wrong number of byte_ids: {vid}'
            byte_img_tensor = self.img_loader(vid, byte_ids)
            assert len(byte_img_tensor)== self.num_blocks, f'Wrong number of byte_img_tensor: {vid}'
            xvalid.append(byte_img_tensor)
            yvalid.append(target_bitrate)
        assert len(xtrain) + len(xvalid) == len(_df_), f'Error while spliting data: {len(xtrain)}, {len(xvalid)}'
        assert len(ytrain) + len(yvalid) == len(_df_), f'Error while spliting data: {len(ytrain)}, {len(yvalid)}'
        
        return (
            xtrain, ytrain, xvalid, yvalid
        )
    
    def img_loader(self, vid, list_byte_ids):
        byte_blocks = []
        for id in list_byte_ids:
            byte_file_path = os.path.join(
                self.byte_folder_path, str(vid), f'{vid}-b{int(id):04d}.npy'
            )
            assert os.path.exists(byte_file_path), f'Byte file not found: {byte_file_path}'
            byte_blocks.append(tf.convert_to_tensor(np.load(byte_file_path)))
        return byte_blocks


class DataTrainer:
    def __init__(
            self,
            data_set,
            out_folder_path,
            out_model_name,
        ) -> None:
        self.xtrain, self.ytrain, self.xvalid, self.yvalid = data_set
        self.out_folder_path = out_folder_path
        self.model_save_path = os.path.join(out_folder_path, out_model_name)
        if not os.path.isdir(self.model_save_path):
            os.mkdir(self.model_save_path)
    
    def get_model_structure(self, input_shape):

        block_cnn_layers = []
        for block in input_shape[0]:
            inputs = keras.Input(shape=input_shape[1:])
            x1 = layers.conv2d()

    def compile_train(self):
        pass

    def save_train_history_to_json(self):
        json_history_path = os.path.join(self.model_save_path, 'history.json')
        with open(json_history_path, 'w') as json_file:
            json.dump(self.history.history, json_file, indent=2)
        print(f"INFO: Saved trainig history to history.json.")


    def save_model_to_disk(self, model):
        model_json = model.to_json()
        with open(f"{self.out_model_name}.json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights(f"{self.out_model_name}.h5")
        print(f"INFO: Saved model to {self.out_model_name}.json and weights to {self.out_model_name}.h5")


def main():

    print('\n\n\nINFO: Initiating programm ...')
    tester = DataLoader(
        byte_folder_path=byte_folder_path,
        csv_data_path=csv_data_path,
        num_blocks=target_block_num,
        batch_size=batch_size,
        train_set_ratio=0.7
    )
    data_set = tester.loader()

if __name__ == '__main__':
    main()
