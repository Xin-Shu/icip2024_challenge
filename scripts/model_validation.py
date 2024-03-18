import os 
import json
import numpy as np
import pandas as pd 
from tqdm import tqdm
import sklearn.metrics as skmetrics

import tensorflow as tf
from data_selector import csv_data_path, target_block_num
from byte_extractor import home, in_folder_path, byte_folder_path, header_size, batch_size
from cnn_train_multi_block import split_ratio, model_name, model_save_path

model_name = 'compress_ratio'

class TestDataLoader:
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

    def img_loader(self, vid, list_byte_ids):
        byte_blocks = []
        for id in list_byte_ids:
            byte_file_path = os.path.join(
                self.byte_folder_path, str(vid), f'{vid}-b{int(id):04d}.npy'
            )
            assert os.path.exists(byte_file_path), f'Byte file not found: {byte_file_path}'
            byte_blocks.append(np.load(byte_file_path))
        return np.array(byte_blocks)

    def loader(self):
        if self.if_shuffle:
            _df_ = self.df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        else:
            _df_ = self.df
        xtest, ytest_br, ytest_ratio = [], [], []
        mid_court = int(len(_df_) * self.train_set_ratio)
        for i in tqdm(range(mid_court, len(_df_)), ascii=True, desc='Loading valid_set'):
            vid = _df_.loc[i, 'id']
            target_ratio = _df_.loc[i, 'compressed_bitrate'] / _df_.loc[i, 'raw_bitrate']
            target_bitrate = _df_.loc[i, 'compressed_bitrate'] 
            byte_ids = _df_.loc[i, 'byte_selection'].split(',')
            assert len(byte_ids)== self.num_blocks, f'Wrong number of byte_ids: {vid}'
            byte_img_tensor = self.img_loader(vid, byte_ids)
            assert len(byte_img_tensor)== self.num_blocks, f'Wrong number of byte_img_tensor: {vid}'
            xtest.append(byte_img_tensor)
            ytest_br.append(target_bitrate)
            ytest_ratio.append(target_ratio)

        return (
            np.array(xtest), np.array(ytest_br), np.array(ytest_ratio)
        )


def pearson_correlation(y_true, y_pred):
    pass


def load_model(model_folder, model_name):
    model_struct_path = os.path.join(model_folder, model_name, f'{model_name}.json')
    with open(model_struct_path, 'r') as json_file:
        model_struct_json = json.load(json_file)
    
    model_weight_path = os.path.join(model_folder, model_name, f'{model_name}.h5')
    # model = tf.keras.models.model_from_json(model_struct_json)
    # model.load_weights(model_weight_path)

    model_keras = os.path.join(model_folder, model_name, f'{model_name}.keras')
    model = tf.keras.models.load_model(model_keras, compile=True)
    print(model.summary())
    return model

def main():
    model = load_model(model_save_path, model_name)
    # loader = TestDataLoader(
    #     byte_folder_path=byte_folder_path,
    #     csv_data_path=csv_data_path,
    #     num_blocks=target_block_num,
    #     batch_size=batch_size,
    #     train_set_ratio=split_ratio
    # )
    # (xtest, ytest_br, ytest_ratio) = loader.loader()
    
    # ypred_ratio = model.predict(xtest, ytest_ratio)
    # for i in range(len(ypred_ratio)):
    #     print(ytest_ratio[i], ypred_ratio[i])
    


if __name__ == '__main__':
    main()
