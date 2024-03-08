import os 
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

from byte_extractor import byte_folder_path, in_folder_path

target_block_num = 20
json_log_path = os.path.join(byte_folder_path, 'log.json')
csv_data_path = os.path.join(in_folder_path.replace('raw', ''), 'data_collection.csv')

def data_select(
        json_log_path,
        csv_data_path,
        target_select_num=10
    ):
    assert os.path.exists(json_log_path), f'Given file not found: {json_log_path}'
    assert os.path.exists(csv_data_path), f'Given file not found: {csv_data_path}'
    df = pd.read_csv(csv_data_path, index_col=False)
    with open(json_log_path, 'r') as json_log_file:
        dict_log = json.load(json_log_file)
        for v_name in tqdm(dict_log['video_list'], ascii=True):
            num_byte_files = dict_log['video_list'][v_name]
            list_ids = np.rint(np.linspace(0, num_byte_files - 1, target_select_num))
            list_ids = [int(id) for id in list_ids]
            assert len(list_ids) == len(set(list_ids)), f'Duplicated byte file IDs: {list_ids}'
            selection_str = ''
            for id in list_ids:
                if id == list_ids[-1]:
                    selection_str += f'{id}'
                else:
                    selection_str += f'{id},'
            df.loc[df['id']==int(v_name), 'byte_selection'] = selection_str
    df.to_csv(csv_data_path, encoding='utf-8', index=False)


def main():
    data_select(
        json_log_path, 
        csv_data_path,
        target_select_num=target_block_num
    )


if __name__ == '__main__':
    main()
