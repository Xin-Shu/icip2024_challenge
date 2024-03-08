import os 
import numpy as np
import pandas as pd


TARGET_SELECT_NUM = 10

for v_name in dict_log['video_list']:
    num_byte_files = dict_log['video_list'][v_name]
    if target_select_num > 0:
        select_method = 'fixed_number_of_files'
        json_select_content['byte_select_method'] = select_method
        list_ids = np.rint(np.linspace(0, num_byte_files - 1, target_select_num))
        list_ids = [int(ctr) for ctr in list_ids]
        if len(list_ids) != len(set(list_ids)):
            continue
        assert len(list_ids) == len(set(list_ids)), f'Duplicated byte file IDs: {list_ids}'
        for search_pattern in search_patterns:
            if search_pattern in v_name:
                json_select_content['video_byte_selection'][search_pattern][v_name] = list_ids
                continue
    else:
        select_method = 'fixed_ratio_of_original_files'
        json_select_content['byte_select_method'] = select_method
        list_ids = np.rint(np.linspace(0, num_byte_files, target_select_num))
        list_ids = [int(ctr) for ctr in list_ids]
        assert len(list_ids) == len(set(list_ids)), f'Duplicated byte file IDs'
        for search_pattern in search_patterns:
            if search_pattern in v_name:
                json_select_content['video_byte_selection'][search_pattern][v_name] = list_ids