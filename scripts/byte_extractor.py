 #!/bin/bash
import os
import glob
import json
import datetime
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

home = os.path.expanduser('~')
in_folder_path = os.path.join(home, '/home/xins/icip2024_challenge/inter4k_vid/raw/')
byte_folder_path = os.path.join(home, '/home/xins/icip2024_challenge/inter4k_vid/byte')
header_size = 1000
batch_size = 128 * 128
version = 0.1

class data_extract:
    def __init__(
            self, 
            v_paths,
            in_path,
            out_path,
            header_size,
            batch_size=128*128,
            version=0.1
        ):
        assert len(v_paths)!=0, f'Given video file list is empty.'
        assert os.path.isdir(out_path), f'Given folder not found: {out_path}'

        self.v_paths = v_paths
        self.in_path = in_path
        self.out_path = out_path
        self.header_size = header_size
        self.batch_size = batch_size
        self.version = version

        self.log_json_path = os.path.join(self.out_path, 'log.json')
        if os.path.exists(self.log_json_path):
            os.remove(self.log_json_path)
        self.write_to_json(if_header=True)

    def byte_counter(self, v_path):
        file_size = os.path.getsize(v_path)
        return file_size
    
    def check_folder(self, folder_path):
        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)
        return os.path.isdir(folder_path)

    def data_extractor(self):
        proc_bar = tqdm(self.v_paths, ascii=True, desc='Processing')
        for v_path in  proc_bar:
            
            assert os.path.exists(v_path), f'Given video file not found: {v_path}'
            v_name = os.path.basename(v_path).split('.')[0]
            sub_folde_path = os.path.join(self.out_path, v_name)
            if not os.path.isdir(sub_folde_path):
                os.mkdir(sub_folde_path)
            proc_bar.set_description(f'Processing: {v_name}')
            v_byte_count = self.byte_counter(v_path)

            _long_msg_ = f'Given file size is incompatiable: {v_path}, file size is {v_byte_count}.'
            assert v_byte_count >= self.header_size + self.batch_size, _long_msg_
            
            file_reader = open(v_path, "rb")
            _ = file_reader.read(self.header_size) # Drop the header,
            byte_file_counter = 0
            while self.header_size + (byte_file_counter + 1) * self.batch_size <= v_byte_count:
                byte = list(file_reader.read(self.batch_size))
                assert len(byte)==128*128, f'Wrong size of read byte: {len(byte)}'
                byte_reshape = np.array(byte).reshape((128, 128))
                byte_file_path = os.path.join(
                    self.out_path, sub_folde_path,
                    f'{v_name}-byte{byte_file_counter:04d}.npy'
                )
                np.save(byte_file_path, byte_reshape)
                byte_file_counter += 1
            self.write_to_json(
                if_header=False,
                v_name=v_name,
                byte_file_count=byte_file_counter
            )

    def write_to_json(self, if_header,  v_name='', byte_file_count=0):
        if if_header:
            with open(self.log_json_path, 'w') as json_file:
                json_content = {
                    '__VERSION__': self.version,
                    'in_folder_path': f'{os.path.dirname(self.v_paths[0])}',
                    'byte_folder_path': f'{os.path.realpath(self.out_path)}',
                    'video_list': {},
                    'datetime': str(datetime.datetime.now()),
                }
                json.dump(json_content, json_file, indent=2)
        else:
            with open(self.log_json_path, 'r') as json_file:
                temp_data = json.load(json_file)
            with open(self.log_json_path, 'w') as json_file:
                temp_data['video_list'][v_name] = byte_file_count
                json.dump(temp_data, json_file, indent=2)

def main():
    v_list = glob.glob(f'{in_folder_path}/*mp4')
    assert os.path.exists(byte_folder_path), f'Given output folder not exists: {byte_folder_path}'
    de = data_extract(
        v_paths=v_list,
        in_path=in_folder_path,
        out_path=byte_folder_path,
        header_size=header_size,
        batch_size=batch_size
    )
    de.data_extractor()

if __name__ == '__main__':
    main()