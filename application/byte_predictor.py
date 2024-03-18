'''
    tensorflow==2.13.1
    pymediainfo==6.1.0
'''

import os
import time
import argparse as argv
from pymediainfo import MediaInfo

class data_extract:
    def __init__(
            self, 
            v_path,
            num_blocks=20,
            header_size=1000,
            batch_size=128,
            version=0.1
        ):
        assert os.path.exists(v_path), f'Given file is not found: {v_path}'

        self.v_path = v_path
        self.num_blocks = num_blocks
        self.header_size = header_size
        self.batch_size = batch_size
        self.version = version

    def linspace(self, start, stop, n):
        if n < 2:
            return stop
        diff = (float(stop) - start)/(n - 1)
        return [diff * i + start  for i in range(n)]

    def byte_counter(self,):
        file_size = os.path.getsize(self.v_path)
        return file_size
    
    def byte_selector(self,):
        file_size = self.byte_counter()
        num_block_available = int((file_size - 1000) / (self.batch_size **2))
        list_ids_raw = self.linspace(0, num_block_available - 1, self.num_blocks)
        return file_size, [int(id) for id in list_ids_raw]
    
    def reshape(self, list_in):
        list_out = []
        for i in range(self.batch_size):
            row_vals = [[vals] for vals in list_in[self.batch_size * i : self.batch_size * (i + 1)]]
            list_out.append(row_vals)
        return list_out
    
    def data_extractor(self):
        file_size, list_ids = self.byte_selector()
        if  file_size < self.header_size + self.batch_size ** 2:
            print('WARNING: video is too small, byte thumbnails will be duplicately selected.')
        file_reader = open(self.v_path, "rb")
        _ = file_reader.read(self.header_size) # Drop the header,
        byte_cube_out = []
        for vid in list_ids:
            file_reader.seek(self.header_size + vid * self.batch_size ** 2)
            byte = list(file_reader.read(self.batch_size ** 2))
            byte_cube_out.append(self.reshape(byte))
        media_info = MediaInfo.parse(self.v_path)
        for track in media_info.tracks:
            if track.track_type == "Video":
                bitrate = track.bit_rate / 1000
        return [byte_cube_out], bitrate

def load_model():
    model_struct_path = os.path.join('model', 'model_struct.json')
    model_weight_path = os.path.join('model', 'model_weight.h5')
    assert os.path.exists(model_struct_path), f'Lost model file: {os.path.abspath(model_struct_path)}'
    assert os.path.exists(model_weight_path), f'Lost model file: {os.path.abspath(model_weight_path)}'

    json_struct = open(model_struct_path).read()
    model = tf.keras.models.model_from_json(json_struct)
    model.load_weights(model_weight_path)

    return model

def load_predict(v_path, model, verbosity):
    data_loader = data_extract(v_path)
    byte_cube, raw_bitrate = data_loader.data_extractor()
    compress_ratio = model.predict(byte_cube, verbose=verbosity)[0][0]
    return raw_bitrate * compress_ratio

if __name__ == '__main__':
    desc_msg = 'This application targets predict the bitrate of a compressed video '\
            'from its original source, with configurations of [libx264, medium preset, '\
            'crf=26]'
    epilog_msg = 'Submitted to ICIP 2024 Grand Challenge on Video complexity.\n'\
                 'Author: Xin Shu, Sigmedia lab, Trinity College Dublin, xins@tcd.ie'\
                 'Supervisor: Anil Kokaram, Sigmedia lab, Trinity College Dublin, anil.kokaram@tcd.ie'
    parser = argv.ArgumentParser(
        prog='byte_predictor.py',
        description=desc_msg,
        epilog=epilog_msg
    )
    parser.add_argument(
        'filename', nargs='+', 
        help='list of video files, or a single one'
    )
    # parser.add_argument(
    #     '-i', '--input', required=True, nargs='+',
    #     help='list of video files, or a single one'
    # )
    parser.add_argument(
        '-v', '--verbose', required=False, default=0,
        help='0: quiet, 1: verbose, default: 0'
    )
    args = parser.parse_args()
    if args.verbose == 0:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    import tensorflow as tf
    assert tf.__version__.startswith('2.13'), f'Wrong version of tensorflow, should be 2.13.*, but got {tf.__version__}'
    model = load_model()
    v_list = args.input if args.filename==None else args.filename
    for v_path in v_list:
        assert os.path.exists(v_path), f'File not found: {v_path}. Exit.'
        assert v_path.endswith('.mp4'), f'Videl File has to be in .mp4 container.'
        start_t = time.time()
        pred_br = load_predict(v_path, model, verbosity=args.verbose)
        end_t = time.time()
        print({'predict_bitrate_kb': pred_br, 'time_used': (end_t - start_t)})

