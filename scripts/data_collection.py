
import os
import glob
import json
import pandas as pd 
from ffmpeg import probe
from tqdm import tqdm 

from multiprocessing.pool import Pool

from transcoder import out_folder_path
from byte_extractor import in_folder_path, byte_folder_path


def get_v_bitrate(v_path):
    probe_ = probe(v_path)
    video_bitrate = next(s for s in probe_['streams'] if s['codec_type'] == 'video')
    bitrate = video_bitrate['bit_rate']
    return bitrate

def get_num_byte_files(v_name, json_content):
    return json_content['video_list'][v_name]

def get_time_to_transcode(v_name):
    timer_txt_folder = os.path.join(out_folder_path, 'timer')
    timer_txt = os.path.join(timer_txt_folder, f'{v_name}.txt')
    txt_file = open(timer_txt, 'r')
    time = float(txt_file.readlines()[1].strip().split(' ')[-1])
    txt_file.close()
    return time

def data_collection(v_path_list):
    json_path = os.path.join(byte_folder_path, 'log.json')
    json_file = open(json_path, 'r')
    byte_log_json = json.load(json_file)
    out_csv_path = os.path.join(in_folder_path.replace('raw', ''), 'data_collection.csv')
    data_frame = []

    for v_path in tqdm(v_path_list, ascii=True):
        v_name = os.path.basename(v_path).split('.')[0]
        raw_bitrate = get_v_bitrate(v_path)
        compressed_bitrate = get_v_bitrate(os.path.join(
            out_folder_path, f'{v_name}.mp4'
        ))
        num_byte_files = get_num_byte_files(v_name, byte_log_json)
        time_transcode = get_time_to_transcode(v_name)
        data_frame.append([v_name, raw_bitrate, compressed_bitrate, num_byte_files, time_transcode])
    
    bitrate_df = pd.DataFrame(data_frame, columns=[
        'id', 'raw_bitrate', 'compressed_bitrate', 'num_byte_files', 'time_transcode'
    ])
    bitrate_df.to_csv(out_csv_path, encoding='utf-8', index=False)

    json_file.close()


def main():
    v_list = glob.glob(f'{in_folder_path}/*.mp4')
    data_collection(v_list)


if __name__ == '__main__':
    main()
