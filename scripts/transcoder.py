import os
import glob
from tqdm import tqdm
import itertools
from concurrent.futures import ThreadPoolExecutor

from byte_extractor import home, in_folder_path

out_folder_path = os.path.join(home, 'icip2024_challenge/inter4k_vid/compressed')
ffmpeg_bin_path = os.path.join(home, 'bin/ffmpeg')


def trancoder_unit(list_var):
    v_in_path, out_folder_path, ffmpeg_bin_path = list_var
    v_name = os.path.basename(v_in_path)
    v_out_path = os.path.join(out_folder_path, v_name)
    if not os.path.isdir(os.path.join(out_folder_path, 'timer')):
        os.mkdir(os.path.join(out_folder_path, 'timer'))
    if os.path.exists(v_out_path):
        return
    timer_txt_path = os.path.join(
        out_folder_path, 'timer', f"{v_name.split('.')[0]}.txt"
    )
    os.system(
        f'v_in_path={v_in_path} v_out_path={v_out_path} '
        f'ffmpeg_bin_path={ffmpeg_bin_path} '
        f'timer_txt_path={timer_txt_path} bash/libx264_enc.sh'
    )
    return v_name

def main():
    v_list = glob.glob(f'{in_folder_path}/*.mp4')
    list_vars = list(itertools.product(
        v_list, [out_folder_path], [ffmpeg_bin_path]
    ))
    with ThreadPoolExecutor(max_workers=50) as v_pool:
        for _info_ in v_pool.map(trancoder_unit, list_vars):
            print(
                f'INFO: Processing {_info_}.'
            )

if __name__ == '__main__':
    main()