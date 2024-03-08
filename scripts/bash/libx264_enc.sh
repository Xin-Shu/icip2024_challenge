#!/bin/bash
<< OVERVIEW
  Variables to invoke this bash script:
    v_in_path
    v_out_path
    timer_txt_path
OVERVIEW

/usr/bin/time --verbose --output=${timer_txt_path} \
ffmpeg \
-y \
-i ${v_in_path} \
-c:v libx264 \
-preset medium \
-crf 26 \
-threads 1 \
${v_out_path} \
-loglevel quiet