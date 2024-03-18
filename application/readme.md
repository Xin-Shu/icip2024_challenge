# ICIP 2024 Grand Challenge on Video Complexity

## Predicting the video bitrate using extracted byte thumbnail

Autor: Xin Shu, Vibhoothi, Prof. Anil Kokaram

## Usage

For Python script usage: 
```
    python bitrate_predictor.py $/path/to/video.mp4$
```
The Docker image can be found using the link below, the tag should be using **final**
```
https://hub.docker.com/r/xinstcd/bitrate_predictor
```
For Docker image usage (with GPU enabled):
```
docker run --volume $/path/to/video/folder/$:/app/test --gpus all xinstcd/bitrate_predictor:final test/$video.mp4/$
```

Both usage should output a Python dictionary with attributes ["predict_bitrate_kb", "time_used"], where the units are **kbyte/s**, and **seconds** respectively. One example of the output:
```
{
    "predict_bitrate_kb": 1500.000,
    "time_used": 1.00
}
```