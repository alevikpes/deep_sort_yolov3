from pathlib import PurePath

from decouple import config


hd = config('HOME')
homedir = PurePath(hd)
projdir = homedir / 'projects'
datadir = projdir / 'my-aerial-data'

_video_in = datadir / 'dn1' / 'dn1-video-2.mp4'


def _get_result_files(video_in):
    _fn_in_stem = video_in.stem
    _fn_in_suf = video_in.suffix
    #_fn_out_name = _fn_in_stem + '_out' + _fn_in_suf
    _fn_out_name = _fn_in_stem + '_out' + '.avi'
    video_out = video_in.with_name(_fn_out_name)
    tf_name = 'tracking_' + video_in.stem + '.txt'
    track_file = video_in.with_name(tf_name)
    return str(video_out), str(track_file)


VIDEO_IN = str(_video_in)
VIDEO_OUT, TRACKING_FILE = _get_result_files(_video_in)
MODEL_FILENAME = str(projdir / 'deep_sort_yolov3' / 'model_data' / 'veri.pb')
CODEC = 'XVID'

# Definition of the parameters
MAX_COSINE_DISTANCE = 0.25
NN_BUDGET = None
NMS_MAX_OVERLAP = 0.1
