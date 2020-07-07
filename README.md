This is a customized tracker implementation of paper [Orientation- and Scale-Invariant Multi-Vehicle Detection and Tracking from Unmanned Aerial Videos](http://jiewang.name/publications/rsmdpi2019). The tracker is modified based on [cosine_metric_learning](https://github.com/nwojke/cosine_metric_learning/tree/feature/veri_dataset) and [deep_sort_yolov3](https://github.com/Qidian213/deep_sort_yolov3).  

For multi-vehicle detection, refer to [UAV-Vehicle-Detection-Dataset](https://github.com/jwangjie/UAV-Vehicle-Detection-Dataset).

#### Key features added compared to [deep_sort_yolov3](https://github.com/Qidian213/deep_sort_yolov3):
1. The `FPS` of the output tracking videos is exactly the same as the input videos. 
2. A `.txt` file ([tracking_DJI_0006.txt](https://github.com/jwangjie/deep_sort_yolov3/blob/master/tracking_DJI_0006.txt)) contains the tracked vehicle locations in the pixel frame is generated.
 

### 1. Train the deep association metric model 
1. Clone [cosine_metric_learning](https://github.com/nwojke/cosine_metric_learning/tree/feature/veri_dataset) `feature/veri_dataset` branch
```
git clone --branch feature/veri_dataset https://github.com/nwojke/cosine_metric_learning.git
```
2. Obtain the [VeRi dataset](https://github.com/VehicleReId/VeRi)
3. Training on Veri dataset
```
python3 train_veri.py --dataset_dir=./VeRi_with_plate --loss_mode=cosine-softmax --log_dir=./output/veri/ --run_id=cosine-softmax
```
4. Monitoring training process
```
tensorboard --logdir ./output/veri/sine-softmax --port 6006
```
5. Obtain the trained weights
```
python3 train_veri.py --mode=freeze --restore_path=output/veri/cosine-softmax/model.ckpt-334551
```
This will create a `veri.pb` file which can be supplied to Deep SORT. Again, the Market1501 script contains a similar function.

### 2. Track multiple vehicles using deep sort package 

1. Download `yolov3_dji_final.weights` from [UAV-Vehicle-Detection-Dataset](https://github.com/jwangjie/UAV-Vehicle-Detection-Dataset). Convert the yolo weight to a keras model by 
```
python3 convert.py yolov3_dji.cfg yolov3_dji_final.weights model_data/yolo_dji.h5
``` 
OR, download my converted one at [yolo_dji.h5](https://drive.google.com/file/d/1XLODUhnXJIEYXgQckw_fQ6-E5fayoNBq/view?usp=sharing)

2. Clone this repository 
```
git clone https://github.com/jwangjie/deep_sort_yolov3
```

3. Run the multi-vehicle tracking 
```
python tracker.py
```

