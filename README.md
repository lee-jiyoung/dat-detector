# DAT-detector: Discriminative Action Tubelet Detector for Weakly-supervised Action Detection

## Jiyoung Lee, Seungryong Kim, Sunok Kim, Kwanghoon Sohn


1. download code
```bash
git clone  https://github.com/lee-jiyoung/dat-detector.git
cd dat-detector/
```

2. install [detectron2](https://github.com/facebookresearch/detectron2)
```bash
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

3. download action recognition networks 
```bash
git clone https://github.com/jeffreyyihuang/two-stream-action-recognition.git videonet/
```

- comment out line 155-158 in `viddeonet/network.py` 
- add comment `return x` in line 154

4. download datasets in `data/` 

[UCFSports](https://www.crcv.ucf.edu/data/UCF_Sports_Action)
[UCF24](https://www.crcv.ucf.edu/data/UCF101.php)
[UCF24-RGB,Flow](https://github.com/gurkirt/realtime-action-detection/tree/master)
[JHMDB](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/#Downloads)
[ImageNet VID](https://drive.google.com/file/d/13iD3maoxiUEqeZmonODm3B03NCGCPJFh/view?usp=drive_link) Originality from https://github.com/sdroh1027/DiffusionVID.

```
mkdir data/
cd data/
```

5. Preprocess the datasets following [STPN](https://github.com/bellos1203/STPN) to extract rgb and flow frames.

```bash
├── data
│   ├──ucf_sports
│   │   ├──Diving-Side
│   │   │   ├──001
│   │   │   │   ├──rgb
│   │   │   │   │   ├──2538-5_70133.jpg
│   │   │   │   │   ├──2538-5_70134.jpg
│   │   │   │   │   ├──...
│   │   │   │   ├──flow
│   │   │   │   │   ├──2538-5_70133.jpg
│   │   │   │   │   ├──2538-5_70134.jpg
│   │   │   │   │   ├──...
│   ├──ucf_24
│   │   ├──...
```

6. Make a datafile list as follows:
```
#number   class_name   mode(train/val/test)    video_name
```

7. train network
```bash
python main.py --stage=0,1,2 --dataset=['ucf_sports', 'ucf24', 'jhmdb', 'imagenet_vod']
```
- stage 0 : extract proposals (preprocessing)
- stage 1 : train ATP network (after training ATP, linking proposals with `linking.py`)
- stage 2 : train DAT network

### References
https://github.com/facebookresearch/detectron2
https://github.com/jeffreyyihuang/two-stream-action-recognition.git