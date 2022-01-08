# ✋🏻 Ego-Vision 손동작 인식 AI 경진대회 Solution (Private 12th)

# Contents

* Overall Solution
* Code Structure
* How To Train & Inference
* Collaborators
# Overall Solution

![pic2](https://user-images.githubusercontent.com/25663769/137305744-efacd02c-23a7-48ae-a222-04b0a4841ff8.png)

* EfficientNetB7과 Swin-Large 모델 앙상블
* Augmentation으로는 Mixup, Rotate, RandomBrightnessContrast, CoarseDropout 사용

# Code Structure

```
.
├── DATA
├── README.md
├── config
│   ├── cfg1.yml
│   └── cfg2.yml
├── dirstructure.txt
├── ensemble.py
├── inference.py
├── models
├── modules
│   ├── dataset.py
│   ├── losses.py
│   ├── models.py
│   ├── optimizers.py
│   ├── schedulers.py
│   └── utils.py
├── submissions
└── train.py
```

# How To Train & Inference

* Model1 & Model2 Train
```
$ python train.py --config cfg1.yml
$ python train.py --config cfg2.yml
```

* Model1 & Model2 Inference 
```
$ python inference.py --config cfg1.yml
$ python inference.py --config cfg2.yml
```


* ensemble
```
$ python ensemble.py
```

# Collaborators
* [shjas94](https://github.com/shjas94)(허재섭)
* [chanyub](https://github.com/chanyub)(신찬엽)