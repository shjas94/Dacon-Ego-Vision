# βπ» Ego-Vision μλμ μΈμ AI κ²½μ§λν Solution (Private 12th)

# Contents

* Overall Solution
* Code Structure
* How To Train & Inference
* Collaborators
# Overall Solution

![pic2](https://user-images.githubusercontent.com/25663769/137305744-efacd02c-23a7-48ae-a222-04b0a4841ff8.png)

* EfficientNetB7κ³Ό Swin-Large λͺ¨λΈ μμλΈ
* AugmentationμΌλ‘λ Mixup, Rotate, RandomBrightnessContrast, CoarseDropout μ¬μ©

# Code Structure

```
.
βββ DATA
βββ README.md
βββ config
βΒ Β  βββ cfg1.yml
βΒ Β  βββ cfg2.yml
βββ dirstructure.txt
βββ ensemble.py
βββ inference.py
βββ models
βββ modules
βΒ Β  βββ dataset.py
βΒ Β  βββ losses.py
βΒ Β  βββ models.py
βΒ Β  βββ optimizers.py
βΒ Β  βββ schedulers.py
βΒ Β  βββ utils.py
βββ submissions
βββ train.py
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
* [shjas94](https://github.com/shjas94)(νμ¬μ­)
* [chanyub](https://github.com/chanyub)(μ μ°¬μ½)