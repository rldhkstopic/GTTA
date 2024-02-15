# Online Test-time Adaptation
This is an open source online test-time adaptation repository based on PyTorch. It is joint work by Robert A. Marsden and Mario Döbler. It is also the official repository for the following works:
- [Introducing Intermediate Domains for Effective Self-Training during Test-Time](https://arxiv.org/abs/2208.07736)
- [Robust Mean Teacher for Continual and Gradual Test-Time Adaptation](https://arxiv.org/abs/2211.13081) (CVPR2023)
- [Universal Test-time Adaptation through Weight Ensembling, Diversity Weighting, and Prior Correction](https://arxiv.org/abs/2306.00650) (WACV2024)


## Segmentation

For running the experiments based on CarlaTTA, you first have to download the dataset splits as provided below. Again, you probably have to change the data directory `_C.DATA_DIR = "./data"` in `conf.py`. Further, you have to download the pre-trained source checkpoints ([download](https://drive.google.com/file/d/1PoeW-GnFr374j-J76H8udblSwrae74LQ/view?usp=sharing)) and extract the zip-file within the `segmentation` subdirectory.

E.g., to run GTTA, use the config file provided in the directory `cfgs` and run:
```
python test_time.py --cfg cfgs/gtta.yaml
```

You can also change the test sequences by setting `LIST_NAME_TEST` to:
+ day2night: `day_night_1200.txt`
+ clear2fog: `clear_fog_1200.txt`
+ clear2rain: `clear_rain_1200.txt`
+ dynamic: `dynamic_1200.txt`
+ highway: `town04_dynamic_1200.txt`

If you choose highway as the test sequence, you have to change the source list and the corresponding checkpoint paths.
```bash
python test_time.py --cfg cfgs/gtta.yaml LIST_NAME_SRC clear_highway_train.txt LIST_NAME_TEST town04_dynamic_1200.txt CKPT_PATH_SEG ./ckpt/clear_highway/ckpt_seg.pth CKPT_PATH_ADAIN_DEC = ./ckpt/clear_highway/ckpt_adain.pth
```

### CarlaTTA
We provide the different datasets of CarlaTTA as individual zip-files on Google-Drive:
+ clear [download](https://drive.google.com/file/d/19HUmZkL5wo4gY7w5cfztgNVga_uNSVUp/view?usp=sharing)
+ day2night [download](https://drive.google.com/file/d/1R3br738UCPGryhWhJE-Uy4sCJW3FaVTr/view?usp=sharing)
+ clear2fog  [download](https://drive.google.com/file/d/1LeNF9PpdJ7lbpsvNwGy9xpC-AYlPiwMI/view?usp=sharing)
+ clear2rain [download](https://drive.google.com/file/d/1TJfQ4CjIOJtrOpUCQ7VyqKBVYQndGNa_/view?usp=sharing)
+ dynamic [download](https://drive.google.com/file/d/1jb1qJMhOSJ48XUQ7eRqT7agnDK9OBwox/view?usp=sharing)
+ dynamic-slow [download](https://drive.google.com/file/d/1RTciKaw2LhlQ4ecKMlarSKyOzsDgaurT/view?usp=sharing)
+ clear-highway [download](https://drive.google.com/file/d/1lZlxwBVBSBAguONX9K6gI2NlWqAxECvB/view?usp=sharing)
+ highway [download](https://drive.google.com/file/d/1Q_3iOuDK4t-W3lvsHwRddDqHTE8GEAIj/view?usp=sharing)


### Acknowledgements
+ Segmentation model is from AdaptSegNet [official](https://github.com/wasidennis/AdaptSegNet)
+ CarlaTTA was generated using Carla [official](https://github.com/carla-simulator/carla)
+ ASM [official](https://github.com/RoyalVane/ASM)
+ SM-PPM [official](https://github.com/W-zx-Y/SM-PPM)
+ MEMO [official](https://github.com/zhangmarvin/memo)
