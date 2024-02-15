
# Online Test-time Adaptation

PyTorch를 기반으로 한 Gradual Online Test-Time Adapatation으로 아래 연구 논문을 기반으로 작성됨

- [Introducing Intermediate Domains for Effective Self-Training during Test-Time](https://arxiv.org/abs/2208.07736)
- [Robust Mean Teacher for Continual and Gradual Test-Time Adaptation](https://arxiv.org/abs/2211.13081) (CVPR2023)
- [Universal Test-time Adaptation through Weight Ensembling, Diversity Weighting, and Prior Correction](https://arxiv.org/abs/2306.00650) (WACV2024)

## Semantic Segmentation

CarlaTTA 기반 실험을 진행하려면 아래 제공된 데이터셋 분할을 다운로드해야됨. 
+ `conf.py`에서 데이터 디렉토리 `_C.DATA_DIR = "./data"`를 변경해야 할 수도 있음
+ 사전 훈련된 소스 체크포인트를 [다운로드](https://drive.google.com/file/d/1PoeW-GnFr374j-J76H8udblSwrae74LQ/view?usp=sharing)하고 `segmentation` 하위 디렉토리에 압축을 풀어야 함

GTTA를 실행하려면 `cfgs` 디렉토리에 제공된 설정 파일을 조건에 맞게 설정하고 아래 명령어를 실행:
```
python test_time.py --cfg cfgs/gtta.yaml
```

`LIST_NAME_TEST`를 설정하여 테스트 시퀀스를 변경 가능:
+ day2night: `day_night_1200.txt`
+ clear2fog: `clear_fog_1200.txt`
+ clear2rain: `clear_rain_1200.txt`
+ dynamic: `dynamic_1200.txt`
+ highway: `town04_dynamic_1200.txt`

highway를 test sequence로 선택한 경우, source list와 해당 체크포인트 경로를 변경
```bash
python test_time.py --cfg cfgs/gtta.yaml LIST_NAME_SRC clear_highway_train.txt LIST_NAME_TEST town04_dynamic_1200.txt CKPT_PATH_SEG ./ckpt/clear_highway/ckpt_seg.pth CKPT_PATH_ADAIN_DEC = ./ckpt/clear_highway/ckpt_adain.pth
```

### CarlaTTA
+ clear [다운로드](https://drive.google.com/file/d/19HUmZkL5wo4gY7w5cfztgNVga_uNSVUp/view?usp=sharing)
+ day2night [다운로드](https://drive.google.com/file/d/1R3br738UCPGryhWhJE-Uy4sCJW3FaVTr/view?usp=sharing)
+ clear2fog  [다운로드](https://drive.google.com/file/d/1LeNF9PpdJ7lbpsvNwGy9xpC-AYlPiwMI/view?usp=sharing)
+ clear2rain [다운로드](https://drive.google.com/file/d/1TJfQ4CjIOJtrOpUCQ7VyqKBVYQndGNa_/view?usp=sharing)
+ dynamic [다운로드](https://drive.google.com/file/d/1jb1qJMhOSJ48XUQ7eRqT7agnDK9OBwox/view?usp=sharing)
+ dynamic-slow [다운로드](https://drive.google.com/file/d/1RTciKaw2LhlQ4ecKMlarSKyOzsDgaurT/view?usp=sharing)
+ clear-highway [다운로드](https://drive.google.com/file/d/1lZlxwBVBSBAguONX9K6gI2NlWqAxECvB/view?usp=sharing)
+ highway [다운로드](https://drive.google.com/file/d/1Q_3iOuDK4t-W3lvsHwRddDqHTE8GEAIj/view?usp=sharing)

### 감사의 말
+ Sementation 모델 : AdaptSegNet [Official](https://github.com/wasidennis/AdaptSegNet)
+ CarlaTTA : Carla [Official](https://github.com/carla-simulator/carla)
+ ASM [Official](https://github.com/RoyalVane/ASM)
+ SM-PPM [Official](https://github.com/W-zx-Y/SM-PPM)
+ MEMO [Official](https://github.com/zhangmarvin/memo)
