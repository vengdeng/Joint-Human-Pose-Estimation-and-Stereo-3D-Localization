# Joint-Human-Pose-Estimation-and-Stereo-Localization
This is Wenlong's master thesis at VITA Lab EPFL and the code is build upon openpifpaf

# Install
Python 3 is required and you need to clone this repository and then:

```sh
pip install numpy cython imgaug pycocotools
pip install torch==1.4.0 torchvision==0.5.0
python setup.py install 
cd openpifpaf/correlation_package
python setup.py install
```
Please notice the correlation module requires GPU, so Mac could not run the model successfully.
### Jupyter Example
We give an example in our example.ipynb to illustrate the usage of our model with kitti dataset. **if you want use your own dataset**, you can fintune the hyperparameter in association_pair.py (ie the depth calculation ratio k and confidence threshold score) file to better match your dataset(**no need to retrain**).  
### Stereo Training 
We load pretrained pifpaf weights for the 2d pose detection. Please download pretrained weights from[pifpaf](https://github.com/vita-epfl/openpifpaf).   
Please keep the folder openpifpaf's name unchanged, since the pretrained pifpaf model will assign weight according to the folder name.
```sh
   python3 -m openpifpaf.train  --momentum=0.95   --epochs=20   --lr-decay 10 20   --batch-size=3   --basenet=resnet152block5   --quad=1   --headnets pif paf psf  --square-edge=401   --regression-loss=laplace   --lambdas 30 2 2 50 3 3 50 3 3   --crop-fraction=0.5 --pretrained (the model from pifpaf)
  ```
### Pretrained Psf model
The pretrained model weights for person localization can be found from the google drive. The model with correlation module is [psf_corr](https://drive.google.com/file/d/13Ezq4_abNJyuWVYlqRhERebZ5DEO81Gi/view?usp=sharing). If you want to use The model without correlation module, please replace the all heads_corr in nets.py with head_psf, the pretrained model is [psf_no_corr](https://drive.google.com/file/d/1fPaNyzXiVN9oYA9OWvQi5BlRk_Uw19PX/view?usp=sharing).
### Stereo Inference
```sh
python3 -m openpifpaf.predict --help
  ```
### Stereo Result
| ALP           |  Type  |error < 0.5|error < 1 |error < 2 |
| ------------- | -------| ----------|----------|----------|
| PSF stereo    | Stereo |  47.6%    | 56.9%    | 63.2%    | 
| MonoLoco      |  Mono  |  27.6%    | 47.8%    | 66.2%    | 
| 3DOP          |  Mono  |  41.5%    | 54.5%    | 63.0%    | 
| MonoDepth     |  Mono  |  19.1%    | 33.0%    | 47.5%    | 

| ALE           |  Type  |    Easy   | Moderate |   Hard   |
| ------------- | -------| ----------|----------|----------|
| PSF stereo    | Stereo | 0.50(0.59)|0.59(0.72)|0.73(0.65)| 
| 3DOP          |  Mono  | 0.54(0.72)|0.85(1.13)|1.56(1.65)| 
| MonoLoco      |  Mono  | 0.85(0.88)|0.97(1.23)|1.14(1.49)|
| MonoDepth     |  Mono  | 1.40(1.69)|2.19(2.98)|2.31(3.77)| 


## Citation

The [paper](https://ieeexplore.ieee.org/abstract/document/9197069) appears at ICRA 2020. If you use, compare with, or refer to this work, please cite

```bibtex
@INPROCEEDINGS{9197069,
  author={W. {Deng} and L. {Bertoni} and S. {Kreiss} and A. {Alahi}},
  booktitle={2020 IEEE International Conference on Robotics and Automation (ICRA)}, 
  title={Joint Human Pose Estimation and Stereo 3D Localization}, 
  year={2020},
  volume={},
  number={},
  pages={2324-2330},
  doi={10.1109/ICRA40945.2020.9197069}}
```