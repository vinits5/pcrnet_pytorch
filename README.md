# Point Cloud Registration Network in PyTorch.

Source Code Author: Vinit Sarode

**[[Paper]](https://arxiv.org/abs/1908.07906)**
**[[Github Link]](https://github.com/vinits5/pcrnet)**

#### This is a pytorch implementation of PCRNet paper.

### Requirements:
1. Cuda 10
2. pytorch==1.3.0
3. transforms3d==0.3.1
4. h5py==2.9.0

### Dataset:
Path for dataset: [Link](https://drive.google.com/drive/folders/19X68JeiXdeZgFp3cuCVpac4aLLw4StHZ?usp=sharing)
1. Download 'train_data' folder from above link for iterative PCRNet.
2. Download 'car_data' folder from above link for PCRNet.

### How to use code:

#### Train Iterative-PCRNet:
python train.py

#### Train PCRNet:
python train.py --iterations 1 --dataset pcr_single

### Citation

```
@InProceedings{vsarode2019pcrnet,
       author = {Sarode, Vinit and Li, Xueqian and Goforth, Hunter and Aoki, Yasuhiro and Arun Srivatsan, Rangaprasad and Lucey, Simon and Choset, Howie},
       title = {PCRNet: Point Cloud Registration Network using PointNet Encoding},
       month = {Aug},
       year = {2019}
}
```

This code builds upon the code provided in Deep Closest Point [DCP](https://github.com/WangYueFt/dcp.git). We thanks the authors of the paper for sharing their code.