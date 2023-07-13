# LeNet-5

此程序实现了使用lenet5网络训练并预测mnist及cifar10这两个数据集，并且可以生成.onnx文件

额外提供了（mnist）ResNet18的训练程序（trainres.py）以及pytorch官方的resnet的其他网络结构

## Setup

环境配置：
python3.10.0
torch2.0.0+cu118
torchvision0.15.0+cu118

```
$ pip install -r requirements.txt   #若下载速度慢可先下载轮子（https://download.pytorch.org/whl/torch_stable.html）
```

## Usage

Start the training procedure

```
$ python train.py
```

Start the predicting procedure

```
$ python predict.py
```
onnx可使用 https://netron.app/ 在线查看

## References

[[1](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based learning applied to document recognition." Proceedings of the IEEE, 86(11):2278-2324, November 1998.
