SqueezeNet[[1](https://arxiv.org/pdf/1602.07360.pdf)]のKerasでの実装です。

# Usage

## 準備

~~~ bash
$ pip3 install --upgrade tensorflow-gpu keras funcy matplotlib h5py
~~~

## 訓練

~~~ bash
$ python3 -m try_squeeze_net.train
~~~

## 訓練結果の確認

~~~ bash
$ python3 -m try_squeeze_net.check
~~~

# Notes

* 主に[https://github.com/nutszebra/squeeze_net](https://github.com/nutszebra/squeeze_net)を参考にして作成しています。論文だけでは、私の能力だと作れませんでした……。
* 検証には、ImageNetではなくcifar10を使用しました。
* [[2](https://arxiv.org/pdf/1603.05027.pdf)]が提唱しているバッチ・ノーマライゼーション→ReLU→畳込みの順序を採用しました。

# References

* SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size [[1](https://arxiv.org/pdf/1602.07360.pdf)]
* Identity Mappings in Deep Residual Networks [[2](https://arxiv.org/pdf/1603.05027.pdf)]
