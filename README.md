SqueezeNet[[1](https://arxiv.org/abs/1602.07360)]の、Kerasでの実装です。たぶん、もっともシンプルなコードなんじゃないかと。

# Usage

## 準備

~~~ bash
$ pip3 install --upgrade tensorflow-gpu keras funcy matplotlib h5py
~~~

## 訓練

~~~ bash
$ python3 train.py
~~~

## 訓練結果の確認

~~~ bash
$ python3 check.py
~~~

![loss](./results/loss.png)

![accuracy](./results/accuracy.png)

私が試した結果だと、cifar10の精度は91.40%になりました。

※weight decayに相当すると思われるkernel_regularizerを追加して再計測中です。

# Notes

* ごめんなさい。Python3とTensorFlowの環境でしか試していません。
* [https://github.com/nutszebra/residual_net](https://github.com/nutszebra/residual_net)と[https://github.com/takedarts/resnetfamily](https://github.com/takedarts/resnetfamily)を参考にして作成しています。
* 検証には、ImageNetではなくCIFAR-10を使用しました。
* [[2](https://arxiv.org/abs/1603.05027)]のバッチ・ノーマライゼーション→ReLU→畳込みの順序を使用しました。
* Kerasに関数型プログラミングのテクニックを適用する方法は、[Kerasと関数型プログラミングを使えば、深層学習（ディープ・ラーニング）は楽ちんですよ](https://tail-island.github.io/programming/2017/10/25/keras-and-fp.html)にまとめました。

# References

* SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size [[1](https://arxiv.org/abs/1602.07360)]
* Identity Mappings in Deep Residual Networks [[2](https://arxiv.org/abs/1603.05027)]
