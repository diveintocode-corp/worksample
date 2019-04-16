# Week4授業前課題3 オブジェクト指向に慣れよう（機械学習コースワークサンプル用）

⚠授業前日までに提出してください。

⚠課題を完了させることは、DIVE INTO CODEを卒業するための必須条件です。

⚠課題では、テキストにも含まれていない新しい知識が含まれています。これは自分で調べて新しいことを把握する力を身に着けてほしいからです。

`課題`を一人で遂行できる力を身につけることで、きちんとした現場に通用する能力を身につけることができます。それでは頑張っていきましょう！

------------
## この課題の目的

- クラスを利用したコードを読み書きできるようにする

------------

`以下の要件をすべて満たしていた場合、合格とします。`

`※Jupyter Notebookを使い課題に沿った検証や説明ができている。`

## オブジェクト指向

これまでの課題では触れてきませんでしたが、`StandardScaler`や`LinearRegression`のような **クラス** と呼ばれるものがPythonなどのプログラム言語では利用できます。

クラスの構文は、オブジェクト指向と呼ばれる考え方を利用したプログラミングの基本的な道具になります。

この課題ではこれまでに既に登場していたクラスを例に、クラスを活用することでどのようなことができるのかを見て学んでいきます。そして課題の後半では`StandardScaler`のクラスをスクラッチで自作します。

## scikit-learnの標準化クラス

課題1で利用したscikit-learnに用意されている標準化を行うためのクラス`StandardScaler`を例に見ていきます。サンプルコードを用意しましたので、これを利用しながら理解していきます。

[sklearn.preprocessing.StandardScaler — scikit-learn 0.20.0 documentation](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)

**サンプルコード**

```py
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

data = load_iris()
X = data.data[:10]

scaler = StandardScaler()
scaler.fit(X)
print("平均 :", scaler.mean_)
print("分散 :", scaler.var_)
X_std = scaler.transform(X)
```

### インスタンス化

クラスを使う際はまず以下のようなコードを書きますが、これを **インスタンス化** と呼びます。

`scaler = StandardScaler()`

StandardScalerというクラスオブジェクトから、scalerと名前をつけたインスタンスオブジェクトが作られました。

**クラスの命名法**

Pythonではクラス名は頭文字が大文字、他は小文字という命名法がPEP8により定められています。単語間にアンダースコアは入れません。これを **CapWords** 方式と呼びます。

[はじめに — pep8-ja 1.0 ドキュメント クラスの名前](https://pep8-ja.readthedocs.io/ja/latest/#id31)

こういった形式のものはクラスだと判断することができます。

**インスタンスは複数作れる**

あるクラスオブジェクトからは複数のインスタンスオブジェクトを作成することが可能です。

```py
scaler0 = StandardScaler()
scaler1 = StandardScaler()
scaler2 = StandardScaler()
```


### 【問題1】これまで利用してきたクラスの列挙

クラスを使う際はインスタンス化を行うことと、クラスの命名法がわかりました。この情報を元に、これまでの課題で利用してきたコードの中でどのようなクラスがあったかを答えてください。

最低でもPandas、matplotlib、scikit-learnからそれぞれ1つ以上見つけてください。

### メソッド

インスタンス化を行った後には、`scaler.fit(X)`のような **メソッド** の実行がきます。`StandardScaler`の`fit`メソッドは後でスケーリングに使われる平均と標準偏差を計算する機能があります。

### インスタンス変数（アトリビュート）

`fit`メソッドにより平均と標準偏差が計算されましたが、見た目には変化があるわけではありません。しかし、scalerインスタンスの内部では計算結果が保存されています。こういったインスタンスの中で値を保存するものを **インスタンス変数** や **アトリビュート（属性）** と呼びます。ここで平均が`scaler.mean_`、標準偏差の2乗した値である分散が`scaler.var_`に保存されています。

以下のようにprint文で出力させることができます。

```py
print("平均 :", scaler.mean_) # 平均 : [4.86 3.31 1.45 0.22]
print("分散 :", scaler.var_) # 分散 : [0.0764 0.0849 0.0105 0.0056]
```

**メソッドとインスタンス変数の命名法**

メソッドやインスタンス変数の命名は関数と同様に、全て小文字で行います。単語をつなぐときにはアンダースコアを入れます。

[はじめに — pep8-ja 1.0 ドキュメント メソッド名とインスタンス変数](https://pep8-ja.readthedocs.io/ja/latest/#id37)

### 【問題2】これまで利用してきたメソッドやインスタンス変数の列挙

これまでの課題で利用してきたコードの中でどのようなメソッドやインスタンス変数があったかを答えてください。

最低でもそれぞれ5つ以上答えてください。

**ndarrayやstrもインスタンス**

ドットをつけるというと、NumPyのndarrayに対して`ndarray.shape`や`ndarray.sum()`のような使い方は何度も利用してきたかと思います。これは、ndarrayもインスタンスオブジェクトであり、`shape`はインスタンス変数、`sum`はメソッドだったということです。

Pythonのコードに登場するデータはどれもインスタンスオブジェクトであり、listやstrもメソッドを持ちます。

（例）

[5. データ構造 — Python 3.6.5 ドキュメント 5.1. リスト型についてもう少し](https://docs.python.jp/3/tutorial/datastructures.html#more-on-lists)

```py
l = ['a']
l.append('b') # listのappendメソッド
```

[4. 組み込み型 — Python 3.6.5 ドキュメント 4.7.1. 文字列メソッド](https://docs.python.jp/3/library/stdtypes.html#string-methods)

```py
s = 'Hello, World!'
s.find('W') # strのfindメソッド
```

### インスタンス変数をメソッドが利用

最終的に以下のようにして標準化を行います。

`X_std = scaler.transform(X)`

これは`fit`メソッドで計算したことでインスタンス変数`mean_`や`var_`に保存されていた値を使い、Xを変換したということです。

このようにクラスには複数のメソッドやインスタンス変数が存在し、これらを組み合わせていろいろな機能を実現します。

### 【問題3】標準化クラスをスクラッチで作成

理解をより深めるため、`StandardScaler`をスクラッチで作成しましょう。scikit-learnは使わず、NumPyなどを活用して標準化の計算を記述します。具体的には`fit`メソッドと`transform`メソッドを作ります。

今回は雛形を用意しました。クラスの作成方法は関数に近いです。メソッドはクラスの中にさらにインデントを一段下げて記述します。

インスタンス変数を作成する際は`self.mean_`のように`self`を付けます。クラスの外から`scaler.mean_`と書いていたscalerの部分が自分自身を表すselfになっています。

**雛形**

```py
class ScratchStandardScaler():
    """
    標準化のためのクラス

    Attributes
    ----------
    mean_ : 次の形のndarray, shape(n_features,)
        平均
    var_ : 次の形のndarray, shape(n_features,)
        分散
    """

    def fit(self, X):
        """
        標準化のために平均と標準偏差を計算する。

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            学習データ
        """

        self.mean_ =
        self.var_ =

        pass

    def transform(self, X):
        """
        fitで求めた値を使い標準化を行う。

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            特徴量

        Returns
        ----------
        X_scaled : 次の形のndarray, shape (n_samples, n_features)
            標準化された特緒量
        """
        pass
        return X_scaled
```

以下のコードが実行できるようにしましょう。

```py
import numpy as np
from sklearn.datasets import load_iris

data = load_iris()
X = data.data[:10]

scratch_scaler = ScratchStandardScaler()
scratch_scaler.fit(X)
print("平均 :", scratch_scaler.mean_)
print("分散 :", scratch_scaler.var_)
X_std = scratch_scaler.transform(X)
```

### ライブラリのソースコードを確認

scikit-learnの場合は公式ドキュメントの右上にソースコードへのリンクがあります。

[![Image from Gyazo](https://t.gyazo.com/teams/diveintocode/1b50849db6c38abe423d20fb5de7a8df.png)](https://diveintocode.gyazo.com/1b50849db6c38abe423d20fb5de7a8df)

[\[source\]](https://github.com/scikit-learn/scikit-learn/blob/7389dba/sklearn/preprocessing/data.py#L480)

どのようなコードになっていたかを確認してみましょう。（問題3に取り組んだ後に見ることを推奨します）スクラッチで作成したものよりも全体的にコードが長いのではないかと思います。`inverse_transform`メソッドのように作成しなかったものもありますが、それだけではありません。例えば以下のように、warning文が記述されているなどします。

```py
if not isinstance(y, string_types) or y != 'deprecated':
    warnings.warn("The parameter y on transform() is "
                  "deprecated since 0.19 and will be removed in 0.21",
                  DeprecationWarning)
```

しかし、特に今注目したいのは次の特殊メソッドについてです。

### 特殊メソッド

ソースコードの中に含まれる、まだ説明していない重要な部分が以下です。

このような`__init__`というメソッドは、どのクラスにも共通して置かれる **コンストラクタ** と呼ばれるメソッドです。

```py
def __init__(self, copy=True, with_mean=True, with_std=True):
    self.with_mean = with_mean
    self.with_std = with_std
    self.copy = copy
```

今回のスクラッチでは`copy`、`with_mean`、`with_std`などのパラメータを省略しましたが、このようにインスタンス化の際にパラメータを指定して保存しておくということはよくある使い方です。

コンストラクタの動作を確認するためのサンプルコードを用意しました。コンストラクタは、インスタンス化が行われる時に自動的に実行されるという働きがあります。こういった特殊な動作をするメソッドを、 **特殊メソッド** と呼びます。

**サンプルコード**

```py
class ExampleClass():
    """
    説明用の簡単なクラス

    Parameters
    ----------
    value : float or int
        初期値

    Attributes
    ----------
    value : float or int
        計算結果
    """
    def __init__(self, value):
        self.value = value
        print("初期値{}が設定されました".format(self.value))
    def add(self, value2):
        """
        受け取った引数をself.valueに加える
        """
        self.value += value2

example = ExampleClass(5)
print("value :", example.value)
example.add(3)
print("value :", example.value)
```

### 【課題4】 四則演算を行うクラスの作成

上記ExampleClassは足し算のメソッドを持っていますが、これに引き算、掛け算、割り算のメソッドを加えてください。

また、コンストラクタに入力されたvalueが文字列や配列など数値以外だった場合には警告文を出し、`self.value=0`とするコードを追加してください。

クラス名や説明文も適切に書き換えてください。

---

## Githubでの提出

- 検証や説明を行ったファイルを`week4-work3.ipynb`として`week4`に格納
