 #　参考サイト：https://qiita.com/knight0503/items/a8bc13a734277e6f79a8

import tensorflow as tf
import tensorflow.examples.tutorials.mnist as mn

mnist = mn.input_data.read_data_sets("MNIST_data", one_hot=True)
"""
###print(mnist)###
Datasets(train=<tensorflow.contrib.learn.python.learn.datasets.mnist.DataSet object at 0x1c22062eb8>,
validation=<tensorflow.contrib.learn.python.learn.datasets.mnist.DataSet object at 0x1c220807b8>,
test=<tensorflow.contrib.learn.python.learn.datasets.mnist.DataSet object at 0x1c22080a90>)
"""

#TensorFlow.Variableについて。公式から。
"""
# Create a variable.
w = tf.Variable(<initial-value>, name=<optional-name>)

# Use the variable in the graph like any Tensor.
y = tf.matmul(w, ...another variable or tensor...)

# The overloaded operators are available too.
z = tf.sigmoid(w + y)

# Assign a new value to the variable with `assign()` or a related method.
w.assign(w + 1.0)
w.assign_add(1.0)
"""

####ここからモデルの定義####
# inputデータの場所確保（形を定義）。mnistのデータは28×28の０１データ。
# 直線にすれば784個のバイナリ
x = tf.placeholder(tf.float32, [None, 784])

# 重みとバイアスの変数を定義
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# 入力の順伝搬方法を定義
# tf.nn: Wrappers for primitive Neural Net (NN) Operations.
y = tf.nn.softmax(tf.matmul(x, W) + b)

# 正解のデータ用の場所確保（形を定義）
y_ = tf.placeholder(tf.float32, [None, 10])

# 損失関数にクロスエントロピーを使う。クロスエントロピーの計算法を定義。
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# 学習係数を指定して勾配降下アルゴリズムを用いてクロスエントロピーを最小化する
# te.trainに置かれている勾配降下最適化マンに先ほど定義したクロスエントロピーの　
# 最小化をお願いする。学習率は0.5
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# 初期化
init = tf.initialize_all_variables()
session = tf.Session()
session.run(init)

####ここから学習####
for i in range(1000):
    # バッチ処理のため、mnistデータから100個のデータをランダムに持ってくる。
    batch_x, batch_y = mnist.train.next_batch(100)
    # feedディクショナリで、inputデータを表すはずの変数に実際のデータを紐づけてあげる。
    session.run(train_step, feed_dict={x: batch_x, y_: batch_y})

####ここからテスト####
#　予測値と正解値を比較してbool値にする
#　argmax(y,1)は予測値の各行で最大となるインデックスをひとつ返す
# つまり一つの画像に対して0~9までの各数字に当てはまる確率が予測されるので、その中で一番
# 大きな予測値と正解の値のインデックスが等しければTrue.そうでなければFalseとなる。
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# 精度を算出するため、boole値を0もしくは1に変換して平均値をとる（つまり、正解数/全データ数）。
# これを正解率にすると定義。
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# 学習済みモデルにテストデータを渡して精度を算出することでモデルを評価する。
print(session.run(accuracy, feed_dict={x: mnist.test.images, y_:mnist.test.labels}))
