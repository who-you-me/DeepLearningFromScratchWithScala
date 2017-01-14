package com.github.who_you_me.deeplearning

import breeze.linalg.{*, DenseMatrix, argmax}
import breeze.stats.distributions.Gaussian
import com.github.who_you_me.deeplearning.common.Gradient
import common.Functions.{crossEntropyError, sigmoid, softmax}
import common.Util.addIntercept

object TwoLayerNet extends App {
  val g = Gaussian(0, 1)
  val net = new TwoLayerNet(784, 100, 10)
  val x = DenseMatrix(g.sample(784 * 100): _*).reshape(100, 784)
  val t = DenseMatrix(g.sample(100 * 10): _*).reshape(100, 10)

  val gradW1, gradW2 = net.numericalGradient(x, t)
  println(gradW1)
  println(gradW2)
}

class TwoLayerNet(inputSize: Int, hiddenSize: Int, outputSize: Int, weightInitStd: Double = 0.01) {
  private val g = Gaussian(0, weightInitStd)
  val W1 = DenseMatrix.vertcat(
    DenseMatrix.zeros[Double](1, hiddenSize),
    DenseMatrix(g.sample(inputSize * hiddenSize): _*).reshape(inputSize, hiddenSize)
  )
  val W2 = DenseMatrix.vertcat(
    DenseMatrix.zeros[Double](1, outputSize),
    DenseMatrix(g.sample(hiddenSize * outputSize): _*).reshape(hiddenSize, outputSize)
  )

  def predict(x: DenseMatrix[Double]) = {
    val z1 = sigmoid(addIntercept(x) * W1)
    softmax(addIntercept(z1) * W2)
  }

  def loss(x: DenseMatrix[Double], t: DenseMatrix[Double]): Double = {
    crossEntropyError(predict(x), t)
  }

  def accuracy(x: DenseMatrix[Double], t: DenseMatrix[Double]): Double = {
    val y = predict(x)
    val argmaxY = argmax(y(*, ::))
    val argmaxT = argmax(t(*, ::))
    (argmaxY :== argmaxT).activeSize.toDouble / x.rows
  }

  def numericalGradient(x: DenseMatrix[Double], t: DenseMatrix[Double]) = {
    def lossW(W: DenseMatrix[Double]): Double =
      loss(x, t)

    val gradW1 = Gradient.numericalGradient(lossW(_), W1)
    val gradW2 = Gradient.numericalGradient(lossW(_), W2)
    (gradW1, gradW2)
  }
}
