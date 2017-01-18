package com.github.who_you_me.deeplearning

import scala.collection.mutable
import breeze.linalg.{*, DenseMatrix, argmax}
import breeze.stats.distributions.Gaussian
import com.github.who_you_me.deeplearning.common.Gradient
import com.github.who_you_me.deeplearning.layer.{Affine, MidLayer, Relu, SoftmaxWithLoss}

object TwoLayerNet extends App {
  val g = Gaussian(0, 1)
  val net = new TwoLayerNet(784, 50, 10)
  val x = DenseMatrix(g.sample(784 * 50): _*).reshape(50, 784)
  val t = DenseMatrix(g.sample(50 * 10): _*).reshape(50, 10)

  val gradW1, gradW2 = net.gradient(x, t)
  println(gradW1)
  println(gradW2)
}

class TwoLayerNet(inputSize: Int, hiddenSize: Int, outputSize: Int, weightInitStd: Double = 0.01) {
  private val g = Gaussian(0, weightInitStd)
  val W1 = DenseMatrix(g.sample(inputSize * hiddenSize): _*).reshape(inputSize, hiddenSize)
  val b1 = DenseMatrix.zeros[Double](1, hiddenSize)
  val W2 = DenseMatrix(g.sample(hiddenSize * outputSize): _*).reshape(hiddenSize, outputSize)
  val b2 = DenseMatrix.zeros[Double](1, outputSize)

  val layers = mutable.LinkedHashMap.empty[String, MidLayer]
  layers += ("Affine1" -> new Affine(W1, b1))
  layers += ("Relu1" -> new Relu())
  layers += ("Affine2" -> new Affine(W2, b2))

  val lastLayer = new SoftmaxWithLoss()

  def predict(x: DenseMatrix[Double]) = {
    var midX = x
    for (layer <- layers.values) {
      midX = layer.forward(midX)
    }
    midX
  }

  def loss(x: DenseMatrix[Double], t: DenseMatrix[Double]): Double = {
    val y = predict(x)
    lastLayer.forward(y, t)
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

  def gradient(x: DenseMatrix[Double], t: DenseMatrix[Double]) = {
    loss(x, t)

    var dout = lastLayer.backward(1.0)

    val revLayers = layers.values.toList.reverse
    for (layer <- revLayers) { dout = layer.backward(dout) }

    (layers("Affine1").dW.get, layers("Affine2").dW.get)
  }
}
