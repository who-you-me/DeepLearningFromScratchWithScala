package com.who_you_me.deeplearning

import scala.collection.mutable
import breeze.linalg.{*, DenseMatrix, argmax}
import breeze.stats.distributions.Gaussian
import common.Gradient
import layer.{Affine, MidLayer, Relu, SoftmaxWithLoss}

class TwoLayerNet(inputSize: Int, hiddenSize: Int, outputSize: Int, weightInitStd: Double = 0.01) {
  private val g = Gaussian(0, weightInitStd)
  var W1 = DenseMatrix(g.sample(inputSize * hiddenSize): _*).reshape(inputSize, hiddenSize)
  var b1 = DenseMatrix.zeros[Double](1, hiddenSize)
  var W2 = DenseMatrix(g.sample(hiddenSize * outputSize): _*).reshape(hiddenSize, outputSize)
  var b2 = DenseMatrix.zeros[Double](1, outputSize)

  val layers = mutable.LinkedHashMap[String, MidLayer](
    "Affine1" -> new Affine(W1, b1),
    "Relu1" -> new Relu(),
    "Affine2" -> new Affine(W2, b2)
  )

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
    def lossW(W: DenseMatrix[Double]): Double = loss(x, t)

    Map(
      "W1" -> Gradient.numericalGradient(lossW, W1),
      "b1" -> Gradient.numericalGradient(lossW, b1),
      "W2" -> Gradient.numericalGradient(lossW, W2),
      "b2" -> Gradient.numericalGradient(lossW, b2)
    )
  }

  def gradient(x: DenseMatrix[Double], t: DenseMatrix[Double]) = {
    loss(x, t)

    var dout = lastLayer.backward()

    val revLayers = layers.values.toList.reverse
    for (layer <- revLayers) { dout = layer.backward(dout) }

    Map(
      "W1" -> layers("Affine1").dW.get,
      "b1" -> layers("Affine1").db.get,
      "W2" -> layers("Affine2").dW.get,
      "b2" -> layers("Affine2").db.get
    )
  }
}
