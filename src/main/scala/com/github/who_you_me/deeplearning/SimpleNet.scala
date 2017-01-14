package com.github.who_you_me.deeplearning

import breeze.linalg.DenseMatrix
import breeze.stats.distributions.Gaussian
import common.Functions.{crossEntropyError, softmax}
import common.Gradient.numericalGradient

object SimpleNet extends App {
  val net = new SimpleNet()
  val x = DenseMatrix(0.6, 0.9).t
  val t = DenseMatrix(0.0, 0.0, 0.1).t

  def f(W: DenseMatrix[Double]): Double =
    net.loss(x, t)

  val dW = numericalGradient(f(_), net.W)
  println(dW)
}

class SimpleNet {
  private val g = Gaussian(0, 1)
  val W = DenseMatrix(g.sample(6): _*).reshape(2, 3)

  def predict(x: DenseMatrix[Double]): DenseMatrix[Double] =
    x * W

  def loss(x: DenseMatrix[Double], t: DenseMatrix[Double]): Double = {
    val z = predict(x)
    val y = softmax(z)
    crossEntropyError(y, t)
  }
}
