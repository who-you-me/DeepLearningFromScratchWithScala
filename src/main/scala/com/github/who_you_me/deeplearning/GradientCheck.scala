package com.github.who_you_me.deeplearning

import breeze.linalg.{DenseMatrix, sum}
import breeze.numerics.abs
import dataset.Mnist

object GradientCheck extends App {
  val mnist = Mnist.get()
  val xTrain = mnist.trainImg.getFlatten(true)
  val tTrain = mnist.trainLabel.getOneHot()

  val network = new TwoLayerNet(784, 50, 10)

  val x = xTrain.slice(0, 3).map(_.toDenseVector)
  val xBatch = DenseMatrix(x: _*)
  val t = tTrain.slice(0, 3).map(_.toDenseVector)
  val tBatch = DenseMatrix(t: _*)

  val gradNumerical = network.numericalGradient(xBatch, tBatch)
  val gradBackprop = network.gradient(xBatch, tBatch)

  for (key <- gradNumerical.keysIterator) {
    val diff = sum(abs(gradBackprop(key) - gradNumerical(key)))
    println(s"$key:$diff")
  }
}
