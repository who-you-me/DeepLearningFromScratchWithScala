package com.who_you_me.deeplearning

import breeze.linalg.{DenseMatrix, sum}
import breeze.numerics.abs
import dataset.Mnist

object GradientCheck extends App {
  val mnist = Mnist.get()
  val indices = List.range(0, 3)
  val xBatch = mnist.trainImg.getFlatten(List(0, 1, 2), true)
  val tBatch = mnist.trainLabel.getOneHot(List(0, 1, 2))

  val network = new TwoLayerNet(784, 50, 10)
  val gradNumerical = network.numericalGradient(xBatch, tBatch)
  val gradBackProp = network.gradient(xBatch, tBatch)

  for (key <- gradNumerical.keysIterator) {
    val diff = sum(abs(gradBackProp(key) - gradNumerical(key)))
    println(s"$key:$diff")
  }
}
