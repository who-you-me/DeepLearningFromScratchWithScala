package com.who_you_me.deeplearning

import breeze.linalg.sum
import breeze.numerics.abs
import dataset.Mnist

object GradientCheck extends App {
  val mnist = Mnist.get()
  val indices = List.range(0, 3)
  val xBatch = mnist.trainImg.getFlatten(indices, normalize = true)
  val tBatch = mnist.trainLabel.getOneHot(indices)

  val network = new TwoLayerNet(784, 50, 10)
  val gradNumerical = network.numericalGradient(xBatch, tBatch)
  val gradBackProp = network.gradient(xBatch, tBatch)

  for (key <- gradNumerical.keysIterator) {
    val diff = sum(abs(gradBackProp(key) - gradNumerical(key)))
    println(s"$key:$diff")
  }
}
