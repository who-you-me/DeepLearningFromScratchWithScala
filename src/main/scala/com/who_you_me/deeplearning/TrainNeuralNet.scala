package com.who_you_me.deeplearning

import scala.util.Random
import dataset.Mnist

object TrainNeuralNet extends App {
  val mnist = Mnist.get()
  val xTrain = mnist.trainImg.getFlatten(true)
  val tTrain = mnist.trainLabel.getOneHot()
  val xTest = mnist.testImg.getFlatten(true)
  val tTest = mnist.testLabel.getOneHot()

  val network = new TwoLayerNet(784, 50, 10)

  val itersNum = 10000
  val trainSize = 60000
  val testSize = 10000
  val batchSize = 100
  val learningRate = 0.1

  var trainLossList = List.empty[Double]
  var trainAccList = List.empty[Double]
  var testAccList = List.empty[Double]

  val iterPerEpoch = List(trainSize / batchSize, 1).max

  for (i <- 0 until itersNum) {
    if (i % 100 == 0) { println(i) }
    val batchMask = Random.shuffle(List.range(0, trainSize)) take batchSize
    val xBatch = mnist.trainImg.getFlatten(batchMask, normalize = true)
    val tBatch = mnist.trainLabel.getOneHot(batchMask)

    val grad = network.gradient(xBatch, tBatch)
    for (key <- List("W1", "b1", "W2", "b2")) {
      network.params(key) -= learningRate * grad(key)
    }

    val loss = network.loss(xBatch, tBatch)
    trainLossList = loss :: trainLossList

    if (i % iterPerEpoch == 0) {
      val trainAcc = network.accuracy(xTrain, tTrain, trainSize)
      val testAcc = network.accuracy(xTest, tTest, testSize)
      trainAccList = trainAcc :: trainAccList
      testAccList = testAcc :: testAccList
      println(trainAcc, testAcc)
    }

    trainLossList = trainLossList.reverse
    trainAccList = trainAccList.reverse
    testAccList = testAccList.reverse
  }
}
