package com.github.who_you_me.deeplearning

import breeze.linalg.DenseMatrix

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
  val batchSize = 100
  val learningRate = 0.1


  var trainLossList = List.empty[Double]
  var trainAccList = List.empty[Double]
  var testAccList = List.empty[Double]

  val iterPerEpoch = List(trainSize / batchSize, 1).max

  for (i <- 0 until itersNum) {
    val batchMask = Random.shuffle(0 to trainSize - 1) take batchSize
    println(batchMask)
    val xBatchList = (for (i <- batchMask) yield xTrain(i)).toList
    val xBatch = DenseMatrix(xBatchList.map(_.toDenseVector): _*)
    val tBatchList = (for (i <- batchMask) yield tTrain(i)).toList
    val tBatch = DenseMatrix(tBatchList.map(_.toDenseVector): _*)

    val grad = network.gradient(xBatch, tBatch)
    network.W1 = network.W1 - learningRate * grad("W1")
    network.b1 = network.b1 - learningRate * grad("b1")
    network.W2 = network.W2 - learningRate * grad("W2")
    network.b2 = network.b2 - learningRate * grad("b2")

    val loss = network.loss(xBatch, tBatch)
    trainLossList = loss :: trainLossList
  }
  println(network.W2, network.b2)
}