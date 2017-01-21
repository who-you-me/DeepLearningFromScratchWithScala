package com.who_you_me.deeplearning

import java.nio.file.{Files, Paths}
import breeze.linalg.{*, DenseMatrix, DenseVector, argmax}
import play.api.libs.json.{JsValue, Json}
import common.CastImplicits.denseVectorToDenseMatrix
import common.Functions.{identityFunction, sigmoid, softmax}
import common.Util.{forwardFunc=>f}
import dataset.Mnist

object NeuralNetwork {
  private val weightPath = "./src/main/resources/sample_weight.json"

  def main(args: Array[String]): Unit = {
    val (xs, ts) = getData()
    val network = initNetwork()

    val accuracy1 = for ((x, t) <- xs zip ts) yield {
      val y = predict(network, x)
      // yは(1 x 10)行列
      val p = argmax(y.toDenseVector)
      if (p == t) 1.0 else 0.0
    }

    println("Accuracy:" + accuracy1.sum / xs.size)

    val batchSize = 100
    val accuracy2 = for (i <- xs.indices by batchSize) yield {
      // DenseMatrixはDenseMatrixのコンストラクタに渡せないのでDenseVectorに変換
      val x = xs.slice(i, i + batchSize).map(_.toDenseVector)
      val xBatch = DenseMatrix(x: _*)
      val tBatch = ts.slice(i, i + batchSize)
      val yBatch = predict(network, xBatch)

      // yBatchは(batchSize x 10)行列
      // 行ごとにargmaxを求める
      val p = argmax(yBatch(*, ::))
      (p.toArray zip tBatch) map { case (a, b) => if (a == b) 1.0 else 0.0 }
    }

    println("Accuracy:" + accuracy2.flatten.sum / xs.size)
  }

  def getData() = {
    val mnist = Mnist.get()
    val xTest = mnist.testImg.getFlatten()
    val tTest = mnist.testLabel.get()
    (xTest, tTest)
  }

  def initNetwork(): Map[String, DenseMatrix[Double]] = {
    def parseMatrix(json: JsValue, key: String): DenseMatrix[Double] = {
      val list = (json \ key).get.as[List[List[Double]]]
      DenseMatrix(list: _*)
    }

    def parseVector(json: JsValue, key: String): DenseVector[Double] = {
      val list = (json \ key).get.as[List[Double]]
      DenseVector(list: _*)
    }

    val jsonStr = String.join(System.getProperty("line.separator"), Files.readAllLines(Paths.get(weightPath)))
    val json = Json.parse(jsonStr)

    Map(
      "W1" -> parseMatrix(json, "W1"),
      "b1" -> parseVector(json, "b1"),
      "W2" -> parseMatrix(json, "W2"),
      "b2" -> parseVector(json, "b2"),
      "W3" -> parseMatrix(json, "W3"),
      "b3" -> parseVector(json, "b3")
    )
  }

  def predict(network: Map[String, DenseMatrix[Double]], x: DenseMatrix[Double]): DenseMatrix[Double] = {
    val (w1, w2, w3) = (network("W1"), network("W2"), network("W3"))
    val (b1, b2, b3) = (network("b1"), network("b2"), network("b3"))

    val z1 = sigmoid(f(x, w1, b1))
    val z2 = sigmoid(f(z1, w2, b2))
    softmax(f(z2, w3, b3))
  }

  def forward(network: Map[String, DenseMatrix[Double]], x: DenseMatrix[Double]): DenseMatrix[Double] = {
    val (w1, w2, w3) = (network("W1"), network("W2"), network("W3"))
    val (b1, b2, b3) = (network("b1"), network("b2"), network("b3"))

    val z1 = sigmoid(f(x, w1, b1))
    val z2 = sigmoid(f(z1, w2, b2))
    identityFunction(f(z2, w3, b3))
  }
}
