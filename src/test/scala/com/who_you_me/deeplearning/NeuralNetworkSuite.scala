package com.who_you_me.deeplearning

import breeze.linalg.{DenseMatrix, DenseVector}
import org.scalactic.TolerantNumerics
import org.scalatest.FunSuite
import common.CastImplicits.denseVectorToDenseMatrix
import NeuralNetwork.forward

class NeuralNetworkSuite extends FunSuite {
  implicit val doubleEquality = TolerantNumerics.tolerantDoubleEquality(1e-8)

  test("forward(init_network, (1.0, 0.5) nearly equals to (0.31682708, 0.69627909)") {
    def initNetwork(): Map[String, DenseMatrix[Double]] = {
      def concat(W: DenseMatrix[Double], b: DenseVector[Double]): DenseMatrix[Double] =
        DenseMatrix.vertcat(b.toDenseMatrix, W)

      Map(
        "W1" -> DenseMatrix((0.1, 0.3, 0.5), (0.2, 0.4, 0.6)),
        "b1" -> DenseVector(0.1, 0.2, 0.3),
        "W2" -> DenseMatrix((0.1, 0.4), (0.2, 0.5), (0.3, 0.6)),
        "b2" -> DenseVector(0.1, 0.2),
        "W3" -> DenseMatrix((0.1, 0.3), (0.2, 0.4)),
        "b3" -> DenseVector(0.1, 0.2)
      )
    }

    val network = initNetwork()
    val x = DenseMatrix(1.0, 0.5).t
    val actual = forward(network, x)
    val expected = List(0.31682708, 0.69627909)
    for ((a, e) <- actual.toArray zip expected) assert(a === e)
  }

}
