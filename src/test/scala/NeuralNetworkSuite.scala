import scala.math.pow

import breeze.linalg.{DenseMatrix, DenseVector}
import org.scalatest.FunSuite
import org.scalactic.TolerantNumerics

import NeuralNetwork._

class NeuralNetworkSuite extends FunSuite {
  implicit val doubleEquality = TolerantNumerics.tolerantDoubleEquality(1e-8)

  test("stepFunction(-1.0, 1.0, 2.0) equals to (0.0, 1.0, 1.0)") {
    val xs = DenseMatrix(-1.0, 1.0, 2.0)
    val expected = DenseMatrix(0.0, 1.0, 1.0)
    assert(stepFunction(xs) == expected)
  }

  test("sigmoid(-1.0, 1.0, 2.0) nearly equals to (0.26894142, 0.73105858, 0.88079708)") {
    val actual = sigmoid(DenseMatrix(-1.0, 1.0, 2.0))
    val expected = List(0.26894142, 0.73105858, 0.88079708)
    for ((a, e) <- actual.toArray zip expected) assert(a === e)
  }

  test("relu(-1.0, 1.0, 2.0) equals to (0.0, 1.0, 2.0)") {
    val xs = DenseMatrix(-1.0, 1.0, 2.0)
    val expected = DenseMatrix(0.0, 1.0, 2.0)
    assert(relu(xs) == expected)
  }

  test("forward(init_network, (1.0, 0.5) nearly equals to (0.31682708, 0.69627909)") {
    def initNetwork(): Map[String, DenseMatrix[Double]] = {
      def concat(W: DenseMatrix[Double], b: DenseVector[Double]): DenseMatrix[Double] =
        DenseMatrix.vertcat(b.toDenseMatrix, W)

      val W1 = DenseMatrix((0.1, 0.3, 0.5), (0.2, 0.4, 0.6))
      val b1 = DenseVector(0.1, 0.2, 0.3)
      val W2 = DenseMatrix((0.1, 0.4), (0.2, 0.5), (0.3, 0.6))
      val b2 = DenseVector(0.1, 0.2)
      val W3 = DenseMatrix((0.1, 0.3), (0.2, 0.4))
      val b3 = DenseVector(0.1, 0.2)

      Map(
        "W1" -> concat(W1, b1),
        "W2" -> concat(W2, b2),
        "W3" -> concat(W3, b3)
      )
    }

    val network = initNetwork()
    val x = DenseMatrix(1.0, 0.5).t
    val actual = forward(network, x)
    val expected = List(0.31682708, 0.69627909)
    for ((a, e) <- actual.toArray zip expected) assert(a === e)
  }

  test("softmax(0.3, 2.9, 4.0) nearly equals to (0.01821127, 0.24519181, 0.73659691)") {
    val actual = softmax(DenseMatrix(0.3, 2.9, 4.0).t)
    val expected = List(0.01821127, 0.24519181, 0.73659691)
    for ((a, e) <- actual.toArray zip expected) assert(a === e)
  }

  test("sumSquaredError returns correct value") {
    val t = DenseVector(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    val y1 = DenseVector(0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0)
    assert(sumSquaredError(y1, t) === 0.0975)
    val y2 = DenseVector(0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.5, 0.0)
    assert(sumSquaredError(y2, t) === 0.7225)
  }

  test("crossEntropyError returns correct value") {
    val tRow = DenseVector(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    val t1 = DenseMatrix(tRow, tRow)
    val y = DenseMatrix((0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0),
                        (0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.5, 0.0))
    val expected = (0.51082545709933802 + 2.3025840919940458) / 2.0
    assert(crossEntropyError(y, t1) === expected)
    val t2 = Stream(2, 2)
    assert(crossEntropyError(y, t2) === expected)
  }

  test("numericalDiff returns correct value") {
    def fTmpl1(x0: Double): Double =
      pow(x0, 2) + pow(4.0, 2)
    assert(numericalDiff(fTmpl1, 3.0) === 6.0)

    def fTmpl2(x1: Double): Double =
      pow(3.0, 2) + pow(x1, 2)
    assert(numericalDiff(fTmpl2, 4.0) === 8.0)
  }

  def f2(x: DenseVector[Double]): Double = {
    assert(x.size == 2)
    pow(x(0), 2) + pow(x(1), 2)
  }

  test("numericalGradient returns correct value") {
    val actual1 = numericalGradient(f2, DenseVector(3.0, 4.0))
    val expected1 = List(6.0, 8.0)
    for ((a, e) <- actual1.toArray zip expected1) assert(a === e)

    val actual2 = numericalGradient(f2, DenseVector(0.0, 2.0))
    val expected2 = List(0.0, 4.0)
    for ((a, e) <- actual2.toArray zip expected2) assert(a === e)

    val actual3 = numericalGradient(f2, DenseVector(3.0, 0.0))
    val expected3 = List(6.0, 0.0)
    for ((a, e) <- actual3.toArray zip expected3) assert(a === e)
  }

  test("gradientDescent returns correct value") {
    val actual = gradientDescent(f2, DenseVector(-3.0, 4.0), lr = 0.1)
    val expected = List(0.0, 0.0)
    for ((a, e) <- actual.toArray zip expected) assert(a === e)
  }
}
