import org.scalatest.FunSuite
import org.scalactic.TolerantNumerics
import breeze.linalg.{DenseMatrix, DenseVector}
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
}
