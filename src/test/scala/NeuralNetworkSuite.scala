import org.scalatest.FunSuite
import org.scalactic.TolerantNumerics
import breeze.linalg.DenseVector
import NeuralNetwork._

class NeuralNetworkSuite extends FunSuite {
  implicit val doubleEquality = TolerantNumerics.tolerantDoubleEquality(1e-8)

  test("stepFunction(-1.0, 1.0, 2.0) equals to (0.0, 1.0, 1.0)") {
    val xs = DenseVector(-1.0, 1.0, 2.0)
    val expected = DenseVector(0.0, 1.0, 1.0)
    assert(stepFunction(xs) == expected)
  }

  test("sigmoid(-1.0, 1.0, 2.0) nearly equals to (0.26894142, 0.73105858, 0.88079708)") {
    val actual = sigmoid(DenseVector(-1.0, 1.0, 2.0))
    val expected = List(0.26894142, 0.73105858, 0.88079708)
    for ((a, e) <- actual.toArray zip expected) assert(a === e)
  }

  test("relu(-1.0, 1.0, 2.0) equals to (0.0, 1.0, 2.0)") {
    val xs = DenseVector(-1.0, 1.0, 2.0)
    val expected = DenseVector(0.0, 1.0, 2.0)
    assert(relu(xs) == expected)
  }

  test("forward(init_network, (1.0, 0.5) nearly equals to (0.31682708, 0.69627909)") {
    val network = initNetwork()
    val xs = DenseVector(1.0, 0.5)
    val actual = forward(network, xs)
    val expected = List(0.31682708, 0.69627909)
    for ((a, e) <- actual.toArray zip expected) assert(a === e)
  }
}
