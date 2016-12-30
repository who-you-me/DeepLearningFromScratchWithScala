import scala.math.pow
import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics.constants.E

object NeuralNetwork {
  def main(args: Array[String]): Unit = {
    val network = initNetwork()
    val x = DenseMatrix(1.0, 0.5)
    val y = forward(network, x)
    println(y)
  }

  implicit def denseVector2DenseMatrix(xs: DenseVector[Double]): DenseMatrix[Double] =
    xs.toDenseMatrix.t

  def initNetwork(): Map[String, DenseMatrix[Double]] =
    Map(
      "W1" -> DenseMatrix((0.1, 0.2), (0.3, 0.4), (0.5, 0.6)),
      "b1" -> DenseMatrix(0.1, 0.2, 0.3),
      "W2" -> DenseMatrix((0.1, 0.2, 0.3), (0.4, 0.5, 0.6)),
      "b2" -> DenseMatrix(0.1, 0.2),
      "W3" -> DenseMatrix((0.1, 0.2), (0.3, 0.4)),
      "b3" -> DenseMatrix(0.1, 0.2)
    )

  def forward(network: Map[String, DenseMatrix[Double]], x: DenseMatrix[Double]): DenseMatrix[Double] = {
    val W1 = network("W1")
    val W2 = network("W2")
    val W3 = network("W3")
    val (b1, b2, b3) = (network("b1"), network("b2"), network("b3"))

    val z1 = sigmoid(W1 * x + b1)
    val z2 = sigmoid(W2 * z1 + b2)
    identityFunction(W3 * z2 + b3)
  }

  def stepFunction(x: Double): Double =
    if (x > 0) 1.0
    else 0.0

  def stepFunction(X: DenseMatrix[Double]): DenseMatrix[Double] =
    X.map(stepFunction)

  def sigmoid(x: Double): Double =
    1.0 / (1.0 + pow(E, -x))

  def sigmoid(X: DenseMatrix[Double]): DenseMatrix[Double] =
    X.map(sigmoid)

  def relu(x: Double): Double =
    List(0, x).max

  def relu(X: DenseMatrix[Double]): DenseMatrix[Double] =
    X.map(relu)

  def identityFunction(X: DenseMatrix[Double]): DenseMatrix[Double] =
    X
}
