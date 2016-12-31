import scala.math.exp
import breeze.linalg.{DenseMatrix, DenseVector}

object NeuralNetwork {
  def main(args: Array[String]): Unit = {
    val network = initNetwork()
    val x = DenseVector(1.0, 0.5)
    val y = forward(network, x)
    println(y)
  }

  def initNetwork(): Map[String, DenseMatrix[Double]] = {
    def concat(W: DenseMatrix[Double], b: DenseVector[Double]): DenseMatrix[Double] =
      DenseMatrix.vertcat(W, b.toDenseMatrix).t

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

  def forward(network: Map[String, DenseMatrix[Double]], x: DenseVector[Double]): DenseVector[Double] = {
    def addIntercept(xs: DenseVector[Double]): DenseVector[Double] =
      DenseVector.vertcat(xs, DenseVector.ones(1))

    val W1 = network("W1")
    val W2 = network("W2")
    val W3 = network("W3")

    val z1 = sigmoid(W1 * addIntercept(x))
    val z2 = sigmoid(W2 * addIntercept(z1))
    identityFunction(W3 * addIntercept(z2))
  }

  def stepFunction(x: Double): Double =
    if (x > 0) 1.0
    else 0.0

  def stepFunction(xs: DenseVector[Double]): DenseVector[Double] =
    xs.map(stepFunction)

  def sigmoid(x: Double): Double =
    1.0 / (1.0 + exp(-x))

  def sigmoid(xs: DenseVector[Double]): DenseVector[Double] =
    xs.map(sigmoid)

  def relu(x: Double): Double =
    List(0, x).max

  def relu(xs: DenseVector[Double]): DenseVector[Double] =
    xs.map(relu)

  def identityFunction(xs: DenseVector[Double]): DenseVector[Double] =
    xs
}
