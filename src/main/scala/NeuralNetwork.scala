import java.nio.file.{Files, Paths}

import scala.math.exp

import breeze.linalg.{*, DenseMatrix, DenseVector, argmax, max, sum}
import play.api.libs.json.{JsValue, Json}

object NeuralNetwork {
  private val weightPath = "./src/main/resources/sample_weight.json"

  def main(args: Array[String]): Unit = {
    val (xs, ts) = getData()
    val network = initNetwork()

//    val accuracy = for ((x, t) <- xs zip ts) yield {
//      val y = predict(network, x)
//      val p = argmax(y)
//      if (p == t) 1.0 else 0.0
//    }
//
//    println("Accuracy:" + accuracy.sum / xs.size)

    val batchSize = 100
    val accuracy = for (i <- xs.indices by batchSize) yield {
      // DenseMatrix(DenseVector1, DenseVector2)とすると縦に連結する（1行が1サンプルとなる）が、
      // 必要なのは1列＝1サンプルの行列のため転置する
      val xBatch = DenseMatrix(xs.slice(i, i + batchSize): _*).t

      val tBatch = ts.slice(i, i + batchSize)
      val yBatch = predict(network, xBatch)

      // 列ごとにargmaxを求める
      // そのままだと返り値が行ベクトルになりtoArrayできないので転置して縦ベクトルにする
      val p = argmax(yBatch(::, *)).t
      (p.toArray zip tBatch) map { case (x, y) => if (x == y) 1.0 else 0.0 }
    }

    println("Accuracy:" + accuracy.flatten.sum / xs.size)
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

    def concat(W: DenseMatrix[Double], b: DenseVector[Double]): DenseMatrix[Double] =
      // 切片(b)もWに含めたいので連結する
      DenseMatrix.vertcat(W, b.toDenseMatrix).t

    val jsonStr = String.join(System.getProperty("line.separator"), Files.readAllLines(Paths.get(weightPath)))
    val json = Json.parse(jsonStr)

    val W1 = parseMatrix(json, "W1")
    val b1 = parseVector(json, "b1")
    val W2 = parseMatrix(json, "W2")
    val b2 = parseVector(json, "b2")
    val W3 = parseMatrix(json, "W3")
    val b3 = parseVector(json, "b3")

    Map(
      "W1" -> concat(W1, b1),
      "W2" -> concat(W2, b2),
      "W3" -> concat(W3, b3)
    )
  }

  def predict(network: Map[String, DenseMatrix[Double]], x: DenseVector[Double]): DenseVector[Double] = {
    val W1 = network("W1")
    val W2 = network("W2")
    val W3 = network("W3")

    val z1 = sigmoid(W1 * addIntercept(x))
    val z2 = sigmoid(W2 * addIntercept(z1))
    softmax(W3 * addIntercept(z2))
  }

  def predict(network: Map[String, DenseMatrix[Double]], x: DenseMatrix[Double]): DenseMatrix[Double] = {
    val W1 = network("W1")
    val W2 = network("W2")
    val W3 = network("W3")

    val z1 = sigmoid(W1 * addIntercept(x))
    val z2 = sigmoid(W2 * addIntercept(z1))
    softmax(W3 * addIntercept(z2))
  }

  def forward(network: Map[String, DenseMatrix[Double]], x: DenseVector[Double]): DenseVector[Double] = {
    val W1 = network("W1")
    val W2 = network("W2")
    val W3 = network("W3")

    val z1 = sigmoid(W1 * addIntercept(x))
    val z2 = sigmoid(W2 * addIntercept(z1))
    identityFunction(W3 * addIntercept(z2))
  }

  def addIntercept(xs: DenseVector[Double]): DenseVector[Double] =
    // xsの末尾に切片項(1)を追加する
    DenseVector.vertcat(xs, DenseVector.ones(1))

  def addIntercept(X: DenseMatrix[Double]): DenseMatrix[Double] =
    // Xの末尾（最終行の下）に切片項（値がすべて1の行ベクトル）を追加する
    DenseMatrix.vertcat(X, DenseMatrix.ones[Double](1, X.cols))

  def stepFunction(x: Double): Double =
    if (x > 0) 1.0
    else 0.0

  def stepFunction(xs: DenseVector[Double]): DenseVector[Double] =
    xs.map(stepFunction)

  def stepFunction(X: DenseMatrix[Double]): DenseMatrix[Double] =
    X.map(stepFunction)

  def sigmoid(x: Double): Double =
    1.0 / (1.0 + exp(-x))

  def sigmoid(xs: DenseVector[Double]): DenseVector[Double] =
    xs.map(sigmoid)

  def sigmoid(X: DenseMatrix[Double]): DenseMatrix[Double] =
    X.map(sigmoid)

  def relu(x: Double): Double =
    List(0, x).max

  def relu(xs: DenseVector[Double]): DenseVector[Double] =
    xs.map(relu)

  def relu(X: DenseMatrix[Double]): DenseMatrix[Double] =
    X.map(relu)

  def identityFunction(xs: DenseVector[Double]): DenseVector[Double] =
    xs

  def softmax(as: DenseVector[Double]): DenseVector[Double] = {
    val c = max(as)
    val expAs = (as - c).map(exp)
    val sumExpAs = sum(expAs)
    expAs / sumExpAs
  }

  def softmax(A: DenseMatrix[Double]): DenseMatrix[Double] = {
    // 行列版softmax関数
    // 列ごとにsoftmaxする
    val seq = for (i <- 0 until A.cols) yield softmax(A(::, i))
    // DenseMatrix(DenseVector1, DenseVector2)とすると縦に連結する（1行が1サンプルのsoftmaxの結果）が、
    // 必要なのは1列＝1サンプルの行列のため転置する
    DenseMatrix(seq: _*).t
  }
}
