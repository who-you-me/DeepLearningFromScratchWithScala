import java.nio.file.{Files, Paths}

import breeze.linalg.{*, DenseMatrix, DenseVector, argmax, max, sum}
import breeze.numerics.{exp, log, pow}
import play.api.libs.json.{JsValue, Json}

object NeuralNetwork {
  private val weightPath = "./src/main/resources/sample_weight.json"
  private val h = 1e-4

  def main(args: Array[String]): Unit = {
    val (xs, ts) = getData()
    val network = initNetwork()

//    val accuracy = for ((x, t) <- xs zip ts) yield {
//      val y = predict(network, x)
//      // yは(1 x 10)行列
//      val p = argmax(y.toDenseVector)
//      if (p == t) 1.0 else 0.0
//    }
//
//    println("Accuracy:" + accuracy.sum / xs.size)

    val batchSize = 100
    val accuracy = for (i <- xs.indices by batchSize) yield {
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

    def concat(W: DenseMatrix[Double], b: DenseVector[Double]): DenseMatrix[Double] = {
      // 切片(b)もWに含めたいので連結する
      DenseMatrix.vertcat(b.toDenseMatrix, W)
    }

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

  def predict(network: Map[String, DenseMatrix[Double]], x: DenseMatrix[Double]): DenseMatrix[Double] = {
    val W1 = network("W1")
    val W2 = network("W2")
    val W3 = network("W3")

    val z1 = sigmoid(addIntercept(x) * W1)
    val z2 = sigmoid(addIntercept(z1) * W2)
    softmax(addIntercept(z2) * W3)
  }

  def forward(network: Map[String, DenseMatrix[Double]], x: DenseMatrix[Double]): DenseMatrix[Double] = {
    val W1 = network("W1")
    val W2 = network("W2")
    val W3 = network("W3")

    val z1 = sigmoid(addIntercept(x)* W1)
    val z2 = sigmoid(addIntercept(z1) * W2)
    identityFunction(addIntercept(z2) * W3)
  }

  def addIntercept(x: DenseMatrix[Double]): DenseMatrix[Double] =
    // xの先頭列に切片項（値がすべて1の列ベクトル）を追加する
    DenseMatrix.horzcat(DenseMatrix.ones[Double](x.rows, 1), x)

  def stepFunction(x: Double): Double =
    if (x > 0) 1.0
    else 0.0

  def stepFunction(x: DenseMatrix[Double]): DenseMatrix[Double] =
    x.map(stepFunction)

  def sigmoid(x: Double): Double =
    1.0 / (1.0 + exp(-x))

  def sigmoid(x: DenseMatrix[Double]): DenseMatrix[Double] =
    x.map(sigmoid)

  def relu(x: Double): Double =
    List(0, x).max

  def relu(x: DenseMatrix[Double]): DenseMatrix[Double] =
    x.map(relu)

  def identityFunction(x: DenseMatrix[Double]): DenseMatrix[Double] =
    x

  private def softmax(a: DenseVector[Double]): DenseVector[Double] = {
    val c = max(a)
    val expA = exp(a - c)
    val sumExpA = sum(expA)
    expA / sumExpA
  }

  def softmax(a: DenseMatrix[Double]): DenseMatrix[Double] = {
    // 行列版softmax関数
    // 行ごとにsoftmaxする
    val seq = for (i <- 0 until a.rows) yield softmax(a(i, ::).t)
    DenseMatrix(seq: _*)
  }

  def sumSquaredError(y: DenseVector[Double], t: DenseVector[Double]): Double =
    0.5 * sum(pow(y - t, 2))

  private def crossEntropyError(y: DenseVector[Double], t: DenseVector[Double]): Double = {
    assert(y.size == t.size)
    val delta = 1e-7
    -sum(t :* log(y + delta))
  }

  private def crossEntropyError(y: DenseVector[Double], label: Int): Double = {
    val delta = 1e-7
    -math.log(y(label) + delta)
  }

  def crossEntropyError(y: DenseMatrix[Double], t: DenseMatrix[Double]): Double = {
    assert((y.rows, y.cols) == (t.rows, t.cols))
    val seq = for (i <- 0 until y.rows) yield {
      val yRow = y(i, ::).t
      val tRow = t(i, ::).t
      crossEntropyError(yRow, tRow)
    }
    sum(seq) / y.rows
  }

  def crossEntropyError(y: DenseMatrix[Double], t: Stream[Int]): Double = {
    val seq = for(i <- 0 until y.rows) yield {
      val yRow = y(i, ::).t
      crossEntropyError(yRow, t(i))
    }
    sum(seq) / y.rows
  }

  def numericalDiff(f: Double => Double, x: Double): Double =
    (f(x + h) - f(x - h)) / (2 * h)

  def numericalGradient(f: DenseVector[Double] => Double, x: DenseVector[Double]): DenseVector[Double] = {
    val grad = DenseVector.zeros[Double](x.size)

    for (i <- 0 until x.size) {
      val x1 = x.copy
      x1.update(i, x(i) + h)

      val x2 = x.copy
      x2.update(i, x(i) - h)

      grad.update(i, (f(x1) - f(x2)) / (2 * h))
    }
    grad
  }

  def gradientDescent(f: DenseVector[Double] => Double,
                      initX: DenseVector[Double],
                      lr: Double = 0.01,
                      stepNum: Int = 100) = {
    var x = initX.copy
    for (i <- 0 until stepNum) {
      val grad = numericalGradient(f, x)
      x -= lr * grad
    }
    x
  }
}
