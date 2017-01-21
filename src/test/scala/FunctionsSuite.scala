import breeze.linalg.{DenseMatrix, DenseVector}
import org.scalatest.FunSuite
import org.scalactic.TolerantNumerics

import com.github.who_you_me.deeplearning.common.Functions._

class FunctionsSuite extends FunSuite {
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

  test("softmax(0.3, 2.9, 4.0) nearly equals to (0.01821127, 0.24519181, 0.73659691)") {
    val actual = softmax(DenseMatrix(0.3, 2.9, 4.0).t)
    val expected = List(0.01821127, 0.24519181, 0.73659691)
    for ((a, e) <- actual.toArray zip expected) assert(a === e)
  }

  test("sumSquaredError returns correct value") {
    val t = DenseMatrix(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    val y1 = DenseMatrix(0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0)
    assert(sumSquaredError(y1, t) === 0.0975)
    val y2 = DenseMatrix(0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.5, 0.0)
    assert(sumSquaredError(y2, t) === 0.7225)
  }

  test("crossEntropyError returns correct value") {
    val y = DenseMatrix(
      (0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0),
      (0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.5, 0.0)
    )

    val tRow = DenseVector(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    val t1 = DenseMatrix(tRow, tRow)
    val t2 = List(2, 2)
    val expected = (0.51082545709933802 + 2.3025840919940458) / 2.0
    assert(crossEntropyError(y, t1) === expected)
    assert(crossEntropyError(y, t2) === expected)
  }
}
