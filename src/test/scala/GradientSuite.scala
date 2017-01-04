import scala.math.pow

import breeze.linalg.DenseVector
import org.scalatest.FunSuite
import org.scalactic.TolerantNumerics

import com.github.who_you_me.deeplearning.common.Gradient._

class GradientSuite extends FunSuite {
  implicit val doubleEquality = TolerantNumerics.tolerantDoubleEquality(1e-8)

  test("numericalDiff returns correct value") {
    def f1(x0: Double): Double =
      pow(x0, 2) + pow(4.0, 2)
    assert(numericalDiff(f1, 3.0) === 6.0)

    def f2(x1: Double): Double =
      pow(3.0, 2) + pow(x1, 2)
    assert(numericalDiff(f2, 4.0) === 8.0)
  }

  {
    def f(x: DenseVector[Double]): Double =
      pow(x(0), 2) + pow(x(1), 2)

    test("numericalGradient returns correct value") {
      val actual1 = numericalGradient(f, DenseVector(3.0, 4.0))
      val expected1 = List(6.0, 8.0)
      for ((a, e) <- actual1.toArray zip expected1) assert(a === e)

      val actual2 = numericalGradient(f, DenseVector(0.0, 2.0))
      val expected2 = List(0.0, 4.0)
      for ((a, e) <- actual2.toArray zip expected2) assert(a === e)

      val actual3 = numericalGradient(f, DenseVector(3.0, 0.0))
      val expected3 = List(6.0, 0.0)
      for ((a, e) <- actual3.toArray zip expected3) assert(a === e)
    }

    test("gradientDescent returns correct value") {
      val actual = gradientDescent(f, DenseVector(-3.0, 4.0), lr = 0.1)
      val expected = List(0.0, 0.0)
      for ((a, e) <- actual.toArray zip expected) assert(a === e)
    }
  }
}
