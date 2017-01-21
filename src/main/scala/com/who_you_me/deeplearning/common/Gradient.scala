package com.who_you_me.deeplearning.common

import breeze.linalg.DenseMatrix

object Gradient {
  val h = 1e-4

  def numericalDiff(f: Double => Double, x: Double): Double =
    (f(x + h) - f(x - h)) / (2 * h)

  def numericalGradient(f: DenseMatrix[Double] => Double, x: DenseMatrix[Double]): DenseMatrix[Double] = {
    val grad = DenseMatrix.zeros[Double](x.rows, x.cols)

    for (i <- 0 until x.rows; j <- 0 until x.cols) {
      val tmp_val = x(i, j)
      x.update(i, j, tmp_val + h)
      val fxh1 = f(x)

      x.update(i, j, tmp_val - h)
      val fxh2 = f(x)

      grad.update(i, j, (fxh1 - fxh2) / (2 * h))
      x.update(i, j, tmp_val)
    }
    grad
  }

  def gradientDescent(f: DenseMatrix[Double] => Double,
                      initX: DenseMatrix[Double],
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
