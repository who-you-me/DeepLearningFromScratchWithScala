package com.github.who_you_me.deeplearning.common

import breeze.linalg.DenseVector

object Gradient {
  val h = 1e-4

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
