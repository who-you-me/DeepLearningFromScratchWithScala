package com.who_you_me.deeplearning.common

import breeze.linalg.{*, DenseMatrix, DenseVector, max, sum}
import breeze.numerics.{exp, log, pow}

object Functions {
  private val delta = 1e-7

  def identityFunction[T](x: T): T =
    x

  def stepFunction(x: Double): Double =
    if (x > 0) 1 else 0

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

  def softmax(x: DenseVector[Double]): DenseVector[Double] = {
    val c = max(x)
    val expX = exp(x - c)
    expX / sum(expX)
  }

  def softmax(x: DenseMatrix[Double]): DenseMatrix[Double] = {
    // row-wise softmax function
    x(*, ::).map(softmax)
  }

  def sumSquaredError(y: DenseMatrix[Double], t: DenseMatrix[Double]): Double = {
    assert((y.rows, y.cols) == (t.rows, t.cols))
    0.5 * sum(pow(y - t, 2))
  }

  def crossEntropyError(y: DenseMatrix[Double], t: DenseMatrix[Double]): Double = {
    assert((y.rows, y.cols) == (t.rows, t.cols))
    -1.0 * sum(t :* log(y + delta)) / y.rows
  }

  def crossEntropyError(y: DenseMatrix[Double], t: List[Int]): Double = {
    assert(y.rows == t.size)
    assert(y.cols >= t.max)
    val seq = for (i <- 0 until y.rows) yield {
      val label = t(i)
      -1.0 * log(y(i, label) + delta)
    }
    sum(seq) / y.rows
  }
}
