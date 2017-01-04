package com.github.who_you_me.deeplearning.common

import breeze.linalg.{DenseMatrix, DenseVector, max, sum}
import breeze.numerics.{exp, log, pow}

object Functions {
  def identityFunction(x: DenseMatrix[Double]): DenseMatrix[Double] =
    x

  private def stepFunction(x: Double): Double =
    if (x > 0) 1.0 else 0.0

  def stepFunction(x: DenseMatrix[Double]): DenseMatrix[Double] =
    x.map(stepFunction)

  private def sigmoid(x: Double): Double =
    1.0 / (1.0 + exp(-x))

  def sigmoid(x: DenseMatrix[Double]): DenseMatrix[Double] =
    x.map(sigmoid)

  private def relu(x: Double): Double =
    List(0, x).max

  def relu(x: DenseMatrix[Double]): DenseMatrix[Double] =
    x.map(relu)

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

  def crossEntropyError(y: DenseMatrix[Double], t: DenseMatrix[Double]): Double = {
    assert((y.rows, y.cols) == (t.rows, t.cols))
    val seq = for (i <- 0 until y.rows) yield {
      val yRow = y(i, ::).t
      val tRow = t(i, ::).t
      crossEntropyError(yRow, tRow)
    }
    sum(seq) / y.rows
  }

  private def crossEntropyError(y: DenseVector[Double], label: Int): Double = {
    val delta = 1e-7
    -math.log(y(label) + delta)
  }

  def crossEntropyError(y: DenseMatrix[Double], t: Stream[Int]): Double = {
    val seq = for(i <- 0 until y.rows) yield {
      val yRow = y(i, ::).t
      crossEntropyError(yRow, t(i))
    }
    sum(seq) / y.rows
  }
}
