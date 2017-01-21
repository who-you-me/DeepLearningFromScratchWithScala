package com.who_you_me.deeplearning.layer

import scala.math.exp
import breeze.linalg.DenseMatrix

class Sigmoid extends MidLayer {
  private var out: Option[DenseMatrix[Double]] = None

  def forward(x: DenseMatrix[Double]): DenseMatrix[Double] = {
    def sigmoid(x: Double) = 1 / (1 + exp(-x))

    this.out = Option(x.map(sigmoid))
    out.get
  }

  def backward(dout: DenseMatrix[Double]): DenseMatrix[Double] = {
    def func(dout: Double, y: Double): Double = dout * (1.0 - y) * y

    assert(this.out.nonEmpty)
    val out = this.out.get

    assert((dout.rows, dout.cols) == (out.rows, out.cols))

    val dx = DenseMatrix.zeros[Double](dout.rows, dout.cols)
    for (i <- 0 until dout.rows; j <- 0 until dout.cols) {
      val v = func(dout(i,j), out(i, j))
      dx.update(i, j, v)
    }
    dx
  }
}
