package com.who_you_me.deeplearning.layer

import breeze.linalg.DenseMatrix
import com.who_you_me.deeplearning.common.Functions.relu

class Relu extends MidLayer {
  private var mask: Option[DenseMatrix[Boolean]] = None

  def forward(x: DenseMatrix[Double]): DenseMatrix[Double] = {
    mask = Some(x.map(_ <= 0))
    relu(x)
  }

  def backward(dout: DenseMatrix[Double]): DenseMatrix[Double] = {
    assert(this.mask.nonEmpty)
    val mask = this.mask.get

    assert((dout.rows, dout.cols) == (mask.rows, mask.cols))

    val dx = dout.copy
    for (i <- 0 until dx.rows; j <- 0 until dout.cols) {
      if (mask(i, j)) dx.update(i, j, 0.0)
    }
    dx
  }
}
