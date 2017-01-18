package com.github.who_you_me.deeplearning.layer

import breeze.linalg.DenseMatrix

class Affine(W: DenseMatrix[Double]) {
  private var x: Option[DenseMatrix[Double]] = None
  private var dW: Option[DenseMatrix[Double]] = None

  def forward(x: DenseMatrix[Double]): DenseMatrix[Double] = {
    this.x = Some(x)
    x * W
  }

  def backward(dout: DenseMatrix[Double]): DenseMatrix[Double] = {
    assert(this.x.nonEmpty)
    val x = this.x.get

    val dx = dout * W.t
    this.dW = Some(x.t * dout)
    dx
  }
}
