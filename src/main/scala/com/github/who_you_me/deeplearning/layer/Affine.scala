package com.github.who_you_me.deeplearning.layer

import breeze.linalg.{*, DenseMatrix, sum}

class Affine(W: DenseMatrix[Double], b: DenseMatrix[Double]) extends MidLayer {
  private var x: Option[DenseMatrix[Double]] = None

  def forward(x: DenseMatrix[Double]): DenseMatrix[Double] = {
    this.x = Some(x)
    val dot = x * W
    dot(*, ::) + b.toDenseVector
  }

  def backward(dout: DenseMatrix[Double]): DenseMatrix[Double] = {
    assert(this.x.nonEmpty)
    val x = this.x.get

    val dx = dout * W.t
    this.dW = Some(x.t * dout)
    this.db = Some(sum(dout(::, *)).t.toDenseMatrix)
    dx
  }
}
