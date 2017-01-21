package com.who_you_me.deeplearning.layer

import breeze.linalg.DenseMatrix
import com.who_you_me.deeplearning.common.Functions.{crossEntropyError, softmax}

class SoftmaxWithLoss extends LastLayer {
  override def forward(x: DenseMatrix[Double], t: DenseMatrix[Double]) = {
    this.t = Some(t)
    this.y = Some(softmax(x))
    this.loss = Some(crossEntropyError(y.get, t))
    this.loss.get
  }

  override def backward(dout: Double = 1.0) = {
    assert(this.y.nonEmpty)
    assert(this.t.nonEmpty)
    val y = this.y.get
    val t = this.t.get

    (y - t) / t.rows.toDouble
  }
}
