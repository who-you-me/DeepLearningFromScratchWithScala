package com.github.who_you_me.deeplearning.layer

import breeze.linalg.DenseMatrix
import com.github.who_you_me.deeplearning.common.Functions.{crossEntropyError, softmax}

class SoftmaxWithLoss {
  private var loss: Option[Double] = None
  private var y: Option[DenseMatrix[Double]] = None
  private var t: Option[DenseMatrix[Double]] = None

  def forward(x: DenseMatrix[Double], t: DenseMatrix[Double]): Double = {
    this.t = Some(t)
    this.y = Some(softmax(x))
    this.loss = Some(crossEntropyError(y.get, t))
    this.loss.get
  }

  def backward(dout: Double = 1.0): DenseMatrix[Double] = {
    assert(this.y.nonEmpty)
    assert(this.t.nonEmpty)
    val y = this.y.get
    val t = this.t.get

    val batchSize = t.rows
    (y - t) / batchSize.toDouble
  }
}
