package com.who_you_me.deeplearning.layer

import breeze.linalg.DenseMatrix

abstract class LastLayer {
  protected var loss: Option[Double] = None
  protected var y: Option[DenseMatrix[Double]] = None
  protected var t: Option[DenseMatrix[Double]] = None
  def forward(x: DenseMatrix[Double], t: DenseMatrix[Double]): Double
  def backward(dout: Double = 1.0): DenseMatrix[Double]
}
