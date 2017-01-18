package com.github.who_you_me.deeplearning.layer

import breeze.linalg.DenseMatrix

abstract class MidLayer {
  var dW: Option[DenseMatrix[Double]] = None
  var db: Option[DenseMatrix[Double]] = None
  def forward(x: DenseMatrix[Double]): DenseMatrix[Double]
  def backward(dout: DenseMatrix[Double]): DenseMatrix[Double]
}
