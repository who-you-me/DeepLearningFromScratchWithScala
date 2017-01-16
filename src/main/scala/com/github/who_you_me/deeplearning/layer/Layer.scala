package com.github.who_you_me.deeplearning.layer

import breeze.linalg.DenseMatrix

abstract class Layer {
  def forward(x: DenseMatrix[Double]): DenseMatrix[Double]
  def backward(dout: DenseMatrix[Double]): DenseMatrix[Double]
}

abstract class BinLayer {
  def forward(x: Double, y: Double): Double
  def backward(dout: Double): (Double, Double)
}
