package com.who_you_me.deeplearning.layer

abstract class BinOpLayer {
  protected var x: Option[Double] = None
  protected var y: Option[Double] = None
  def forward(x: Double, y: Double): Double
  def backward(dout: Double): (Double, Double)
}
