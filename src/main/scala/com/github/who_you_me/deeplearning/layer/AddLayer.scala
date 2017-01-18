package com.github.who_you_me.deeplearning.layer

class AddLayer {
  def forward(x: Double, y: Double): Double = x + y
  def backward(dout: Double): (Double, Double) = (dout, dout)
}
