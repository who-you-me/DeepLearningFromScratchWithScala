package com.github.who_you_me.deeplearning.layer

abstract class Layer {
  def forward(x: Double, y: Double): Double
  def backward(dout: Double): (Double, Double)
}
