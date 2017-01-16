package com.github.who_you_me.deeplearning.layer

class AddLayer extends Layer {
  override def forward(x: Double, y: Double) = x + y

  override def backward(dout: Double) = {
    val dx = dout * 1
    val dy = dout * 1
    (dx, dy)
  }
}
