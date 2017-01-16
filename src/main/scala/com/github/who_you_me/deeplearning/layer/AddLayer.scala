package com.github.who_you_me.deeplearning.layer

class AddLayer extends BinLayer {
  override def forward(x: Double, y: Double) = x + y
  override def backward(dout: Double) = (dout, dout)
}
