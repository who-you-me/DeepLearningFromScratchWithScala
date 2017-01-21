package com.who_you_me.deeplearning.layer

class AddLayer extends BinOpLayer {
  override def forward(x: Double, y: Double) = x + y
  override def backward(dout: Double) = (dout, dout)
}
