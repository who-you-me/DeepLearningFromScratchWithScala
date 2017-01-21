package com.who_you_me.deeplearning.layer

class MulLayer extends BinOpLayer {
  override def forward(x: Double, y: Double) = {
    this.x = Some(x)
    this.y = Some(y)
    x * y
  }

  override def backward(dout: Double) = {
    assert(this.x.nonEmpty & this.y.nonEmpty)
    val dx = dout * this.y.get
    val dy = dout * this.x.get
    (dx, dy)
  }
}
