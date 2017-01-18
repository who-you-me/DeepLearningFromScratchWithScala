package com.github.who_you_me.deeplearning.layer

class MulLayer {
  private var x: Option[Double] = None
  private var y: Option[Double] = None

  def forward(x: Double, y: Double): Double = {
    this.x = Some(x)
    this.y = Some(y)
    x * y
  }

  def backward(dout: Double): (Double, Double) = {
    assert(this.x.nonEmpty & this.y.nonEmpty)
    val dx = dout * this.y.get
    val dy = dout * this.x.get
    (dx, dy)
  }
}
