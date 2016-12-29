import breeze.linalg._

object Perceptron {
  def gate(w: DenseVector[Double], b: Double)(x1: Double, x2: Double): Double = {
    assert(w.size == 2)
    val x = DenseVector(x1, x2)
    if ((w dot x) + b <= 0) 0.0
    else 1.0
  }

  def AND(x1: Double, x2: Double): Double = {
    val w = DenseVector(0.5, 0.5)
    val b = -0.7
    gate(w, b)(x1, x2)
  }

  def NAND(x1: Double, x2: Double): Double = {
    val w = DenseVector(-0.5, -0.5)
    val b = 0.7
    gate(w, b)(x1, x2)
  }

  def OR(x1: Double, x2: Double): Double = {
    val w = DenseVector(0.5, 0.5)
    val b = -0.2
    gate(w, b)(x1, x2)
  }

  def XOR(x1: Double, x2: Double): Double = {
    val s1 = NAND(x1, x2)
    val s2 = OR(x1, x2)
    AND(s1, s2)
  }
}
