package com.github.who_you_me.deeplearning

import breeze.linalg.DenseVector

object Perceptron {
  private def gate(w: DenseVector[Double], b: Double)(x1: Double, x2: Double): Double = {
    assert(w.size == 2)
    val x = DenseVector(x1, x2)
    if ((w dot x) + b <= 0) 0.0
    else 1.0
  }

  val AND = gate(DenseVector(0.5, 0.5), -0.7)_
  val NAND = gate(DenseVector(-0.5, -0.5), 0.7)_
  val OR = gate(DenseVector(0.5, 0.5), -0.2)_

  def XOR(x1: Double, x2: Double): Double = {
    val s1 = NAND(x1, x2)
    val s2 = OR(x1, x2)
    AND(s1, s2)
  }
}
