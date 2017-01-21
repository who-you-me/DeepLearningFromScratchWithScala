package com.who_you_me.deeplearning.common

import breeze.linalg.{*, DenseMatrix}

object Util {
  def forwardFunc(x: DenseMatrix[Double], w: DenseMatrix[Double], b: DenseMatrix[Double]): DenseMatrix[Double] = {
    val xw = x * w
    xw(*, ::) + b.toDenseVector
  }
}
