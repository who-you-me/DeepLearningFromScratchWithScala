package com.github.who_you_me.deeplearning.common

import breeze.linalg.DenseMatrix

object Util {
  def addIntercept(x: DenseMatrix[Double]): DenseMatrix[Double] =
  // xの先頭列に切片項（値がすべて1の列ベクトル）を追加する
    DenseMatrix.horzcat(DenseMatrix.ones[Double](x.rows, 1), x)
}
