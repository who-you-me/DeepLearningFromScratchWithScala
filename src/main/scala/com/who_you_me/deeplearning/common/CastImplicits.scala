package com.who_you_me.deeplearning.common

import breeze.linalg.{DenseMatrix, DenseVector}

object CastImplicits {
  implicit def denseVectorToDenseMatrix(x: DenseVector[Double]): DenseMatrix[Double] =
    x.toDenseMatrix
}
