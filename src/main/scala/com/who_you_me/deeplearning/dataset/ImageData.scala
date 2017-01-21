package com.who_you_me.deeplearning.dataset

import breeze.linalg.DenseMatrix

class ImageData(prefix: String) extends Data(prefix) {
  private val data = new ImageLoader(s"$filePrefix-images-idx3-ubyte.gz").load()

  def get(normalize: Boolean = true): Stream[DenseMatrix[Double]] = {
    val xs = data.toStream.map(toDenseMatrix)
    if (normalize) normalizeFunc(xs)
    else xs
  }

  def getFlatten(normalize: Boolean = true): Stream[DenseMatrix[Double]] =
    flattenFunc(get(normalize))

  private def toDenseMatrix(x: Array[Array[Int]]): DenseMatrix[Double] =
    DenseMatrix(x: _*).map(_.toDouble)

  private def normalizeFunc(xs: Stream[DenseMatrix[Double]]): Stream[DenseMatrix[Double]] =
    xs.map(_ / 255.0)

  private def flattenFunc(xs: Stream[DenseMatrix[Double]]): Stream[DenseMatrix[Double]] =
    xs.map(_.t.toDenseVector.toDenseMatrix)
}
