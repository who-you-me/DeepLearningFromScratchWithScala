package com.who_you_me.deeplearning.dataset

import breeze.linalg.DenseMatrix

class ImageData(prefix: String) extends Data(prefix) {
  private val data = new ImageLoader(s"$filePrefix-images-idx3-ubyte.gz").load()

  def getFlatten(normalize: Boolean): Stream[DenseMatrix[Double]] = {
    val xs = data.toStream.map(toDenseMatrix)
    if (normalize) xs.map(_ / 255.0) else xs
  }

  def getFlatten(indices: List[Int], normalize: Boolean): DenseMatrix[Double] = {
    val rows = for (i <- indices) yield {
      val row = toDenseMatrix(data(i))
      if (normalize) row / 255.0 else row
    }
    DenseMatrix(rows.map(_.toDenseVector): _*)
  }

  private def toDenseMatrix(x: Array[Array[Int]]): DenseMatrix[Double] =
    DenseMatrix(x: _*).map(_.toDouble)
}
