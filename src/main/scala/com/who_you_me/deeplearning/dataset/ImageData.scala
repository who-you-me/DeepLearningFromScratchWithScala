package com.who_you_me.deeplearning.dataset

import breeze.linalg.{DenseMatrix, DenseVector}

class ImageData(prefix: String) extends Data(prefix) {
  private val data = new ImageLoader(s"$filePrefix-images-idx3-ubyte.gz").load()

  def get(normalize: Boolean): Stream[DenseMatrix[Double]] =
    data.toStream.map(toDenseMatrix(convert(normalize)))

  def getFlatten(normalize: Boolean): Stream[DenseMatrix[Double]] =
    data.toStream.map(toFlattenDenseMatrix(convert(normalize)))

  def getFlatten(indices: List[Int], normalize: Boolean): DenseMatrix[Double] = {
    val func = toDenseVector(convert(normalize))_
    val rows = for (i <- indices) yield { func(data(i)) }
    DenseMatrix(rows: _*)
  }

  private def convert(normalize: Boolean)(x: Int): Double = {
    if (normalize) x.toDouble / 255
    else x.toDouble
  }

  private def toDenseMatrix(f: Int => Double)(x: Array[Array[Int]]): DenseMatrix[Double] =
    // Array[Array[Int](28)](28)をDenseMatrix[Double](28, 28)に変換する
    // 個々の値はfで変換する
    DenseMatrix(x: _*).map(f)

  private def toFlattenDenseMatrix(f: Int => Double)(x: Array[Array[Int]]): DenseMatrix[Double] =
    // Array[Array[Int](28)](28)をDenseMatrix[Double](1, 784)に変換する
    // 個々の値はfで変換する
    DenseMatrix(x.flatten: _*).t.map(f)

  private def toDenseVector(f: Int => Double)(x: Array[Array[Int]]): DenseVector[Double] =
    // Array[Array[Int](28)](28)をDenseVector[Double](784)に変換する
    // 個々の値はfで変換する
    DenseVector(x.flatten: _*).map(f)
}
