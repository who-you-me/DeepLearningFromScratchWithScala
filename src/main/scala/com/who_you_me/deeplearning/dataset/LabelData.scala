package com.who_you_me.deeplearning.dataset

import breeze.linalg.{DenseMatrix, DenseVector}

class LabelData(prefix: String) extends Data(prefix) {
  private val data = new LabelLoader(s"$filePrefix-labels-idx1-ubyte.gz").load()

  def get(): Stream[Int] = data.toStream

  def get(indices: List[Int]): List[Int] =
    for (i <- indices) yield { data(i) }

  def getOneHot(): Stream[DenseMatrix[Double]] = {
    get().map(oneHotMatrix)
  }

  def getOneHot(indices: List[Int]): DenseMatrix[Double] = {
    val rows = for (i <- indices) yield { oneHotVector(data(i)) }
    DenseMatrix(rows: _*)
  }

  private def oneHotMatrix(label: Int): DenseMatrix[Double] = {
    val matrix = DenseMatrix.zeros[Double](1, 10)
    matrix.update(0, label, 1.0)
    matrix
  }

  private def oneHotVector(label: Int): DenseVector[Double] = {
    val vector = DenseVector.zeros[Double](10)
    vector.update(label, 1.0)
    vector
  }
}
