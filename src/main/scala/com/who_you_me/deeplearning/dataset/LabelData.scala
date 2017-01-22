package com.who_you_me.deeplearning.dataset

import breeze.linalg.DenseMatrix

class LabelData(prefix: String) extends Data(prefix) {
  private val data = new LabelLoader(s"$filePrefix-labels-idx1-ubyte.gz").load()

  def get(): Stream[Int] = data.toStream

  private def oneHotMatrix(label: Int): DenseMatrix[Double] = {
    val matrix = DenseMatrix.zeros[Double](1, 10)
    matrix.update(0, label, 1.0)
    matrix
  }

  def getOneHot(): Stream[DenseMatrix[Double]] = {
    get().map(oneHotMatrix)
  }

  def getOneHot(indices: List[Int]): DenseMatrix[Double] = {
    val rows = for (i <- indices) yield { oneHotMatrix(data(i)) }
    DenseMatrix(rows.map(_.toDenseVector): _*)
  }
}