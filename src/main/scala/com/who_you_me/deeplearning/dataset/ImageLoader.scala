package com.who_you_me.deeplearning.dataset

import scala.collection.mutable.ListBuffer

class ImageLoader(fileName: String) extends Loader(fileName) {
  val magicNumber = stream.readInt()
  assert(magicNumber == 2051, s"Wrong magic number: $magicNumber")

  val count = stream.readInt()
  val rows = stream.readInt()
  val cols = stream.readInt()

  def load(): Array[Array[Array[Int]]] = {
    print(s"Converting $fileName to 2D Array...")

    val buf = new ListBuffer[Array[Array[Int]]]
    for (_ <- 0 until count) buf += loadOne()
    stream.close()

    println("Done")
    buf.toArray
  }

  private def loadOne(): Array[Array[Int]] = {
    Array.fill[Int](rows, cols) { stream.readUnsignedByte() }
  }
}
