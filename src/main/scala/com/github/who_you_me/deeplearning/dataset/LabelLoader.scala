package com.github.who_you_me.deeplearning.dataset

import scala.collection.mutable.ListBuffer

class LabelLoader(fileName: String) extends Loader(fileName) {
  val magicNumber = stream.readInt()
  assert(magicNumber == 2049, s"Wrong magic number: $magicNumber")

  val count = stream.readInt()

  def load(): Array[Int] = {
    print(s"Converting $fileName to Array...")

    val buf = new ListBuffer[Int]
    for (_ <- 0 until count) buf += stream.readUnsignedByte()
    stream.close()

    println("Done")
    buf.toArray
  }
}
