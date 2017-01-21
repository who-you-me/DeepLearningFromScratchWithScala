package com.who_you_me.deeplearning.dataset

import java.io._
import java.nio.file.{Files, Paths}

object Mnist {
  private val path = List(Const.dataSetDir, "mnist.ser").mkString(File.separator)

  def get() = {
    if (Files.exists(Paths.get(path))) {
      val ois = new ObjectInputStream(new FileInputStream(path))

      try ois.readObject().asInstanceOf[Mnist]
      finally ois.close()
    } else {
      val mnist = new Mnist
      val oos = new ObjectOutputStream(new FileOutputStream(path))

      try oos.writeObject(mnist)
      finally oos.close()

      mnist
    }
  }
}

class Mnist extends Serializable {
  val trainImg = new ImageData("train")
  val trainLabel = new LabelData("train")
  val testImg = new ImageData("test")
  val testLabel = new LabelData("test")
}
