package com.github.who_you_me.deeplearning.dataset

import java.io._
import java.net.URL
import java.nio.file.{Files, Paths}
import java.util.zip.GZIPInputStream

class Loader(fileName: String) {
  val urlBase = "http://yann.lecun.com/exdb/mnist/"

  val dataSetDir = Const.dataSetDir
  if (!Files.exists(Paths.get(dataSetDir))) new File(dataSetDir).mkdirs()

  private val path = List(dataSetDir, fileName).mkString(File.separator)
  if (!Files.exists(Paths.get(path))) download(fileName)

  protected val stream = new DataInputStream(new GZIPInputStream(new FileInputStream(path)))

  private def download(fileName: String): Unit = {
    print(s"Downloading $fileName...")

    val stream = new URL(urlBase + fileName).openStream()
    val buf = Stream.continually(stream.read).takeWhile(_ != -1).map(_.byteValue).toArray
    val file = new BufferedOutputStream(new FileOutputStream(path))

    try file.write(buf)
    finally { file.close(); stream.close() }

    println("Done")
  }
}
