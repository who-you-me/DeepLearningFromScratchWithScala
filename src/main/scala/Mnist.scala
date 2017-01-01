import java.io._
import java.net.URL
import java.nio.file.{Files, Paths}
import java.util.zip.GZIPInputStream

import scala.collection.mutable.ListBuffer
import breeze.linalg.{DenseMatrix, DenseVector}

class Mnist(prefix: String) {
  assert(Set("train", "test").contains(prefix))
  val filePrefix = if (prefix == "test") "t10k" else prefix

  private val imageLoader = new ImageLoader(s"$filePrefix-images-idx3-ubyte.gz")
  private val labelLoader = new LabelLoader(s"$filePrefix-labels-idx1-ubyte.gz")

  private def load(normalize: Boolean, oneHotLabel: Boolean) = {
    val images = imageLoader.loadAsMatrix(normalize)
    val labels = labelLoader.load(oneHotLabel)
    (images, labels)
  }

  private def loadFlatten(normalize: Boolean, oneHotLabel: Boolean) = {
    val images = imageLoader.loadAsVector(normalize)
    val labels = labelLoader.load(oneHotLabel)
    (images, labels)
  }
}

object Mnist {
  def load(normalize: Boolean = true, oneHotLabel: Boolean = false) = {
    val train = new Mnist("train").load(normalize, oneHotLabel)
    val test = new Mnist("test").load(normalize, oneHotLabel)
    (train, test)
  }

  def loadFlatten(normalize: Boolean = true, oneHotLabel: Boolean = false) = {
    val train = new Mnist("train").loadFlatten(normalize, oneHotLabel)
    val test = new Mnist("test").loadFlatten(normalize, oneHotLabel)
    (train, test)
  }
}

class Loader(fileName: String) {
  val urlBase = "http://yann.lecun.com/exdb/mnist/"

  private val dataSetDir = List(System.getenv("HOME"), ".cache", "mnist").mkString(File.separator)
  private val dir = new File(dataSetDir)
  if (!dir.exists) dir.mkdirs()

  private val filePath = List(dataSetDir, fileName).mkString(File.separator)
  if (!Files.exists(Paths.get(filePath))) download(fileName)

  protected val stream = new DataInputStream(new GZIPInputStream(new FileInputStream(filePath)))

  private def download(fileName: String): Unit = {
    print("Downloading " + fileName + "... ")

    val stream = new URL(urlBase + fileName).openStream()
    val buf = Stream.continually(stream.read).takeWhile(_ != -1).map(_.byteValue).toArray
    val file = new BufferedOutputStream(new FileOutputStream(filePath))

    file.write(buf)
    file.close()

    println("Done")
  }
}

class ImageLoader(fileName: String) extends Loader(fileName) {
  val magicNumber = stream.readInt()
  assert(magicNumber == 2051, "Wrong magic number: " + magicNumber)

  val count = stream.readInt()
  val rows = stream.readInt()
  val cols = stream.readInt()

  def loadAsMatrix(normalize: Boolean): List[DenseMatrix[Double]] = {
    print("Converting " + fileName + " to DenseMatrix ...")

    val buf = new ListBuffer[DenseMatrix[Double]]
    for (_ <- 0 until count) {
      buf += loadOneAsMatrix(normalize)
    }

    println("Done")
    buf.toList
  }

  def loadAsVector(normalize: Boolean): List[DenseVector[Double]] = {
    print("Converting " + fileName + " to DenseVector ...")

    val buf = new ListBuffer[DenseVector[Double]]
    for (_ <- 0 until count) buf += loadOneAsVector(normalize)

    println("Done")
    buf.toList
  }

  private def loadOneAsMatrix(normalize: Boolean): DenseMatrix[Double] = {
    val matrix = DenseMatrix.zeros[Double](rows, cols)
    for {y <- 0 until cols
         x <- 0 until rows} {
      val value: Double = stream.readUnsignedByte()
      if (normalize) matrix(y, x) = value / 255.0
      else matrix(y, x) = value
    }
    matrix
  }

  private def loadOneAsVector(normalize: Boolean): DenseVector[Double] = {
    val matrix = loadOneAsMatrix(normalize)
    matrix.t.toDenseVector
  }
}

class LabelLoader(fileName: String) extends Loader(fileName) {
  val magicNumber = stream.readInt()
  assert(magicNumber == 2049, "Wrong magic number: " + magicNumber)

  val count = stream.readInt()

  def load(oneHotLabel: Boolean): List[Int] = {
    // TODO: oneHotLabel

    print("Converting " + fileName + " to List ...")
    val buf = new ListBuffer[Int]
    for (_ <- 0 until count) buf += stream.readUnsignedByte()

    println("Done")
    buf.toList
  }
}

