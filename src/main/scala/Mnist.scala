import java.io._
import java.net.URL
import java.nio.file.{Files, Paths}
import java.util.zip.GZIPInputStream

import scala.collection.mutable.ListBuffer
import breeze.linalg.{DenseMatrix, DenseVector}

object Const {
  val dataSetDir = List(System.getenv("HOME"), ".cache", "mnist").mkString(File.separator)
}

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

class Data(prefix: String) extends Serializable {
  assert(Set("train", "test").contains(prefix))
  val filePrefix = if (prefix == "test") "t10k" else prefix
}

class ImageData(prefix: String) extends Data(prefix) {
  private val data = new ImageLoader(s"$filePrefix-images-idx3-ubyte.gz").load()

  def get(normalize: Boolean = true): Stream[DenseMatrix[Double]] =
    if (normalize) normalizeFunc(data.toStream)
    else data.toStream

  def getFlatten(normalize: Boolean = true): Stream[DenseVector[Double]] = {
    flattenFunc(get(normalize))
  }

  private def normalizeFunc(xs: Stream[DenseMatrix[Double]]): Stream[DenseMatrix[Double]] =
    xs.map(_ / 255.0)

  private def flattenFunc(xs: Stream[DenseMatrix[Double]]): Stream[DenseVector[Double]] =
    xs.map(_.t.toDenseVector)
}

class LabelData(prefix: String) extends Data(prefix) {
  private val data = new LabelLoader(s"$filePrefix-labels-idx1-ubyte.gz").load()

  def get(): Stream[Int] = data.toStream

  //  def getOneHot(): List[List[Int]]
}

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

class ImageLoader(fileName: String) extends Loader(fileName) {
  val magicNumber = stream.readInt()
  assert(magicNumber == 2051, s"Wrong magic number: $magicNumber")

  val count = stream.readInt()
  val rows = stream.readInt()
  val cols = stream.readInt()

  def load(): List[DenseMatrix[Double]] = {
    print(s"Converting $fileName to DenseMatrix...")

    val buf = new ListBuffer[DenseMatrix[Double]]
    for (_ <- 0 until count) buf += loadOne()
    stream.close()

    println("Done")
    buf.toList
  }

  private def loadOne(): DenseMatrix[Double] = {
    val matrix = DenseMatrix.zeros[Double](rows, cols)
    for {y <- 0 until cols
         x <- 0 until rows} {
      matrix(y, x) = stream.readUnsignedByte()
    }
    matrix
  }
}

class LabelLoader(fileName: String) extends Loader(fileName) {
  val magicNumber = stream.readInt()
  assert(magicNumber == 2049, s"Wrong magic number: $magicNumber")

  val count = stream.readInt()

  def load(): List[Int] = {
    print(s"Converting $fileName to List...")

    val buf = new ListBuffer[Int]
    for (_ <- 0 until count) buf += stream.readUnsignedByte()
    stream.close()

    println("Done")
    buf.toList
  }
}
