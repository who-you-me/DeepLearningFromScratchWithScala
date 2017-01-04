import org.scalatest.FunSuite

import com.github.who_you_me.deeplearning.Perceptron._

class PerceptronSuite extends FunSuite {
  test("AND function returns correct output") {
    assert(AND(0, 0) == 0.0)
    assert(AND(0, 1) == 0.0)
    assert(AND(1, 0) == 0.0)
    assert(AND(1, 1) == 1.0)
  }

  test("NAND function returns correct output") {
    assert(NAND(0, 0) == 1.0)
    assert(NAND(0, 1) == 1.0)
    assert(NAND(1, 0) == 1.0)
    assert(NAND(1, 1) == 0.0)
  }

  test("OR function returns correct output") {
    assert(OR(0, 0) == 0.0)
    assert(OR(0, 1) == 1.0)
    assert(OR(1, 0) == 1.0)
    assert(OR(1, 1) == 1.0)
  }

  test("XOR function returns correct output") {
    assert(XOR(0, 0) == 0.0)
    assert(XOR(0, 1) == 1.0)
    assert(XOR(1, 0) == 1.0)
    assert(XOR(1, 1) == 0.0)
  }
}
