package com.who_you_me.deeplearning.layer

import breeze.linalg.DenseMatrix
import org.scalactic.TolerantNumerics
import org.scalatest.FunSuite

class LayerSuite extends FunSuite {
  implicit val doubleEquality = TolerantNumerics.tolerantDoubleEquality(1e-8)

  test("MulLayer and AddLayer returns correct value") {
    val apple = 100
    val appleNum = 2
    val orange = 150
    val orangeNum = 3
    val tax = 1.1

    val mulAppleLayer = new MulLayer()
    val mulOrangeLayer = new MulLayer()
    val addAppleOrangeLayer = new AddLayer()
    val mulTaxLayer = new MulLayer()

    val applePrice = mulAppleLayer.forward(apple, appleNum)
    val orangePrice = mulOrangeLayer.forward(orange, orangeNum)
    val allPrice = addAppleOrangeLayer.forward(applePrice, orangePrice)
    val price = mulTaxLayer.forward(allPrice, tax)

    val dPrice = 1
    val (dAllPrice, dTax) = mulTaxLayer.backward(dPrice)
    val (dApplePrice, dOrangePrice) = addAppleOrangeLayer.backward(dAllPrice)
    val (dOrange, dOrangeNum) = mulOrangeLayer.backward(dOrangePrice)
    val (dApple, dAppleNum) = mulAppleLayer.backward(dApplePrice)

    assert(price === 715.0)
    assert(dApple === 2.2)
    assert(dAppleNum === 110.0)
    assert(dOrange === 3.3)
    assert(dOrangeNum === 165.0)
    assert(dTax === 650.0)
  }

  {
    val x = DenseMatrix(
      (-1.0, 1.0, 0.0),
      (-10.0, -5.0, 100.0)
    )
    val dout = DenseMatrix(
      (10.0, 10.0, 10.0),
      (-20.0, -20.0, -20.0)
    )

    test("Relu returns correct value") {
      val relu = new Relu()

      val expected1 = DenseMatrix(
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 100.0)
      )
      assert(relu.forward(x) == expected1)

      val expected2 = DenseMatrix(
        (0.0, 10.0, 0.0),
        (0.0, 0.0, -20.0)
      )
      assert(relu.backward(dout) == expected2)
    }

    test("Sigmoid returns correct value") {
      val sigmoid = new Sigmoid()

      val actual1 = sigmoid.forward(x)
      val expected1 = DenseMatrix(
        (0.2689414213699951, 0.7310585786300049, 0.5),
        (4.5397868702434395e-05, 0.0066928509242848554, 1.0)
      )
      for ((a, e) <- actual1.toArray zip expected1.toArray) assert(a === e)

      val actual2 = sigmoid.backward(dout)
      val expected2 = DenseMatrix(
        (1.9661193324148185, 1.9661193324148185, 2.5),
        (-0.0009079161547190335, -0.1329611334158031, 0.0)
      )
      for ((a, e) <- actual2.toArray zip expected2.toArray) assert(a === e)
    }

    test("Affine returns correct value") {
      val w = DenseMatrix(
        (1.0, 2.0),
        (5.0, 6.0),
        (9.0, 10.0)
      )
      val b = DenseMatrix(-10.0, 10.0).t
      val affine = new Affine(w, b)

      val actual1 = affine.forward(x)
      val expected1 = DenseMatrix(
        (-6.0, 14.0),
        (855.0, 960.0)
      )
      assert(actual1 == expected1)

      val dout = DenseMatrix(
        (5.0, 10.0),
        (-10.0, -5.0)
      )
      val dx = affine.backward(dout)
      val expectedDx = DenseMatrix(
        (25.0, 85.0, 145.0),
        (-20.0, -80.0, -140.0)
      )
      assert(dx == expectedDx)

      val dW = affine.dW.get
      val expectedDW = DenseMatrix(
        (95.0, 40.0),
        (55.0, 35.0),
        (-1000.0, -500.0)
      )
      assert(dW == expectedDW)

      val db = affine.db.get
      val expectedDb = DenseMatrix(-5.0, 5.0).t
      assert(db == expectedDb)
    }

    test("SoftmaxWithLoss returns correct value") {
      val softmaxWithLoss = new SoftmaxWithLoss()

      val t = DenseMatrix(
        (0.0, 1.0, 0.0),
        (0.0, 1.0, 0.0)
      )
      assert(softmaxWithLoss.forward(x, t) === 52.703802982222193)

      val dx = softmaxWithLoss.backward()
      val expected = DenseMatrix(
        (0.09003057317038046, -0.3347590442251781, 0.24472847105479767),
        (0.0, -1.0, 1.0)
      ) / 2.0
      for ((x, e) <- dx.toArray zip expected.toArray) assert(x === e)
    }
  }
}
