package com.who_you_me.deeplearning.layer

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
}
