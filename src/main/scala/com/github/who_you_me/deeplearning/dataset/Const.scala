package com.github.who_you_me.deeplearning.dataset

import java.io.File

object Const {
  val dataSetDir = List(System.getenv("HOME"), ".cache", "mnist").mkString(File.separator)
}
