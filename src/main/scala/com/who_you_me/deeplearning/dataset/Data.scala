package com.who_you_me.deeplearning.dataset

import java.io.Serializable

class Data(prefix: String) extends Serializable {
  assert(Set("train", "test").contains(prefix))
  val filePrefix = if (prefix == "test") "t10k" else prefix
}
