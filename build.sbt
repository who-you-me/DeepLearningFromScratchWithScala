name := "DeepLearningFromScratch"

version := "1.0"

scalaVersion := "2.11.8"

libraryDependencies  ++= Seq(
  "org.scalanlp" %% "breeze" % "0.12",
  "org.scalanlp" %% "breeze-natives" % "0.12"
)

resolvers += "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"
