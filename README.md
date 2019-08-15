# Overview

**Java and Scala library for Apache Spark**

Library that can obtain feature importance of tree model prediction or classification result with column name in spark.ml.

It is licensed under [MIT](https://opensource.org/licenses/MIT).


# How to use(Java)

## Maven

```xml
<dependency>
	<groupId>org.riversun</groupId>
	<artifactId>spark-ml-feature-importance-helper</artifactId>
	<version>1.0.0</version>
</dependency>
```

## Example

You can use this library from Java.

```java
// Get model from pipeline stage
GBTRegressionModel gbtModel = (GBTRegressionModel) (pipelineModel.stages()[stageIndex]);

// Do prediction
Dataset<Row> predictions = pipelineModel.transform(testData);

// Get schema from result DataSet
StructType schema = predictions.schema();

// Get sorted feature importances with column name
List<Importance> importanceList =
       new FeatureImportance.Builder(gbtModel, schema)
         .sort(Order.DESCENDING)
         .build()
         .getResult();
```

# How To Use(Scala)

## build.sbt

```sbt
libraryDependencies += "org.riversun" % "spark-ml-feature-importance-helper" % "1.0.0"
```

## Example

```scala
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.regression.{GBTRegressionModel, GBTRegressor}
import org.apache.spark.sql.SparkSession
import org.riversun.ml.spark.FeatureImportance
import org.riversun.ml.spark.FeatureImportance.Order

object GradientBoostedTreeRegressorExample {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession
      .builder
      .appName("GradientBoostedTreeRegressorExample")
      .master("local[*]")
      .getOrCreate()

    val dataset = spark.read.format("csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load("data/mllib/gem_price.csv") // gem_price_ja.csv for Japanese

    val stringIndexers = Array("material", "shape", "brand", "shop")
      .map { colName =>
        new StringIndexer()
          .setInputCol(colName)
          .setOutputCol(colName + "Index")
      }

    val assembler = new VectorAssembler()
      .setInputCols(stringIndexers.map(indexer => indexer.getOutputCol) :+ "weight")
      .setOutputCol("features")

    val gbtr = new GBTRegressor()
      .setLabelCol("price")
      .setFeaturesCol("features")
      .setPredictionCol("prediction")

    val pipeline = new Pipeline().setStages(stringIndexers :+ assembler :+ gbtr);

    val splits = dataset.randomSplit(Array(0.7, 0.3), 1L)
    val trainingData = splits(0)
    val testData = splits(1)

    val model = pipeline.fit(trainingData)

    val predictions = model.transform(testData)

    val gbtModel = model.stages.last.asInstanceOf[GBTRegressionModel];
    val schema = predictions.schema

    val importances = new FeatureImportance.Builder(gbtModel, schema)
      .sort(Order.DESCENDING)
      .build.getResult

    importances.forEach(println)

    spark.stop()
  }
}
```

# Example result of feature importances

```
FeatureInfo [rank=0, score=0.35155564557381036, name=weight]
FeatureInfo [rank=1, score=0.23487364413432302, name=brandIndex]
FeatureInfo [rank=2, score=0.22461466434553393, name=materialIndex]
FeatureInfo [rank=3, score=0.09654096046037855, name=shapeIndex]
FeatureInfo [rank=4, score=0.09241508548595412, name=shopIndex]
```


