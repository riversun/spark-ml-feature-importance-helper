# Overview

**Java library for Apache Spark**

Library that can obtain feature importance of tree model prediction or classification result with column name in spark.ml.

It is licensed under [MIT](https://opensource.org/licenses/MIT).

# How to use

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


## Example result of feature importances

```
FeatureInfo [rank=0, score=0.3015643580333446, name=weight]
FeatureInfo [rank=1, score=0.2707593044437997, name=materialIndex]
FeatureInfo [rank=2, score=0.20696065038166056, name=brandIndex]
FeatureInfo [rank=3, score=0.11316392134864546, name=shapeIndex]
FeatureInfo [rank=4, score=0.10755176579254973, name=shopIndex]
```

