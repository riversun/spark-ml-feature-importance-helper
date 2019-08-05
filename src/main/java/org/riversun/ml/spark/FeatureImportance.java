package org.riversun.ml.spark;

import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collector;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.apache.spark.ml.PredictionModel;
import org.apache.spark.ml.classification.DecisionTreeClassificationModel;
import org.apache.spark.ml.classification.GBTClassificationModel;
import org.apache.spark.ml.classification.RandomForestClassificationModel;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.regression.DecisionTreeRegressionModel;
import org.apache.spark.ml.regression.GBTRegressionModel;
import org.apache.spark.ml.regression.RandomForestRegressionModel;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

/**
 * Helper to associate column name and feature importance
 *
 * Code example
 * <code>
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
 * </code>
 * 
 * @author Tom Misawa (riversun.org@gmail.com)
 *
 */
public class FeatureImportance {

    public enum Order {
        ASCENDING, DESCENDING, UNSORTED
    }

    private PredictionModel<Vector, ?> model;
    private StructType schema;
    private Order sort;

    public static class Builder {

        private PredictionModel<Vector, ?> model;
        private StructType schema;
        private Order sort = Order.DESCENDING;;

        /**
         * Set RegressionModel or ClassificationModel of Desicion Tree series
         * 
         * @param model
         *            Only the following classes that support the featureimportance
         *            can be set.
         * 
         *            GBTRegressionModel, GBTClassificationModel,
         *            RandomForestRegressionModel, RandomForestClassificationModel,
         *            DecisionTreeRegressionModel, DecisionTreeClassificationModel
         * @param schema
         *            Set schema from Dataset<Row>{@link #schema(StructType)}
         */
        public Builder(PredictionModel<Vector, ?> model, StructType schema) {
            this.model = model;
            this.schema = schema;
        }

        /**
         * Sort by score
         * 
         * @param sort
         * @return
         */
        public Builder sort(Order sort) {
            this.sort = sort;
            return Builder.this;
        }

        /**
         * Build FeatureImportance Object
         * 
         * 
         * @return
         */
        public FeatureImportance build() {

            if (model == null || schema == null) {
                throw new NullPointerException();
            }

            return new FeatureImportance(this);
        }

    }

    private FeatureImportance(Builder builder) {
        this.model = builder.model;
        this.schema = builder.schema;
        this.sort = builder.sort;
    }

    /**
     * Link the column and the importance and returns as sorted List
     * 
     * @return
     */
    public List<Importance> getResult() {

        final Vector featureImportances;

        if (this.model instanceof GBTRegressionModel) {
            featureImportances = ((GBTRegressionModel) this.model).featureImportances();
        } else if (this.model instanceof GBTClassificationModel) {
            featureImportances = ((GBTClassificationModel) this.model).featureImportances();
        } else if (this.model instanceof RandomForestRegressionModel) {
            featureImportances = ((RandomForestRegressionModel) this.model).featureImportances();
        } else if (this.model instanceof RandomForestClassificationModel) {
            featureImportances = ((RandomForestClassificationModel) this.model).featureImportances();
        } else if (this.model instanceof DecisionTreeRegressionModel) {
            featureImportances = ((DecisionTreeRegressionModel) this.model).featureImportances();
        } else if (this.model instanceof DecisionTreeClassificationModel) {
            featureImportances = ((DecisionTreeClassificationModel) this.model).featureImportances();
        } else {
            throw new RuntimeException(this.model + " doesn't have feature importances."
                    + "You should specify an instance of "
                    + "GBTRegressionModel,GBTClassificationModel,"
                    + "RandomForestRegressionModel,RandomForestClassificationModel,"
                    + "DecisionTreeRegressionModel,DecisionTreeClassificationModel");
        }

        return zipImportances(featureImportances, this.model.getFeaturesCol(), schema);
    }

    private List<Importance> zipImportances(Vector featureImportances, String featuresCol, StructType schema) {

        final int indexOfFeaturesCol = (Integer) schema.getFieldIndex(featuresCol).get();

        final StructField featuresField = schema.fields()[indexOfFeaturesCol];

        final Metadata metadata = featuresField
                .metadata();

        final Metadata featuresFieldAttrs = metadata
                .getMetadata("ml_attr")
                .getMetadata("attrs");

        final Map<Integer, String> idNameMap = new HashMap<>();

        final String[] fieldKeys = { "nominal", "numeric", "binary" };

        final Collector<Metadata, ?, HashMap<Integer, String>> metaDataMapperFunc = Collectors
                .toMap(
                        metaData -> (int) metaData.getLong("idx"), // key of map
                        metaData -> metaData.getString("name"), // value of map
                        (oldVal, newVal) -> newVal,
                        HashMap::new);

        for (String fieldKey : fieldKeys) {
            if (featuresFieldAttrs.contains(fieldKey)) {
                idNameMap.putAll(Arrays
                        .stream(featuresFieldAttrs.getMetadataArray(fieldKey))
                        .collect(metaDataMapperFunc));
            }
        }

        final double[] importanceScores = featureImportances.toArray();

        final List<Importance> rawImportanceList = IntStream
                .range(0, importanceScores.length)
                .mapToObj(idx -> new Importance(idx, idNameMap.get(idx), importanceScores[idx], 0))
                .collect(Collectors.toList());

        final List<Importance> descSortedImportanceList = rawImportanceList
                .stream()
                .sorted(Comparator.comparingDouble((Importance ifeature) -> ifeature.score).reversed())
                .collect(Collectors.toList());

        for (int i = 0; i < descSortedImportanceList.size(); i++) {
            descSortedImportanceList.get(i).rank = i;
        }

        final List<Importance> finalImportanceList;

        switch (this.sort) {
        case ASCENDING:
            final List<Importance> ascSortedImportantFeatureList = descSortedImportanceList
                    .stream()
                    .sorted(Comparator.comparingDouble((Importance ifeature) -> ifeature.score))
                    .collect(Collectors.toList());

            finalImportanceList = ascSortedImportantFeatureList;
            break;

        case DESCENDING:
            finalImportanceList = descSortedImportanceList;
            break;

        case UNSORTED:
        default:
            finalImportanceList = rawImportanceList;
            break;
        }

        return finalImportanceList;

    }
}
