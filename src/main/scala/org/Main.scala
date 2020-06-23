package org

import org.apache.spark.sql.SparkSession
import com.microsoft.ml.spark.MultiColumnAdapter
import org.apache.spark.ml.feature.VectorAssembler
import ml.dmlc.xgboost4j.scala.spark.{XGBoostClassifier, XGBoostClassificationModel}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import com.microsoft.ml.spark.ComputeModelStatistics
import com.microsoft.ml.spark.metrics.MetricConstants

import org.transformers._


object Main {
  def main(args: Array[String]): Unit = {

    val spark: SparkSession = SparkSession.builder()
      .appName("MLTest")
      .enableHiveSupport()
      .getOrCreate()


    val sdf = spark.table("ml_test_data")

    val splits = sdf.select("nom_0", "nom_1", "nom_2","target")
      .na.fill(-999).randomSplit(Array(0.8, 0.2), seed = 11L)
    val trainDF_ = splits(0)
    val testDF = splits(1)

    trainDF_.persist()
    testDF.repartition(200).persist()

    val trainSplits = trainDF_.randomSplit(Array(0.9, 0.1), seed = 123)
    val trainDF = trainSplits(0)
    val evalDF = trainSplits(1)

    trainDF.repartition(200).persist()
    evalDF.repartition(200).persist()

    val features = trainDF_.columns.filter(x => x != "target")
    val newFetures = features.map(x => x + "_prob")

    val bayesEncoder = new MultiColumnAdapter()
      .setBaseStage(new TargetBayesEncoderEstimator().setTargetCol("target"))
      .setInputCols(features)
      .setOutputCols(newFetures)

    val vectorAssembler = new VectorAssembler()
      .setInputCols(bayesEncoder.getOutputCols)
      .setOutputCol("features")

    val newDF = bayesEncoder.fit(trainDF).transform(evalDF)
    val evalDfTransform = vectorAssembler.transform(newDF)

    evalDfTransform.persist()

    val params = Map(
      "missing" -> -999,
      "objective" -> "binary:logistic",
      "nthread" ->  1,
      "learning_rate" ->  0.1f,
      "n_estimators" -> 1500,
      "verbosity" ->  2,
      "max_depth" ->  5,
      "min_child_weight" ->  2.0,
      "subsample" ->  0.9,
      "colsample_bytree" ->  0.9f,
      "colsample_bylevel" ->  1.0,
      "max_bin" ->  256,
      "max_delta_step" ->  0.0,
      "scale_pos_weight" ->  1.0,
      "useExternalMemory" -> true)

    val xgboost = new XGBoostClassifier(params)
      .setFeaturesCol("features")
      .setLabelCol("target")
      .setPredictionCol("prediction")
      .setProbabilityCol("probability")
      .setRawPredictionCol("rawPrediction")
      .setNumWorkers(100)
      .setMaximizeEvaluationMetrics(true)
      .setEvalMetric("auc")
      .setNumEarlyStoppingRounds(100)
      .setEvalSets(Map("eval_df" -> evalDfTransform))

    val boostingPipeline = new Pipeline()
      .setStages(
        Array(bayesEncoder, vectorAssembler, xgboost)
      )

    val predictionsDF = boostingPipeline
      .fit(trainDF)
      .transform(testDF)

    val modelEvaluator = new ComputeModelStatistics()
      .setLabelCol("target")
      .setScoresCol("probability")
      .setScoredLabelsCol("prediction")
      .setEvaluationMetric(MetricConstants.ClassificationMetrics)

    val metrics = modelEvaluator.transform(predictionsDF)

    metrics
    .write
      .format("orc")
      .mode("overwrite")
      .saveAsTable("ml_test_metrics")

    spark.stop()



  }

}
