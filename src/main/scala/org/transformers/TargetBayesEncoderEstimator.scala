package org.transformers

import org.apache.spark.sql.{Dataset, DataFrame}
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.util.DefaultParamsWritable

import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._

import com.microsoft.ml.spark.{HasInputCol, HasOutputCol}
import com.microsoft.ml.spark.ConstructorWritable
import com.microsoft.ml.spark.ConstructorReadable
import com.microsoft.ml.spark.Wrappable

import scala.reflect.runtime.universe._

class GroupMeanModel(
                      val uid: String,
                      val bayesDF: DataFrame,
                      val inputCol: String,
                      val outputCol: String,
                      val targetCol: String
                    )
  extends Model[GroupMeanModel]
    with ConstructorWritable[GroupMeanModel]
{
  val ttag: TypeTag[GroupMeanModel] = typeTag[GroupMeanModel]
  def objectsToSave: List[Any] = List(uid, bayesDF, inputCol, outputCol, targetCol)

  override def copy(extra: ParamMap): GroupMeanModel =
    new GroupMeanModel(uid, bayesDF, inputCol, outputCol, targetCol)

  override def transform(dataset: Dataset[_]): DataFrame = {
    dataset
      .toDF
      .join(broadcast(bayesDF), Seq(inputCol), "left")
      .withColumn(
        outputCol,
        col("Probability")
          .cast(DoubleType))
      .drop("Probability")
  }

  override def transformSchema (schema: StructType): StructType = schema
    .add(
      StructField(outputCol, DoubleType)
    )
}

class TargetBayesEncoderEstimator(override val uid: String) extends Estimator[GroupMeanModel]
  with HasInputCol with HasOutputCol with Wrappable with DefaultParamsWritable
{
  def this() = this(Identifiable.randomUID("GroupImputer"))

  val targetCol: Param[String] = new Param[String](
    this, "targetCol", "Target column"
  )
  def setTargetCol(v: String): this.type = super.set(targetCol, v)
  def getTargetCol: String = $(targetCol)

  override def fit(dataset: Dataset[_]): GroupMeanModel = {
    val globalCount = dataset.count()
    val globalPosTarget = dataset.select(avg(col($(targetCol)))).first.getDouble(0)

    val probsDF = dataset
      .toDF
      .groupBy($(inputCol))
      .agg(count("*").alias("valueCount"))
      .select(col($(inputCol)), (col("valueCount")/lit(globalCount)).alias("groupProba"))

    val meanDF = dataset
      .toDF
      .groupBy($(inputCol))
      .agg(mean(col($(targetCol))).alias("groupMean"))
      .select(col($(inputCol)), col("groupMean"))

    val bayesDF = meanDF
      .join(broadcast(probsDF), Seq($(inputCol)))
      .select(col($(inputCol)),
        ((col("groupMean")*col("groupProba"))/lit(globalPosTarget))
          .alias("Probability"))

    new GroupMeanModel(
      uid, bayesDF, getInputCol, getOutputCol, getTargetCol
    )
  }

  override def transformSchema(schema: StructType): StructType =
    schema
      .add(
        StructField(
          $(outputCol),
          DoubleType
        )
      )

  override def copy(extra: ParamMap): Estimator[GroupMeanModel] = {
    val to = new TargetBayesEncoderEstimator(this.uid)
    copyValues(to, extra).asInstanceOf[TargetBayesEncoderEstimator]
  }
}

object GroupMeanModel extends ConstructorReadable[GroupMeanModel]