val trainDF = sqlContext.read.format("csv").option("delimiter", ",").option("header", "true").load("s3://spark-aaron-grocery/train.csv")
val testDF = sqlContext.read.format("csv").option("delimiter", ",").option("header", "true").load("s3://spark-aaron-grocery/test.csv")
val oilDF = sqlContext.read.format("csv").option("delimiter", ",").option("header", "true").load("s3://spark-aaron-grocery/oil.csv")
val itemsDF = sqlContext.read.format("csv").option("delimiter", ",").option("header", "true").load("s3://spark-aaron-grocery/items.csv")
val storesDF = sqlContext.read.format("csv").option("delimiter", ",").option("header", "true").load("s3://spark-aaron-grocery/stores.csv")
val transactionsDF = sqlContext.read.format("csv").option("delimiter", ",").option("header", "true").load("s3://spark-aaron-grocery/transactions.csv")
val holidaysDF = sqlContext.read.format("csv").option("delimiter", ",").option("header", "true").load("s3://spark-aaron-grocery/holidays_events.csv").withColumnRenamed("type", "htype")

var dataDF = trainDF.join(storesDF, "store_nbr")
    dataDF = dataDF.join(itemsDF, "item_nbr")
    dataDF = dataDF.join(oilDF, "date")
    dataDF = dataDF.join(holidaysDF, Seq("date"), "left_outer")
    
    dataDF = dataDF.drop("description").drop("item_nbr").drop("dcoilwtico").drop("id").na.drop(Seq("onpromotion"))
    
var dataDF2 = testDF.join(storesDF, "store_nbr")
    dataDF2 = dataDF2.join(itemsDF, "item_nbr")
    dataDF2 = dataDF2.join(oilDF, "date")
    dataDF2 = dataDF2.join(holidaysDF, Seq("date"), "left_outer")
    
    dataDF2 = dataDF2.drop("description").drop("item_nbr").drop("dcoilwtico").drop("id").na.drop(Seq("onpromotion"))

dataDF = dataDF.select("store_nbr", "onpromotion", "city", "state", "type", "cluster", "family", "class", "perishable", "htype", "locale", "locale_name", "transferred", "date", "unit_sales")
dataDF = dataDF.withColumn("unit_sales_double", col("unit_sales").cast("double"))

dataDF = dataDF.na.fill("__NA__", Seq("htype"))
dataDF = dataDF.na.fill("__NA__", Seq("locale"))
dataDF = dataDF.na.fill("__NA__", Seq("locale_name"))
dataDF = dataDF.na.fill("__NA__", Seq("transferred"))
dataDF.show()

dataDF2 = dataDF2.na.fill("__NA__", Seq("htype"))
dataDF2 = dataDF2.na.fill("__NA__", Seq("locale"))
dataDF2 = dataDF2.na.fill("__NA__", Seq("locale_name"))
dataDF2 = dataDF2.na.fill("__NA__", Seq("transferred"))

dataDF2 = dataDF2.select("store_nbr", "onpromotion", "city", "state", "type", "cluster", "family", "class", "perishable", "htype", "locale", "locale_name", "transferred", "date")

dataDF2.show()

import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorSlicer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.regression.DecisionTreeRegressor
import org.apache.spark.ml.regression.DecisionTreeRegressionModel
import org.apache.spark.ml.regression.{GBTRegressionModel, GBTRegressor}

val StoreIndexer = new StringIndexer().setInputCol("store_nbr").setOutputCol("store_nbr_indexed").setHandleInvalid("keep")
val OnpromotionIndexer = new StringIndexer().setInputCol("onpromotion").setOutputCol("onpromotion_indexed").setHandleInvalid("keep")
val CityIndexer = new StringIndexer().setInputCol("city").setOutputCol("city_indexed").setHandleInvalid("keep")
val StateIndexer = new StringIndexer().setInputCol("state").setOutputCol("state_indexed").setHandleInvalid("keep")
val TypeIndexer = new StringIndexer().setInputCol("type").setOutputCol("type_indexed").setHandleInvalid("keep")
val ClusterIndexer = new StringIndexer().setInputCol("cluster").setOutputCol("cluster_indexed").setHandleInvalid("keep")
val FamilyIndexer = new StringIndexer().setInputCol("family").setOutputCol("family_indexed").setHandleInvalid("keep")
val ClassIndexer = new StringIndexer().setInputCol("class").setOutputCol("class_indexed").setHandleInvalid("keep")
val PerishableIndexer = new StringIndexer().setInputCol("perishable").setOutputCol("perishable_indexed").setHandleInvalid("keep")
val HtypeIndexer = new StringIndexer().setInputCol("htype").setOutputCol("htype_indexed").setHandleInvalid("keep")
val LocaleIndexer = new StringIndexer().setInputCol("locale").setOutputCol("locale_indexed").setHandleInvalid("keep")
val LocaleNameIndexer = new StringIndexer().setInputCol("locale_name").setOutputCol("locale_name_indexed").setHandleInvalid("keep")
val TransferredIndexer = new StringIndexer().setInputCol("transferred").setOutputCol("transferred_indexed").setHandleInvalid("keep")
val DateIndexer = new StringIndexer().setInputCol("date").setOutputCol("date_indexed").setHandleInvalid("keep")

val Array(trainingData, testData) = dataDF.randomSplit(Array(0.7, 0.3))

val assemblerTrain = new VectorAssembler()
                    .setInputCols(Array("store_nbr_indexed", "onpromotion_indexed", "city_indexed", "state_indexed", "type_indexed", "cluster_indexed", "family_indexed", "class_indexed", "perishable_indexed", "htype_indexed", "locale_indexed", "locale_name_indexed", "transferred_indexed", "date_indexed"))
                    .setOutputCol("features")
                    
val slicerTrain = new VectorSlicer().setInputCol("rawFeatures").setOutputCol("slicedfeatures").setNames(Array("store_nbr_indexed", "onpromotion_indexed", "city_indexed", "state_indexed", "type_indexed", "cluster_indexed", "family_indexed", "class_indexed", "perishable_indexed", "htype_indexed", "locale_indexed", "locale_name_indexed", "transferred_indexed", "date_indexed", "unit_sales_double"))

val dt = new DecisionTreeRegressor().setLabelCol("unit_sales_double").setFeaturesCol("features").setMaxBins(1000)

val gbt = new GBTRegressor()
  .setLabelCol("unit_sales_double")
  .setFeaturesCol("features")
  .setMaxIter(150)
  .setMaxBins(1000)

val pipeline = new Pipeline().setStages(Array(StoreIndexer, OnpromotionIndexer, CityIndexer, StateIndexer, TypeIndexer, ClusterIndexer, FamilyIndexer, ClassIndexer, PerishableIndexer, HtypeIndexer, LocaleIndexer, LocaleNameIndexer, TransferredIndexer, DateIndexer, assemblerTrain, gbt))

// Train model.  This also runs the indexer.
val model = pipeline.fit(trainingData)

// Make predictions.
val predictions = model.transform(testData)

import org.apache.spark.ml.evaluation.RegressionEvaluator

val evaluator = new RegressionEvaluator()
  .setLabelCol("unit_sales_double")
  .setPredictionCol("prediction")
  .setMetricName("rmse")
val rmse = evaluator.evaluate(predictions)
println("Root Mean Squared Error (RMSE) on test data = " + rmse)
