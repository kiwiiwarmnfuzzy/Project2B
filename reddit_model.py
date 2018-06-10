from __future__ import print_function
from pyspark.sql.types import ArrayType, StringType, IntegerType, BooleanType
from pyspark.sql.functions import udf, from_unixtime, lower, regexp_replace
from pyspark import SparkConf, SparkContext, SparkFiles
from pyspark.sql import SQLContext
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, CrossValidatorModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from re import match

# use hashtable to speed up look up
states = {'alabama', 'alaska', 'arizona', 'arkansas', 'california', 'colorado', 'connecticut', 'delaware', 'district of columbia', 'florida', 'georgia', 'hawaii', 'idaho', 'illinois', 'indiana', 'iowa', 'kansas', 'kentucky', 'louisiana', 'maine', 'maryland', 'massachusetts', 'michigan', 'minnesota', 'mississippi', 'missouri', 'montana', 'nebraska', 'nevada', 'new hampshire', 'new jersey', 'new mexico', 'new york', 'north carolina', 'north dakota', 'ohio', 'oklahoma', 'oregon', 'pennsylvania', 'rhode island', 'south carolina', 'south dakota', 'tennessee', 'texas', 'utah', 'vermont', 'virginia', 'washington', 'west virginia', 'wisconsin', 'wyoming'}

sampleRate = 0.05

def main(context):
    """Main function takes a Spark SQL context."""
    # TASK 1
    # the read is from the parquet file
    comments = sqlContext.read.parquet("comments-minimal.parquet")
    submissions = sqlContext.read.parquet("submissions.parquet")
    
    # only look at columns that are useful
    comments = comments.select("id","created_utc","body","author_flair_text", "link_id", "score").\
        withColumnRenamed("score", "commentscore")
    submissions = submissions.select("id", "title", "score").\
        withColumnRenamed("score", "storyscore")

    #comments.write.parquet("comments-minimal.parquet")
    #submissions.write.parquet("submissions.parquet")
    
    # TASK 2
    labeled_data = sqlContext.read.format("csv").options(header='true', inferSchema='true').load('labeled_data.csv')

    #here we do the join on comment id
    #joined = comments.join(labeled_data, comments.id == labeled_data.Input_id)
    comments.join(labeled_data, comments.id == labeled_data.Input_id).explain()
    return
    # TASK 4
    #sanitize_new ignores processed string given by sanitize
    from cleantext import sanitize
    def sanitize_new(text):
        r = sanitize(text)[1:]
        return r[0].split(" ")+r[1].split(" ")+r[2].split(" ")

    # TASK 5
    #create the udf, generate new column of n-grams
    sanitize_udf = udf(sanitize_new, ArrayType(StringType()))
    joined = joined.withColumn("ngrams", sanitize_udf(joined.body))

    # TASK 6A
    # construct feature vector based on "ngrams"
    #store the transformed column in "features"
    #CountVectroizer produces sparse vector by default so no need to change   
    cv = CountVectorizer(inputCol="ngrams", outputCol = "features",minDF=5.0, binary=True)
    cv_model = cv.fit(joined)
    joined = cv_model.transform(joined)

    # TASK 6B
    # construct pos column and neg column
    #for this project, only look at label on Trump
    pos_udf = udf(lambda label: 1 if label == 1 else 0 ,IntegerType())
    neg_udf = udf(lambda label: 1 if label ==-1 else 0 ,IntegerType())
    joined = joined.withColumn("poslabel", pos_udf(joined.labeldjt))
    joined = joined.withColumn("neglabel", neg_udf(joined.labeldjt))
    
    # TASK 7
    #train logistic regression model
    #code adopted from project spec
#     #Initialize two logistic regression models.
#     poslr = LogisticRegression(labelCol="poslabel", featuresCol="features", maxIter=10)
#     neglr = LogisticRegression(labelCol="neglabel", featuresCol="features", maxIter=10)
#     poslr.setThreshold(0.2)
#     neglr.setThreshold(0.25)
#     # This is a binary classifier so we need an evaluator that knows how to deal with binary classifiers.
#     posEvaluator = BinaryClassificationEvaluator(labelCol="poslabel")
#     negEvaluator = BinaryClassificationEvaluator(labelCol="neglabel")
#     # There are a few parameters associated with logistic regression. We do not know what they are a priori.
#     # We do a grid search to find the best parameters. We can replace [1.0] with a list of values to try.
#     # We will assume the parameter is 1.0. Grid search takes forever.
#     posParamGrid = ParamGridBuilder().addGrid(poslr.regParam, [1.0]).build()
#     negParamGrid = ParamGridBuilder().addGrid(neglr.regParam, [1.0]).build()
#     # We initialize a 5 fold cross-validation pipeline.
#     posCrossval = CrossValidator(
#         estimator=poslr,
#         evaluator=posEvaluator,
#         estimatorParamMaps=posParamGrid,
#         numFolds=5)
#     negCrossval = CrossValidator(
#         estimator=neglr,
#         evaluator=negEvaluator,
#         estimatorParamMaps=negParamGrid,
#         numFolds=5)
#     # Although crossvalidation creates its own train/test sets for
#     # tuning, we still need a labeled test set, because it is not
#     # accessible from the crossvalidator (argh!)
#     # Split the data 50/50
#     posTrain, posTest = joined.randomSplit([0.5, 0.5])
#     negTrain, negTest = joined.randomSplit([0.5, 0.5])

#     # Train the models
#     print("Training positive classifier...")
#     posModel = posCrossval.fit(posTrain)
#     print("Training negative classifier...")
#     negModel = negCrossval.fit(negTrain)

#     # save the models
#     posModel.save("www/pos.model")
#     negModel.save("www/neg.model")

    #load instead
    posModel = CrossValidatorModel.load("www/pos.model")
    negModel = CrossValidatorModel.load("www/neg.model")
    #print("finished loading model")
    
    # TASK 8.1
    # selected column 'created_utc' and transformed in 10.2 using from_unixtime

    # TASK 8.2
    # title of submission of the comment
    comments = comments.withColumn("clean_id", regexp_replace("link_id", r'^t3_', ''))
    comments = comments.join(submissions, comments.clean_id == submissions.id).drop(submissions.id)
    
    # TASK 9 
    #filter out comments with "\s" and starts with "&gt"
    comments = comments.filter(~comments.body.rlike(r'^&gt')).\
        filter(~comments.body.rlike(r'\\s'))
    #sample
    comments = comments.sample(False, sampleRate, None) # 1 serves as the seed so model is reproducible
    #redo 4,5,6a 
    comments = comments.withColumn("ngrams", sanitize_udf(comments.body))
    comments = cv_model.transform(comments)
    #print("done with transforming the sampled comments")

    #make predictions
    comments = posModel.transform(comments).\
        drop("body", "link_id", "clean_id", "ngrams","rawPrediction", "probability").\
        withColumnRenamed("prediction", "poslabel")
    comments = negModel.transform(comments).drop("features", "rawPrediction", "probability").\
        withColumnRenamed("prediction", "neglabel")

    # TASK 10.1
    # compute the percentage of positive, negative comments 
    #print("Percentage of positive comments")
    result = comments.select('poslabel').groupBy().avg()
    result.repartition(1).write.format("com.databricks.spark.csv").\
        option("header","true").save("pos-perc.csv")
    #print("Percenetage of negative comments")
    result = comments.select('neglabel').groupBy().avg()
    result.repartition(1).write.format("com.databricks.spark.csv").\
        option("header","true").save("neg-perc.csv")

    # TASK 10.2
    #2. by date
    comments = comments.withColumn("date", from_unixtime(comments.created_utc, "YYYY-MM-dd"))
    result = comments.groupBy("date").agg({"poslabel" : "mean", "neglabel" : "mean"})
    result.repartition(1).write.format("com.databricks.spark.csv").\
        option("header","true").save("time_data.csv")

    # TASK 10.3
    #3. by state
    val_state_udf = udf(lambda state: state if state in states else None, StringType())
    comments = comments.withColumn("state", val_state_udf(lower(comments.author_flair_text)))
    comments = comments.filter(comments.state.isNotNull())
    result = comments.groupBy("state").agg({"poslabel" : "mean", "neglabel" : "mean"})
    result.show(truncate=False)
    #print(result.count())
    result.repartition(1).write.format("com.databricks.spark.csv").\
        option("header","true").save("state_data.csv")
    
    # TASK 10.4
    #4a. by comment score
    result = comments.groupBy("commentscore").agg({"poslabel" : "mean", "neglabel" : "mean"})
    result.repartition(1).write.format("com.databricks.spark.csv").\
        option("header","true").save("comment_score.csv")
    #4b. by story score
    result = comments.groupBy("storyscore").agg({"poslabel" : "mean", "neglabel" : "mean"})
    result.repartition(1).write.format("com.databricks.spark.csv").\
        option("header","true").save("story_score.csv")

    # DELIVERABLE 4.
    story = result.orderBy('avg(poslabel)', ascending = False).limit(10)
    # join is too expensive, subquery is also expensive
    score_list = set(story.select('storyscore').toPandas()['storyscore'])
    comments[comments.storyscore.isin(score_list)].select('storyscore','title').limit(20).show(truncate = False)

    story = result.orderBy('avg(neglabel)', ascending = False).limit(10)
    score_list = set(story.select('storyscore').toPandas()['storyscore'])
    comments[comments.storyscore.isin(score_list)].select('storyscore','title').limit(20).show(truncate = False)


if __name__ == "__main__":
    conf = SparkConf().setAppName("CS143 Project 2B")
    conf = conf.setMaster("local[*]")
    sc   = SparkContext(conf=conf)
    sc.setLogLevel('WARN')
    sqlContext = SQLContext(sc)
    sc.addPyFile("cleantext.py")
    main(sqlContext)
