from __future__ import print_function
from pyspark.sql.types import ArrayType, StringType, IntegerType, BooleanType
from pyspark.sql.functions import udf
from pyspark import SparkConf, SparkContext, SparkFiles
from pyspark.sql import SQLContext
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, CrossValidatorModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from re import match

def main(context):
    """Main function takes a Spark SQL context."""

    # the read is from the parquet file
    comments = sqlContext.read.parquet("comments-minimal.parquet")
    submissions = sqlContext.read.parquet("submissions.parquet")
    
    # only look at columns that are useful
    comments = comments.select("id","created_utc","body","author_flair_text", "link_id")
    submissions = submissions.select("id", "title", "score")

    #comments.write.parquet("comments-minimal.parquet")
    #submissions.write.parquet("submissions.parquet")

    labeled_data = sqlContext.read.format("csv").options(header='true', inferSchema='true').load('labeled_data.csv')

    #here we do the join on comment id
    joined = comments.join(labeled_data, comments.id == labeled_data.Input_id)

    #sanitize_new ignores processed string given by sanitize
    from cleantext import sanitize
    def sanitize_new(text):
        r = sanitize(text)[1:]
        return r[0].split(" ")+r[1].split(" ")+r[2].split(" ")

    #create the udf, generate new column of n-grams
    sanitize_udf = udf(sanitize_new, ArrayType(StringType()))
    joined = joined.withColumn("ngrams", sanitize_udf(joined.body))

    #6a: construct feature vector based on "ngrams"
    #store the transformed column in "features"
    #CountVectroizer produces sparse vector by default so no need to change   
    cv = CountVectorizer(inputCol="ngrams", outputCol = "features",minDF=5.0, binary=True)
    cv_model = cv.fit(joined)
    joined = cv_model.transform(joined)

    #6b: construct pos column and neg column
    #for this project, only look at label on Trump
    pos_udf = udf(lambda label: 1 if label == 1 else 0 ,IntegerType())
    neg_udf = udf(lambda label: 1 if label ==-1 else 0 ,IntegerType())
    joined = joined.withColumn("poslabel", pos_udf(joined.labeldjt))
    joined = joined.withColumn("neglabel", neg_udf(joined.labeldjt))
    
    #7: train logistic regression model
    #code adopted from project spec
    # Initialize two logistic regression models.
#     poslr = LogisticRegression(labelCol="poslabel", featuresCol="features", maxIter=10)
#     neglr = LogisticRegression(labelCol="neglabel", featuresCol="features", maxIter=10)
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
#     #print("Training positive classifier...")
#     #posModel = posCrossval.fit(posTrain)
#     #print("Training negative classifier...")
#     #negModel = negCrossval.fit(negTrain)

#     # save the models
#     #posModel.save("www/pos.model")
#     #negModel.save("www/neg.model")

    #load instead
    posModel = CrossValidatorModel.load("www/pos.model")
    negModel = CrossValidatorModel.load("www/neg.model")
    
    #8.2 title of submission of the comment
    subid_udf = udf(lambda text: text[3:] if text[:3] == "t3_" else text, StringType()) #strip "t3_"
    comments = comments.withColumn("clean_id", subid_udf(comments.link_id))
    comments = comments.join(submissions, comments.clean_id == submissions.id).drop(submissions.id)

    #9 
    #filter out comments with "\s" and starts with "&gt"
    filter_udf = udf(lambda text: False if (match(r'^&gt', text) or match(r'\\s', text)) else True, BooleanType())
    comments = comments.filter(filter_udf(comments.body))

    #since CountVectorizer takes too long, sample from it
    comments = comments.sample(False, 0.01, 1) # 1 serves as the seed so model is reproducible
    #redo 4,5,6a 
    comments = comments.withColumn("ngrams", sanitize_udf(comments.body))
    comments = cv_model.transform(comments)
    print("done with transforming the sampled comments")
    #make predictions
    posResult = posModel.transform(comments).select("probability")
    negResult = negModel.transform(comments).select("probability")
    #label
    poslabel_udf = udf(lambda prob: 1 if prob[1] >  .2  else 0, IntegerType())
    neglabel_udf = udf(lambda prob: 1 if prob[1] >  .25 else 0, IntegerType())
    posResult = posResult.withColumn("poslabel", poslabel_udf(posResult.probability))
    negResult = negResult.withColumn("neglabel", neglabel_udf(negResult.probability))




if __name__ == "__main__":
    conf = SparkConf().setAppName("CS143 Project 2B")
    conf = conf.setMaster("local[*]")
    sc   = SparkContext(conf=conf)
    sqlContext = SQLContext(sc)
    sc.addPyFile("cleantext.py")
    print("calling main")
    main(sqlContext)
