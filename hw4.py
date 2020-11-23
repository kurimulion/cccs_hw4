# Add Spark Python Files to Python Path
from sgd import logisticRegression
import sys
import os
SPARK_HOME = "/usr/lib/spark" # Set this to wherever you have compiled Spark
os.environ["SPARK_HOME"] = SPARK_HOME # Add Spark path
os.environ["SPARK_LOCAL_IP"] = "10.128.0.10" # Set Local IP
sys.path.append(SPARK_HOME + "/python") # Add python files to Python Path
sys.path.append(SPARK_HOME + "/python/lib")

from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark.mllib.regression import LabeledPoint
from pyspark import SparkConf, SparkContext


def getSparkContext():
    """
    Gets the Spark Context
    """
    conf = (SparkConf()
         .setMaster("local") # run on local
         .setAppName("Logistic Regression") # Name of App
         .set("spark.executor.memory", "1g")) # Set 1 gig of memory
    sc = SparkContext(conf = conf) 
    return sc

def mapper(line):
    """
    Mapper that converts an input line to a feature vector
    """
    values = [float(x) for x in line.split(',')]
    return LabeledPoint(values[-1], values[:-1])

sc = getSparkContext()

# Load and parse the data
data = sc.textFile("gs://cccs_hw3/data_banknote_authentication.txt")
parsedData = data.map(mapper)

# Train model
model = LogisticRegressionWithSGD.train(parsedData)

# Predict the first elem will be actual data and the second 
# item will be the prediction of the model
labelsAndPreds = parsedData.map(lambda point: (point.label, 
                                model.predict(point.features)))

# Evaluating the model on training data
trainErr = labelsAndPreds.filter(lambda point: point[0] != point[1]).count() / float(parsedData.count())

# Print some stuff
print("Training Error = " + str(trainErr))

model = logisticRegression(1, 4)
model.train(parsedData)
labelsAndPreds = parsedData.map(lambda point: (point.label, 
                                model.predict(point.features)))
trainErr = labelsAndPreds.filter(lambda point: point[0] != point[1]).count() / float(parsedData.count())
print("Training Error = " + str(trainErr))