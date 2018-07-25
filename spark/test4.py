from pyspark import SparkContext

logFile = "/Users/hhy/desktop/1.txt"
sc = SparkContext("local", "Simple App")
logData = sc.textFile(logFile).cache()

numAs = logData.filter(lambda s: 'w' in s).count()
numBs = logData.filter(lambda s: 'l' in s).first()
#包含W或l的行数
numCs = numBs.union(numAs).first()
print(numAs, numBs)

''''///////////////'''
sc = SparkContext("local", "Simple App")
nums=sc.parallelize([1,2,3,4])
sum=nums.map(lambda x:x*x).collect()
for num in sum:
    print(num)

''''///////////////'''
#单词计数
logFile = "/Users/hhy/desktop/1.txt"
sc = SparkContext("local", "Simple App")
rdd=sc.textFile(logFile).cache()
words=rdd.flatMap(lambda x:x.split(" "))
result=words.map(lambda x:(x,1)).reduceByKey(lambda x,y:x+y).collect()
#new=result.map(lambda k_v:(k_v[1],k_v[0])).sortByKey(ascending=False).collect()
#words=rdd.flatMap(lambda x:x.split(" ")).countByValue()
#查看rdd分区数
#print(rdd.getNumPartitions())
print(result)

''''///////////////'''
#dataframe
from pyspark.sql import SparkSession
spark=SparkSession.builder.appName("example").config("spark.some.config.option","some-value").getOrCreate()
df=spark.read.json("/Users/hhy/Desktop/people.json")
#df.show()
#df.select("name").show()
rdd=df.rdd.collect()
for num in rdd:
    print(num)

''''///////////////'''
#map与flapmap
lines=sc.parallelize(["hello world","hi"])
#words=lines.map(lambda line:line.split(" "))
words=lines.flatMap(lambda line:line.split(" "))
print(words.first())