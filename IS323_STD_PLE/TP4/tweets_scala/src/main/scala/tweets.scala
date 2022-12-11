import org.apache.spark.sql.SparkSession
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD

import org.json4s._
import org.json4s.jackson.JsonMethods._

object Tweets_Scala {

  def jsonStrToMap(jsonStr: String): Map[String, Any] = {
    implicit val formats = org.json4s.DefaultFormats
    return parse(jsonStr).extract[Map[String, Any]]
  }

  def getData(x: Map[String, Any]): Array[Any] = {
    val tweet = x("text")

    val retweeted_status = x.getOrElse("retweeted_status", None)
    val retweet_count = if (retweeted_status == None) 0
    else retweeted_status.asInstanceOf[Map[String, Any]]("retweet_count")

    val user = x.getOrElse("user", None)
    val screen_name = if (user == None) None
    else user.asInstanceOf[Map[String, Any]]("screen_name")

    val followers_count = if (user == None) 0
    else user.asInstanceOf[Map[String, Any]]("followers_count")

    val entities = x.getOrElse("entities", None)
    val hashtags = if (entities == None) List()
    else entities.asInstanceOf[Map[String, Any]]("hashtags")

    val hashtagsFull = hashtags.asInstanceOf[List[Map[String, Any]]].map("#"+_("text"))

    return Array(tweet, retweet_count, screen_name, followers_count, hashtagsFull)
  }

	def main(args: Array[String]): Unit = {
		val conf = new SparkConf().setAppName("Tweets_Scala")
		val sc = new SparkContext(conf)
		sc.setLogLevel("ERROR")

    val data = sc.textFile("/user/fzanonboito/CISD/tiny_twitter.json")
    val dataJsonMap = data.map(jsonStrToMap(_))

    val dataTab = dataJsonMap.map(getData)
    println("Number of tweets:")
    println(dataTab.count())

    // x(0).asInstanceOf[String]
    // x(1).asInstanceOf[int]
    // x(2).asInstanceOf[String]
    // x(3).asInstanceOf[int]
    // x(4).asInstanceOf[List[String]]

    val hashtags = dataTab.flatMap(x => x(4).asInstanceOf[List[String]])

    val hashtagsCount = hashtags.map(x => (x, 1)).reduceByKey(_+_)

    println("20 most popular hashtags:")
    hashtagsCount.map(x => (x._2, x._1)).top(20).foreach(println)

    // println("Number of hashtags:")
    // println(hashtags.distinct().count())

    val hashtagsByTweet = dataTab.map(_(4).asInstanceOf[List[String]]).filter(_.nonEmpty)
    val hashtagsCombByTweet = hashtagsByTweet.map(_.combinations(2).toList.map(x => (x(0), x(1))))
    val hashtagsPermByTweet = hashtagsByTweet.map(_.combinations(2).map(_.permutations.toList.map(x => (x(0), x(1)))).toList).flatMap(x => x)

    val hashtagsValPBT = hashtagsPermByTweet.flatMap(x => x).map((_, 1))
    // val hashtagsValPBT = hashtagsPermByTweet.flatMap(x => x.map((_, 1))) check perfs

    val groupAll = hashtagsValPBT.groupBy(x => x._1._1)
    val cleanVals = groupAll.map(x => (x._1, x._2.map(y => (y._1._2, y._2))))

    val sumAll = cleanVals.mapValues(_.groupBy(_._1).map(x => (x._1, x._2.foldLeft(0)((sum, i) => sum + i._2))))

    sumAll.take(20).foreach(println)
    sumAll.filter(_._1 == "#BTS").collect()(0)._2.foreach(println)
    sumAll.filter(_._1 == "#BTS").flatMap(_._2).map(x => (x._2, x._1)).top(100).foreach(println)

	}
}
