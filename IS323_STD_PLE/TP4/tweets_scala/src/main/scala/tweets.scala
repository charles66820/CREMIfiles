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
    // dataTab.take(1).foreach(println)

    println("Oui :")
		println(dataTab.count())
		dataTab.take(10).foreach(println)
	}
}
