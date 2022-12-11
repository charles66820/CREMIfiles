# TP4

## notes

Start interactive shell :

```bash
spark-shell --master yarn
```

Build and start spark app :

```bash
cd tweets_scala
mvn package
scp target/tweets_scala-0.0.1.jar lsd_remote:
ssh lsd_remote
spark-submit --master yarn tweets_scala-0.0.1.jar
```

`/user/fzanonboito/CISD/smaller_twitter.json`
`/user/fzanonboito/CISD/tiny_twitter.json`

## 1. load & pre-process data

```scala
import org.json4s._
import org.json4s.jackson.JsonMethods._

def jsonStrToMap(jsonStr: String): Map[String, Any] = {
  implicit val formats = org.json4s.DefaultFormats
  return parse(jsonStr).extract[Map[String, Any]]
}

val data = sc.textFile("/user/fzanonboito/CISD/tiny_twitter.json")
val dataJsonMap = data.map(jsonStrToMap(_))

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

val dataTab = dataJsonMap.map(getData)
dataTab.take(1)
```
