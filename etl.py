import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, date_format
from pyspark.sql.types import *
from pyspark.sql.functions import monotonically_increasing_id



config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID']= config.get('AWS','AWS_ACCESS_KEY_ID')
os.environ['AWS_SECRET_ACCESS_KEY']= config.get('AWS','AWS_SECRET_ACCESS_KEY')


def create_spark_session():
    """
    Function: Registers a new or re-uses the existing spark session.
    """
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    return spark


def process_song_data(spark, input_data, output_data):
    """
    Function: Processes JSON files (song data) in the path given in param: input_data 
    and saves outputs in parquet format in the path provided in param: output_data 
    
    Parameters:
            param spark: spark session
            param input_data: file path for input data
            param output_data: file path for output data
    """
    # get filepath to song data file
    song_data = os.path.join(input_data, 'song_data/*.json')
    
    ## Song Schema
    song_schema = StructType([
        StructField("artist_id", StringType()),
        StructField("artist_latitude", DoubleType()),
        StructField("artist_location", StringType()),
        StructField("artist_longitude", StringType()),
        StructField("artist_name", StringType()),
        StructField("duration", DoubleType()),
        StructField("num_songs", IntegerType()),
        StructField("title", StringType()),
        StructField("year", IntegerType()),
        ])
    
    # read song data file
    df_song = spark.read.json(song_data, schema=song_schema)

    ## List of Fields in Table songs 
    song_fields = ["title", "artist_id", "year", "duration"]
    
    ## extract columns to create songs table & Add new field song_id
    songs_table = df_song \
                .select(song_fields) \
                .dropDuplicates() \
                .withColumn("song_id", monotonically_increasing_id())

    # write songs table to parquet files partitioned by year and artist
    songs_table \
    .write.mode("overwrite") \
    .partitionBy("year", "artist_id") \
    .parquet(output_data + "songs")


    ## List of Fields in Table artists
    artists_fields = ["artist_id", "artist_name as name", "artist_location as location", 
                      "artist_latitude as latitude", "artist_longitude as longitude"]
    # extract columns to create artists table
    artists_table = df_song \
                    .selectExpr(artists_fields) \
                    .dropDuplicates()
    
    # write artists table to parquet files
    artists_table \
    .write.mode("overwrite") \
    .parquet(output_data + 'artists')


def process_log_data(spark, input_data, output_data):
    """
    Function: Processes JSON files (log data) in the path given in param: input_data 
    and saves outputs in parquet format in the path provided in param: output_data 
    
    Parameters:
            param spark: spark session
            param input_data: file path for input data
            param output_data: file path for output data
    """
    # get filepath to log data file
    log_data = os.path.join(input_data, 'log_data/*.json')

    # read log data file
    df_log = spark.read.json(log_data)
    
    # filter by actions for song plays
    df_log = df_log \
            .filter(log_df.page == 'NextSong')
  
    ## List of Fields in Table users
    users_fields = ["userId as user_id", "firstName as first_name", 
                    "lastName as last_name", "gender", "level"]
    # extract columns for users table 
    users_table = df_log \
                .selectExpr(users_fields) \
                .dropDuplicates() 
    
    # write users table to parquet files
    users_table \
    .write.mode("overwrite") \
    .parquet(output_data + 'users')

    # create timestamp column from original timestamp column
    get_timestamp = udf(lambda x: x / 1000, TimestampType())
    df_log = df_log \
            .withColumn("timestamp", get_timestamp(df_log.ts))
    
    # create datetime column from original timestamp column
    get_datetime = udf(lambda x: datetime.fromtimestamp(x), TimestampType())
    df_log = df_log \
            .withColumn("start_time", get_datetime(df_log.timestamp))
    
    # extract columns to create time table
    df_log = df_log \
            .withColumn("hour", hour("start_time")) \
            .withColumn("day", dayofmonth("start_time")) \
            .withColumn("week", weekofyear("start_time")) \
            .withColumn("month", month("start_time")) \
            .withColumn("year", year("start_time")) \
            .withColumn("weekday", dayofweek("start_time"))

    time_table = df_log \
                .select("start_time", "hour", "day", "week", "month", "year", "weekday")

    
    # write time table to parquet files partitioned by year and month
    time_table \
    .write.mode("overwrite") \
    .partitionBy("year", "month") \
    .parquet(output_data + "time")

    # read in song data to use for songplays table
    songs_df = spark.read.parquet(os.path.join(output_data, "song_data/*/*/*/*.json"))
    songs_logs = df_log \
                .join(songs_df, (df_log.song == songs_df.title))

    # extract columns from joined song and log datasets to create songplays table 
    df_artists = spark.read.parquet(os.path.join(output_data, "artists"))
    artists_songs = songs_logs \
                    .join(df_artists, (songs_logs.artist == df_artists.name))
    songplays_table = artists_songs.join(
                                        time_table,
                                        artists_songs_logs.ts == time_table.ts, 'left'
                                        ) \
                                    .drop(artists_songs.year)

    # write songplays table to parquet files partitioned by year and month
    songplays_table = songplays \
                        .select(
                            col('start_time'),
                            col('userId').alias('user_id'),
                            col('level'),
                            col('song_id'),
                            col('artist_id'),
                            col('sessionId').alias('session_id'),
                            col('location'),
                            col('userAgent').alias('user_agent'),
                            col('year'),
                            col('month'),
                            ) \
                        .repartition("year", "month")

    songplays_table \
    .write.mode("overwrite") \
    .partitionBy("year", "month") \
    .parquet(output_data, 'songplays')




def main():
    spark = create_spark_session()
    input_data = config.get('IO','INPUT_DATA')
    output_data = config.get('IO','OUTPUT_DATA') 
    
    process_song_data(spark, input_data, output_data)    
    process_log_data(spark, input_data, output_data)


if __name__ == "__main__":
    main()
