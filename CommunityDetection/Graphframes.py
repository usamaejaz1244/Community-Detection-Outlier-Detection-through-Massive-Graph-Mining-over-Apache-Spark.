import os

os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages graphframes:graphframes:0.6.0-spark2.3-s_2.11 pyspark-shell'
import hashlib
import pyspark
from pyspark.sql import *
from pyspark.sql.functions import udf
from graphframes import *
from collections import Counter


sprk_cntxt = pyspark.SparkContext("local[*]")
sprk_session = SparkSession.builder.appName('CommunityDetection').getOrCreate()
sqlContext = SQLContext(sprk_cntxt)
# Reading Data from the CommonCrawl Dataset
CommonCrawl_Data = sprk_session.read.parquet("data/outlinks_pq/*.snappy.parquet")  # Parquet: column oriented data storage format for hadoop ecosystem.
# Returns the count of the Data(Data Size).
print(CommonCrawl_Data.count())

# parent: Full URL of Parent node, where the html link was pulled.
# ParentDomain: Top domain of Parent.
# Child: Full URL of Child node, the link found on the `Parent` web page.
# ChildDomain: Top domain of child.

# Renaming columns.
df = CommonCrawl_Data.withColumnRenamed("_c0", "Parent") \
    .withColumnRenamed("_c1", "ParentDomain") \
    .withColumnRenamed("_c2", "ChildDomain") \
    .withColumnRenamed("_c3", "Child") \
    .filter("ParentDomain is not null and ChildDomain is not null")

df.show(10)

#####################################################################################
# # Slicing the data
#
# # add an index column
# df = df.withColumn('id', pyspark.sql.functions.monotonically_increasing_id())
#
# # Sort by index and get first 4000 rows
# working_set = df.sort('id').limit(2000)
#
# # # Remove the working set, and use this `df` to get the next working set
# # df = df.subtract(working_set)


#####################################################################################

# flatMap : Return a new RDD by first applying a function to all elements of this RDD, and then flattening the results.
# distinct : Return a new RDD containing the distinct elements in this RDD.

# Assigining ID's to the set of parents and children Domains.
ParentChild_id = df.select("ParentDomain", "ChildDomain").rdd.flatMap(lambda x: x).distinct()
print(ParentChild_id.count())


def NodeHash(x):
    return hashlib.sha1(x.encode("UTF-8")).hexdigest()[:8]


NodeHash_udf = udf(NodeHash)
# (UDFs): Pyspark "UserDefindFunctions" are an easy way to turn your ordinary python code into something scalable.
# The basic ways is to explicitly define a udf that you can use as a pyspark function.

# map(self, f, preservesPartitioning=False) : Return a new RDD by applying a function to each element of this RDD.

Graph_Vertices = ParentChild_id.map(lambda x: (NodeHash(x), x)).toDF(["id", "name"])
Graph_Vertices.show(10)

Graph_Edges = df.select("ParentDomain", "ChildDomain") \
    .withColumn("src", NodeHash_udf("ParentDomain")) \
    .withColumn("dst", NodeHash_udf("ChildDomain")) \
    .select("src", "dst")
Graph_Edges.show(10)


# Creating GraphFrame through Graph_Vertices and Graph_Edges.
Graphs = GraphFrame(Graph_Vertices, Graph_Edges)

# Applying Label Propagation Alogorithm(LPA).
Community_Graphs = Graphs.labelPropagation(maxIter=5)
Community_Graphs.persist().show(10)
# persist : Set the RDD's storage level to persist its values across operations after the first time it is computed.

print("There are", Community_Graphs.select('label').distinct().count(), Community_Graphs, "Communities in the Dataset.")

########################################################################################################################################################
#                            Outlier Detection
########################################################################################################################################################
# Step 1: Distinct labels list -> distinct communities
#  Loop
Distinct_Communities=Community_Graphs.select('label').distinct()
# for lbl in Community_Graphs:
#     if lbl not in Distinct_Communities:
#         Distinct_Communities.append(lbl)
# print (Distinct_Communities)

# Step 2: for each distinct community get list of vertices

for com in Distinct_Communities.collect():
    Vertices_List = []
    for comm in Community_Graphs.collect():
        if com['label'] == comm['label']:
            Vertices_List.append((comm['id'],comm['name']))
    # if com in Community_Graphs["label"]:
    # Vertices_List.append(Community_Graphs["label"[com.select["id"]]])
    # Step 3: for each vertex check whole graph edges and pick edges that have the vertex in either source or destination
    New_Edges = []
    for vrtx in Vertices_List:
        for edge in Graph_Edges.collect():
            if vrtx[0] == edge['src'] or vrtx[0] == edge['dst']:
                # if vrtx in Graph_Edges["src" or "dst"]:
                New_Edges.append((edge['src'],edge['dst']))
    # Step 4: Compile all lists of edges and pick unique
    Edges_List = []
    for eg in New_Edges:
        if eg not in Edges_List:
            Edges_List.append(eg)

    print("There are " + str(len(Vertices_List)) + " Vertices in "+str(com['label']))
    # Step 5: Build graph frame from vertices of step 2 and edges of step 4
    # v = sqlContext.createDataFrame(Vertices_List, ["id", "name"])
    # e = sqlContext.createDataFrame(Edges_List, ["src", "dst"])
    # New_Graph = GraphFrame(v, e)

    # Outlier_Community = New_Graph.labelPropagation(maxIter=5)
    # print("There are", Outlier_Community.select('label').distinct().count(), Outlier_Community, "Communities in: "+str(com['label']))
    # Outlier_Community.persist().show(10)

    # Step 6: Count vertices for each label if count is below threshold then that community is outlier
    # labels_count=Counter()
    # for community in Outlier_Community.collect():
    #     labels_count[community["label"]]+=1

    # all_communities_count=labels_count.most_common()
    # print(all_communities_count[-int(len(all_communities_count)/10)])
    # print all_communities_count


    # Repeat above steps for each community


