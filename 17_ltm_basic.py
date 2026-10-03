from langgraph.store.memory import InMemoryStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings


embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    project="imperial-medium-503420-s2",
    vertexai=True,
)
#creating a store
store = InMemoryStore(index={'embed': embedding_model, 'dims':1536})

#creating a namespace
namespace = ("user", "u1")

#creating memories
#Adding memories
store.put(namespace, "2", {"data": "User prefers dark mode"})

#Another namespace 
namespace2 = ("user", "u2")
store.put(namespace2, "1", {"data": "User love mangoes"})
store.put(namespace2, "2", {"data": "User prefers silence over violence"})
store.put(namespace2, "3", {"data": "user wants Zebras to sleep"})
store.put(namespace2, "4", {"data": "User is mysterious and secretive"})
store.put(namespace2, "5", {"data": "User don't like people who are loud and obnoxious"})
store.put(namespace2, "6", {"data": "User love learn new things and explore the world"})
store.put(namespace2, "7", {"data": "User long for traveling and adventure"})
store.put(namespace2, "8", {"data": "User like crows and they are best friends of user"})
store.put(namespace2, "9", {"data": "User wants to sleep but can't"})
store.put(namespace2, "10", {"data": "User likes asthetic pics"})


#Retrieving memories
#store.get(namespace, key)

# print(store.get(namespace, "1"))
# print(store.get(namespace, "2")) 


#Retrieving all memories for a namespace
#normal search
# items = store.search(namespace)
# for item in items:
#     print(item.value)

#Semantic search
items = store.search(namespace2,query="What user likes?", limit = 3)
for item in items:
    print(item.value)
    