#!/usr/bin/env python
# coding: utf-8

# <H1>Homework 3 (Group 23) - Find the perfect place to stay in Texas!</H1>
# 

# <p> The goal of this homework is to create different search engines for <b>Airbnb</b> queries, using different methods to retrieve and eventually rank the results. The first step is to read the data corresponding to the different available Airbnb solutions, from a <b>.csv</b> file, and store them in a <b>dataframe</b>.  After that, we will be able to store each row of the dataframe  in a <b>.tsv</b> document, and use these .tsv files for any following information retrieval.</p>

# <H2> 1) Data </H2>

# <H3> Importing variables and libraries </H3>

# For the sake of order and tidyness, we choose to put most of the global variables in a separate module, ***gvf.py***, where we also import the libraries we are going to use. 

# In[1]:


# Choose file path, for the .tsv documents we are going to store (as they are many thousands, it's a good idea
# to store them in a different folder)

file_path = "docs"
#file_path = "airbnb_data"


# In[2]:


# Import modules and variables from external file, including the dataframe containing the .csv file.
import pandas as pd
from gvf import *


# In[ ]:


# Remove duplicates from the dataframe, and update the index
df = df.drop_duplicates(subset=["title", "description"], keep = False)
df.index = list(range(len(df)))


# <H2> 2) Create documents </H2>
# 
# <p> As announced, the first step is to use pandas to read the .csv file and visualize it on a dataframe. Then, we will create the documents corresponding to the different entries in the dataframe </p>

# <H3> Read .csv file </H3>

# In[4]:


# Print the first line of the DataFrame, as a check

df.head()


# <H3> Create .tsv files </H3>

# <p>
# To create each .tsv file, we iterate the .csv source file, line by line, storing each line of the information in variables, one  for each field. Once the end of the row is reached, we join the information into a new file where the data will be separated by tabulations.  The process will be repeated until we reach the last row of the source document.
# </p>

# In[5]:


# Create a new file with .tsv extension for each row of the dataset

for i in range(len(df)):

    # Open .tsv file.
    file = open(file_path + "/doc_" + str(i) + ".tsv", "w", encoding = "utf-8")
        
    # Get data from dataframe and saving it into a variable
    average_rate_per_night = str(df["average_rate_per_night"][i])
    bedrooms_count = str(df["bedrooms_count"][i])
    city = str(df["city"][i])
    date_of_listing = str(df["date_of_listing"][i])    
    description = str(df["description"][i])
    latitude = str(df["latitude"][i])
    longitude = str(df["longitude"][i])
    title = str(df["title"][i])
    url = str(df["url"][i])
                
    # Join the fields that will be written in the file
    entry = "	".join([average_rate_per_night, 
                        bedrooms_count,
                        city, 
                        date_of_listing, 
                        description, 
                        latitude, 
                        longitude,
                        title,
                        url
                       ])
    
    
    # Write in the .tsv file
    file.write(entry)
    
    # Close the .tsv file
    file.close()


# <p>
#     We now have a file with .tsv extension for each line of the original .csv document.
# </p>

# <H3> Modify .tsv files </H3>
# 
# <p> We are now going to modify the information stored on <b>description</b> and <b>title</b>. In particular, we are going to convert all the words to lowercase, and remove line breaks and punctuation marks, as well as stopwords and unwanted characters. We will also perform a stemming on the remaining words.</p>

# In[6]:


for i in range(len(df)):

    file = open(file_path + "/doc_" + str(i) + ".tsv", "r",
                encoding = "utf-8")                                 # Only read 
    entry = file.read()                                             # Storing the information
    file.close()                                                    # Closing the file
    
    entry = entry.split("\t")                                       # Separating with tabs

    description = entry[4]                                          # Accessing the description info
    description = text_formatting(description)                      # Sending description info to the function
    add_voc(description)                                            # Sending description info to the Vocabulary
    entry[4] = description                                          # Receiving the description info modified

    title = entry[7]                                                # Accessing the title info
    title = text_formatting(title)                                  # Sending title info to the function
    add_voc(title)                                                  # Sending title info to the Vocabulary
    entry[7] = title                                                # Receiving the title info modified

    entry = "	".join(entry)                                       # Joining the info to put it back in file

    file = open(file_path + "/doc_" + str(i) + ".tsv", "w", 
                encoding = "utf-8")                                 # Modifying the file with the processed data
    file.write(entry)                                               # Writing in the file
    file.close()                                                    # Closing the file
    


# <H3> Build vocabulary </H3> 

# It's now time to build the vocabulary that we are going to use in order to assign a unique ***term_id*** to each word. We create it by simply putting each word in a line of a ***.txt*** file, in alphabetical order. The term_id of the word will be its index in the list (i.e. its line in the file)

# In[7]:


# The vocabulary variable was created in the in the initialization section (gvf.py)
# Here we are sorting the list and deleting duplicate words
vocabulary = list(sorted(set(vocabulary)))  
vocabulary.remove("")


# In[8]:


# In this step we will go through the vocabulary to assign an index to each word.
# At the same time we put them in lower case.

file = open("vocabulary.txt", "w", encoding = "utf-8")            

for i in range(len(vocabulary)):            
    file.write(vocabulary[i].lower() + "\n")
    
file.close()


# In[9]:


# For completeness, we now read back the vocabulary from the file

vocabulary_file = open("vocabulary.txt", "r", encoding = "utf-8")
vocabulary = vocabulary_file.read()
vocabulary = vocabulary.split("\n")
vocabulary_file.close()


# <H3>Dictionary Creation</H3>
# <p> The next step is to create a dictionary (an <b>inverted index</b>), which we will simply call <b>dictionary</b> and that will be of the form:</p>
#     
# <code> {
# term_id_1:[document_1, document_2, document_4],
# term_id_2:[document_1, document_3, document_5, document_6],
#     ...}
# </code>
# 
# <p>
# It will allow us to know, for each term_id, which are the documents that contain the corresponding word.
# </p>

# In[10]:


# For every document

for doc_index in range(len(df)):

    # Open .tsv file.
    file = open(file_path + "/doc_" + str(doc_index) + ".tsv", "r", encoding = "utf-8")
    
    # Read entry
    entry = file.read()
    entry = entry.split("\t")
    
    # Get description and title
    description = entry[4]
    title = entry[7]
    
    # Merge in a single string variable the title and the description, and get the set of words
    des_tit = description + " " + title    
    des_tit = set(des_tit.split(" "))
    
    # For every word in the description and title
    for word in des_tit:
            
        # Get the word index in the vocabulary
        term_id = vocabulary.index(word)

        # If the index it's not yet in the dictionary (the inverted index)
        if (term_id not in dictionary):
            dictionary[term_id] = [doc_index]               # Add the term_id and doc index to the dictionary

        elif (term_id in dictionary):                       # Else, if it's already in the dictionary
            dictionary[term_id].append(doc_index)           # Append the doc index to the dictionary
            
    file.close()


# <p> In order to avoid computing the inverted index every time we have to execute a query, we can store it in an external <b>.txt</b> file, and simply read it whenever we need it. </p>

# In[11]:


# Write the inverted index to a file, in order to avoid computing it every time.

file = open("inverted_index.txt", "w")

for key in dictionary:
    file.write(str(key) + ": " + str(dictionary[key]) + "\n")
    
file.close()


# In[12]:


# For completeness, read back the inverted index from the file.

file = open("inverted_index.txt", "r")

dictionary = dict()
txt = file.read().split("\n")

for i in range(len(txt)-1):
    line = txt[i].replace(":", "").replace("[", "").replace("]", "").replace("(", "").replace(")", "").replace(",", "").split(" ")
    dictionary[int(line[0])] = []
    for j in range(1, len(line)):
            dictionary[int(line[0])].append(int(line[j]))
            
file.close()


# <H2> 3) Search engine </H2>
# 
# <p> It's now time to create our search engines. The first one will simply look for documents (i.e. Airbnb results) that contain all the words in the user query. In the second one, instead, we will implement a ranking system, computing the <b>TFIDF</b> for each word in each document, and then calculating the <b>cosine similarity</b> between the query vector and each one of the vectors corresponding to the documents.</p>

# <H3>3.1) Conjunctive query</H3>
# <p>We are now going to read the user query as input. Since we are dealing with conjunctive queries (AND), each of the returned documents should contain all the words in the query. </p>

# In[14]:


# Get input query

query = str(input())


# In[15]:


# Format the query

query = text_formatting(query)
query = query.split(" ")


# <p> 
#     In this code we implement the query search, using two list, one for the word's index and other the document's index. We then find and return the intersection of the lists of documents corresponding to each term_id
# </p>

# In[16]:


vocabulary_file = open("vocabulary.txt", "r", 
                       encoding = "utf-8")                  # Open the vocabulary file
vocabulary = vocabulary_file.read()                         # Reading 
vocabulary = vocabulary.split("\n")                         # Splitting each word in a new Row

term_id_list = list()                                       # Creating a list for words-indexes
doc_list = list()                                           # Creating a list of matching documents

# For every word in the query

for word in query:                                 # Searching for each word in the query into the vocabulary  
    
    
    # If the word is in the vocabulary
    if word in vocabulary:
        
        # Get the vocabulary index for the word
        term_id = vocabulary.index(word)
        
        # Append it to the list of term_ids
        term_id_list.append(term_id)
    
    # Otherwise, empty the list (since we want only documents that contain all of the keywords,
    # if a word isn't in any document, we don't want to get any results)
    else:
        term_id_list = list()
        break
                
vocabulary_file.close()

# Get list of documetns containing the words (term_ids) just found
for term_id in term_id_list:
    if term_id in dictionary:
        doc_list.append(dictionary[term_id])


# In[17]:


# Compute the intersection of the sets of documents corresponding to each term_id we have found

if len(doc_list) > 0:
    selected_docs = set(doc_list[0])

    for l in doc_list:
        selected_docs = selected_docs.intersection(set(l))
else:
    selected_docs = list()
    
selected_docs = list(selected_docs)


# In[18]:


# Print the results in the required format

df_results = df.iloc[selected_docs].drop(labels = ["Unnamed: 0", 
                                      "average_rate_per_night", 
                                      "bedrooms_count", 
                                      "date_of_listing", 
                                      "latitude", 
                                      "longitude"], axis = 1)

df_results.index = list(range(1, len(df_results)+1))
df_results


# <H3> 3.2) Conjunctive query & Ranking score </H3> 

# We are now going to implement a scoring system. The general principle is to compute the distance between the query and each one of the documents. This is done by creating a vector for each one of them. Each component of the vectors corresponds to a word. In the query vector, if the word is contained, the component will be $1$, otherwise it will be $0$. In the documents vectors, the components are the $TFIDF = TF \cdot IDF$ of each word in the given document.</p>
# 
# <p>The <b>TF</b> (term frequency) is defined as:</p> 
# 
# <p>$TF = \frac{\textrm{n° of occurences of word in doc}}{\textrm{tot. number of words in doc}}$</p>
# 
# <p>while the <b>IDF</b> (inverse document frequency) is:</p>
# 
# <p>$IDF = log_{10}\left(\frac{\textrm{tot. n° of docs}}{\textrm{n° of docs containing word}}\right)$</p>

# In[19]:


vocabulary_file = open("vocabulary.txt", "r", 
                       encoding = "utf-8")                     # Open the vocabulary file
vocabulary = vocabulary_file.read()                            # Reading 
vocabulary = vocabulary.split("\n")

  
# For every document
for doc_index in range(len(df)):

    # Open file
    file = open(file_path + "/doc_" + str(doc_index) + ".tsv", "r", 
                encoding = "utf-8")

    # Read entry
    entry = file.read()
    entry = entry.split("\t")

    # Get description and title
    description = entry[4]
    title = entry[7]

    # Merge in a single string variable the title and the description
    des_tit = description + " " + title
    des_tit = des_tit.split(" ")

    # Compute TFIDF
    
    # Define counter (from collections)
    counter = Counter(des_tit)
    
    # For every word in the description and title
    for word in set(des_tit):
        
        # Compute the term frequency in the given document
        tf = (counter[word]/len(des_tit))
        
        # Get the term_id from the vocabulary
        term_id = vocabulary.index(word)
    
        # Compute the inverse document frequency
        N = len(df)  
        n = len(dictionary[term_id])
        idf = math.log10(N/n)
    
        # Compute the TFIDF
        tfidf = tf*idf

        # We now build the dictionary2 (second inverted index)
        
        # If the word is not yet in the dictionary
        if (term_id not in dictionary2):
            dictionary2[term_id] = [(doc_index, tfidf)]      # Add the term_id and (doc index, tfidf) tuple to the dictionary
        # Else, if it's already in the dictionary
        elif (term_id in dictionary2):                       
            dictionary2[term_id].append((doc_index, tfidf))  # Append the (doc index, tfidf) tuple

    # Close file
    file.close()
        
# Close vocabulary
vocabulary_file.close()


# <p> We are now going to compute a second dictionary (a second <b>inverted index</b>) called <b>dictionary2</b>, in the form:</p>
# 
# <code>{
# term_id_1:[(document1, tfIdf_{term,document1}), (document2, tfIdf_{term,document2}), (document4, tfIdf_{term,document4}), ...],
# term_id_2:[(document1, tfIdf_{term,document1}), (document3, tfIdf_{term,document3}), (document5, tfIdf_{term,document5}), (document6, tfIdf_{term,document6}), ...],
# ...}
# </code>
# 
# <p> Once again, we will store it in an external <b>.txt</b> file, for future usage</p>

# In[20]:


# Once again, it's convenient to store the inverted index in a file

file = open("inverted_index2.txt", "w")

for key in dictionary2:
    file.write(str(key) + ": " + str(dictionary2[key]) + "\n")
    
file.close()


# In[21]:


# For completeness we read the dictionary 2 back from the file just created

file = open("inverted_index2.txt", "r")

dictionary2 = dict()
txt = file.read().split("\n")

for i in range(len(txt)-1):
    line = txt[i].replace(":", "").replace("[", "").replace("]", "").replace("(", "").replace(")", "").replace(",", "").split(" ")
    dictionary2[int(line[0])] = []
    for j in range(1, len(line)):
        if j%2 == 1:
            dictionary2[int(line[0])].append((int(line[j]), float(line[j+1])))
            
file.close()


# <p> The next step is to fill the <b>query vector</b> components with <b>1</b> if the word corresponding to the component is present in the query, or <b>0</b> if it's not present. We will also fill the <b>documents vectors</b> components with the <b>TFIDF</b> of each term. Then we will compute the similarity between the query vector and each one of the document vectors as the cosine:</p>
# 
# $ cos(\alpha) = (\vec{q} \cdot \vec{d})/(|{\vec{q}}| \cdot |{\vec{d}}|)$
# 
# <p> of the angle $\alpha$ between the two vectors </p>

# In[22]:


vocabulary_file = open("vocabulary.txt", "r", 
                       encoding = "utf-8")          # Open the vocabulary file
vocabulary = vocabulary_file.read()                 # Reading 
vocabulary = vocabulary.split("\n")

# Create a vector for the query

v_query = [0]*(len(dictionary)+1)      # Initialize the components to 0


# Create a matrxi containing the vectors for the documents
v_docs = list()

for doc_index in range(len(df)):       # Initialize the components to 0
    v_docs.append([])
    for j in range(len(dictionary)+1):
        v_docs[doc_index].append(0)


# Fill the query vector with 1 (if a word is present) or leave it to 0 (if a word is not present)        
for word in query:
    if word in vocabulary:
        term_id = vocabulary.index(word)
        v_query[int(term_id)] = 1

# Fill the documents vectors with the TFIDFs for the words        
for term_id in dictionary2:
    for tpl in dictionary2[term_id]:
        doc_index = int(tpl[0])
        tfidf = float(tpl[1])
        v_docs[doc_index][int(term_id)] = tfidf

# Close vocabulary file
vocabulary_file.close()


# We can also store the cosines (i.e. our similarity scores) in a heap structure, in order to make the following sorting more efficient, from a computational point of view, than it would be by using a simple list.

# In[23]:


# Convert query vector to numpy array
a_query = np.array(v_query)

# Create a heap of scores and a dictionary of scores
heap = list()
heapq.heapify(heap)
scores_dictionary = dict()

# For every document
for doc_index in range(len(df)):
    
    # Convert document vector to numpy array
    a_doc = np.array(v_docs[doc_index])
    
    # Compute the cosine of the angle between the query vector and the document vector
    cos = np.dot(a_query, a_doc)/(np.linalg.norm(a_query)*np.linalg.norm(a_doc))
    
    # Put the result in the dictionary
    scores_dictionary[doc_index] = cos
    
    # Update the heap
    heapq.heappush(heap, cos)


# We can now retrieve the ***top k*** (in this case, we chose $k = 10$) scores, and associate them to the corresponding documents, in order to show the correct ranking in the final output.

# In[24]:


# Set the number k of top documents
k = 10

# Get the ordered list of top_k scores from the heap
top_k = heapq.nlargest(k, heap)


# In[25]:


# Initialize list of top documents (corresponding to top scores)
top_k_docs = list()

# Fill list of top documents
for i in range(len(top_k)):
    doc_index = list(scores_dictionary.keys())[list(scores_dictionary.values()).index(top_k[i])]
    top_k_docs.append(doc_index)
    del scores_dictionary[doc_index]


# In[28]:


# Print results

df_results = df.iloc[top_k_docs].drop(labels = ["Unnamed: 0", 
                                           "average_rate_per_night", 
                                           "bedrooms_count", 
                                           "date_of_listing", 
                                           "latitude", 
                                           "longitude"], axis = 1)
df_results.index = list(range(1, k+1))

df_results["similarity"] = [round(x,2) for x in top_k]
df_results


# 
# <H2>Define a new score!</H2>
# <p>The general idea is to create a <b>scoring function</b> and assign a score to each document. The scoring function we chose to use is a weighted sum of three scores, associated to <b>distance</b>, <b>number of bedrooms</b>, and <b>price</b> (average rate per night).</p>  
# 
# <p>The weights have been estimated heuristically, assuming that the distance has a greater impact on the quality of the results than any other parameter, while the contribution of the number of bedrooms and price is almost identical. We therefore chose to assign a weight $w_d = 0.6$ to the distance, $w_b = 0.2$ to the number of bedrooms and $w_p = 0.2$ to the price.</p>
# 
# <p>The scoring function for the single variables are a <b>negative exponential</b> $y = e^{-x/10}$ for the distance (taking the value <b>1</b>, maximum score, when the distance is equal zero, and so the city is exactly the one we were searching for), and a <b>gaussian</b> $y = e^{-(x-x_{ex})^2}$ for the number of bedrooms and the price (which have a peak value <b>1</b> corresponding to the exact number of bedrooms or th exact price, while the score decreases if the variables are too high or too low with respect to the exact values searched by the user).</p>
# 
# <p>The final scoring function is thus:<br>
# $y_{score} = w_d \cdot e^{-d/10} + w_b \cdot e^{-(n-n_{ex})^2} + w_p \cdot e^{-(p-p_{ex})^2}$
# </p>
# 
# <p>where $d$ is the distance, $n_{ex}$ the exact number of bedrooms the user searched for, and $p_{ex}$ the exact price.</p>

# In[26]:


# Get the input query

query = str(input())

# Get a copy of the query, which will undergo to a lighter form of text formatting
# as we don't want the city names to be stemmed
query_copy = query.lower()
for char in string.punctuation.replace("$", ""):
        query_copy = query_copy.replace(char, '')


# In[27]:


# Perform a full text formatting on the original query

query = text_formatting(query)
query = query.split(" ")
query_copy = query_copy.split(" ")


# In[28]:


# Initialize dictionary3, and dictionary of cities

dictionary3 = dict()
cities_dictionary = dict()


# In[29]:


# For every document
for doc_index in range(len(df)):

    # Open .tsv file.
    file = open(file_path + "/doc_" + str(doc_index) + ".tsv", "r", encoding = "utf-8")
    
    # Read entry
    entry = file.read()
    entry = entry.split("\t")
    
    # Get relevant information
    average_price_per_night = entry[0]
    bedrooms_count = entry[1]
    city = entry[2]
    latitude = entry[5]
    longitude = entry[6]
    # The score of the document and the distance of the city inside it are initially set to 0
    score = 0
    distance = 0
    
    # Update dictionary3
    dictionary3[doc_index] = [average_price_per_night, 
                              bedrooms_count, 
                              city, 
                              latitude, 
                              longitude, 
                              distance,
                              score]
    
    # Update cities_dictionary, with the city in the given document, and the corresponding coordinates
    city = city.lower()
    if city not in cities_dictionary:
        cities_dictionary[city] = (float(latitude), float(longitude))
    
    # Close .tsv file
    file.close()


# In[30]:


# We now define and initialize the variables corresponding to the relevant information we are going to 
# search in the query

q_price = 0
q_beds = 0
q_coord = (float("nan"),float("nan"))

# For every word in the query
for i in range(len(query)):
    
    # If it contains a $ symbol, we get the price the user is searching for
    if "$" in query[i]:
        q_price = float(query[i].replace("$", ""))
        
    # If it's the word "bedroom", then we assume that the previous word is the number of bedrooms the user wants
    # So we store it in a variable
    if query[i] == "bedroom":
        q_beds = int(query[i-1])
        
# For every city in the cities_dictionary        
for city in cities_dictionary:
    
    # Split it using whitespaces
    city = city.split(" ")
    
    # If the name of the city has only one word
    if len(city) == 1:
        # Search for a word in the query that corresponds to the city name
        for word in query_copy:
            if word == city[0]:
                city = city[0]
                # Get the coordinates of the city the user is searching for
                q_coord = cities_dictionary[city]
        
    # If the name of the city has only one word
    else:
        # Search like in the previous case, but checking on both the words
        for i in range(len(query_copy)):
            if (query_copy[i] == city[0]) and (query_copy[i+1] == city[1]):
                city = city[0] + " " + city[1]
                q_coord = cities_dictionary[city]


# Our choice for the weights is based on a heuristic reasoning, refined through successive trial and error. In principle, it seems reasonable to give the distance (between the city searched by the user, and the cities in the results) a higher contribution to the final score, than the other two weights. Indeed, for example, a user might easily be willing to pay a little more in order to get a house as close as possible to the desired destination/location, so the distance should definitely have the upper hand over the price.

# In[31]:


# Initalize the weights for the city distance, the number of bedrooms and the price

w_city = 0.6
    
if q_beds != 0:
    w_beds = 0.2
else:
    w_beds = 0
    
if q_price != 0:
    w_price = 0.2
else:
    w_price = 0
    
# Initialize heap
heap = list()
heapq.heapify(heap)
            
# For every document
for doc_index in range(len(df)):
    
    # Get the price
    d_price = dictionary3[doc_index][0]
    d_price = d_price.replace("$", "")
    d_price = float(d_price)
    
    # Get the number of bedrooms
    try:
        d_beds = int(dictionary3[doc_index][1])
    except ValueError:
        d_beds = 0
        
    # Get the city
    d_city = dictionary3[doc_index][2]
    d_city = d_city.lower()
    d_coord = cities_dictionary[d_city]

    # Compute the distance between the city searched by the user, and the one in the document
    if not math.isnan(q_coord[0]) and not math.isnan(q_coord[1]) and not math.isnan(d_coord[0]) and not math.isnan(d_coord[1]):
        dist = geodesic(q_coord, d_coord).km
        # Compute the city score, using the distance
        s_city = math.exp(-(dist/10))
    else:
        dist = float("nan")
        s_city = 0
        
    # Compute the scores for the number of bedrooms and price in the document
    s_beds = math.exp(-(d_beds - q_beds)**2)
    s_price = math.exp(-(d_price - q_price)**2)
    
    # Compute the final score (from 0 to 1) for the document as the weighted sum of the single scores
    score = w_city*s_city + w_beds*s_beds + w_price*s_price
    
    # Update the dictionary3
    # Since we now have used all the informations for the document and we don't need them anymore
    # we just replace them with the score of the document
    
    dictionary3[doc_index] = score
    
    # Update the heap
    heapq.heappush(heap, score)


# In[32]:


# Set the number k of top documents
k = 10

# Get the ordered list of top_k scores from the heap
top_k = heapq.nlargest(k, heap)


# In[33]:


# Initalize list of top documents
top_k_docs = list()

# Replace scores in the heap with corresponding documents
for i in range(len(top_k)):
    doc_index = list(dictionary3.keys())[list(dictionary3.values()).index(top_k[i])]
    top_k_docs.append(doc_index)
    del dictionary3[doc_index]


# In[34]:


# Print results
df_results = df.iloc[top_k_docs].drop(labels = ["Unnamed: 0", 
                                           "average_rate_per_night", 
                                           "bedrooms_count", 
                                           "date_of_listing", 
                                           "latitude", 
                                           "longitude"], axis = 1)
df_results.index = list(range(1, k+1))
df_results.index.name = 'ranking'
df_results


# <H2>Quick execution</H2>
# 
# This section allows the user to directly execute queries with the three search engines without going through all the file creation (once the files have already been created)

# In[1]:


file_path = "/home/jagg/Data/airbnb_data"

from gvf import *

df = df.drop_duplicates(subset=["title", "description"], keep = False)
df.index = list(range(len(df)))

file = open("inverted_index.txt", "r")

dictionary = dict()
txt = file.read().split("\n")

for i in range(len(txt)-1):
    line = txt[i].replace(":", "").replace("[", "").replace("]", "").replace("(", "").replace(")", "").replace(",", "").split(" ")
    dictionary[int(line[0])] = []
    for j in range(1, len(line)):
            dictionary[int(line[0])].append(int(line[j]))
            
file.close()


# <H3>3.1) Conjunctive query</H3>

# In[3]:


query = str(input())


# In[4]:


query = text_formatting(query)
query = query.split(" ")

vocabulary_file = open("vocabulary.txt", "r", 
                       encoding = "utf-8")                     
vocabulary = vocabulary_file.read()                            
vocabulary = vocabulary.split("\n")                            

term_id_list = list()                                          
doc_list = list()                                              

for word in query:                                             
    
    if word in vocabulary:
        
        
        term_id = vocabulary.index(word)
        term_id_list.append(term_id)
    
    else:
        term_id_list = list()
        break
                
vocabulary_file.close()

for term_id in term_id_list:
    if term_id in dictionary:
        doc_list.append(dictionary[term_id])
        
if len(doc_list) > 0:
    
    selected_docs = set(doc_list[0])

    for l in doc_list:
        selected_docs = selected_docs.intersection(set(l))
else:
    selected_docs = list()
    
selected_docs = list(selected_docs)

df_results = df.iloc[selected_docs].drop(labels = ["Unnamed: 0", 
                                      "average_rate_per_night", 
                                      "bedrooms_count", 
                                      "date_of_listing", 
                                      "latitude", 
                                      "longitude"], axis = 1)

df_results.index = list(range(1, len(df_results)+1))
df_results


# <H3> 3.2) Conjunctive query & Ranking score </H3> 

# In[5]:


file = open("inverted_index2.txt", "r")

dictionary2 = dict()
txt = file.read().split("\n")

for i in range(len(txt)-1):
    line = txt[i].replace(":", "").replace("[", "").replace("]", "").replace("(", "").replace(")", "").replace(",", "").split(" ")
    dictionary2[int(line[0])] = []
    for j in range(1, len(line)):
        if j%2 == 1:
            dictionary2[int(line[0])].append((int(line[j]), float(line[j+1])))
            
file.close()

vocabulary_file = open("vocabulary.txt", "r", 
                       encoding = "utf-8")                     
vocabulary = vocabulary_file.read()                            
vocabulary = vocabulary.split("\n")

v_query = [0]*(len(dictionary)+1)
v_docs = list()

for doc_index in range(len(df)):
    v_docs.append([])
    for j in range(len(dictionary)+1):
        v_docs[doc_index].append(0)


for word in query:
    if word in vocabulary:
        term_id = vocabulary.index(word)
        v_query[int(term_id)] = 1

for term_id in dictionary2:
    for tpl in dictionary2[term_id]:
        doc_index = int(tpl[0])
        tfidf = float(tpl[1])
        v_docs[doc_index][int(term_id)] = tfidf

vocabulary_file.close()

a_query = np.array(v_query)

heap = list()
heapq.heapify(heap)
scores_dictionary = dict()


for doc_index in range(len(df)):
    
    a_doc = np.array(v_docs[doc_index])
    
    cos = np.dot(a_query, a_doc)/(np.linalg.norm(a_query)*np.linalg.norm(a_doc))
    
    scores_dictionary[doc_index] = cos
    heapq.heappush(heap, cos)
    
k = 10

top_k = heapq.nlargest(k, heap)

top_k_docs = list()

for i in range(len(top_k)):
    doc_index = list(scores_dictionary.keys())[list(scores_dictionary.values()).index(top_k[i])]
    top_k_docs.append(doc_index)
    del scores_dictionary[doc_index]
    
df_results = df.iloc[top_k_docs].drop(labels = ["Unnamed: 0", 
                                           "average_rate_per_night", 
                                           "bedrooms_count", 
                                           "date_of_listing", 
                                           "latitude", 
                                           "longitude"], axis = 1)
df_results.index = list(range(1, k+1))

df_results["similarity"] = [round(x,2) for x in top_k]
df_results


# <H3> Define a new score! </H3>

# In[9]:


query = str(input())
query_copy = query.lower()
for char in string.punctuation.replace("$", ""):
        query_copy = query_copy.replace(char, '')
        
query = text_formatting(query)
query = query.split(" ")
query_copy = query_copy.split(" ")


# In[10]:


dictionary3 = dict()
cities_dictionary = dict()

for doc_index in range(len(df)):

    file = open(file_path + "/doc_" + str(doc_index) + ".tsv", "r", encoding = "utf-8")
    
    entry = file.read()
    entry = entry.split("\t")
    
    average_price_per_night = entry[0]
    bedrooms_count = entry[1]
    city = entry[2]
    latitude = entry[5]
    longitude = entry[6]
    score = 0
    distance = 0
    
    dictionary3[doc_index] = [average_price_per_night, 
                              bedrooms_count, 
                              city, 
                              latitude, 
                              longitude, 
                              distance,
                              score]
    
    city = city.lower()
    if city not in cities_dictionary:
        cities_dictionary[city] = (float(latitude), float(longitude))
    
    file.close()
    
q_price = 0
q_beds = 0
q_coord = (float("nan"),float("nan"))

for i in range(len(query)):
    
    if "$" in query[i]:
        q_price = float(query[i].replace("$", ""))
        
    if query[i] == "bedroom":
        q_beds = int(query[i-1])
            
for city in cities_dictionary:
    
    city = city.split(" ")
    
    if len(city) == 1:
        for word in query_copy:
            if word == city[0]:
                city = city[0]            
                q_coord = cities_dictionary[city]
        
    else:
        for i in range(len(query_copy)):
            if (query_copy[i] == city[0]) and (query_copy[i+1] == city[1]):
                city = city[0] + " " + city[1]
                q_coord = cities_dictionary[city]
                
w_city = 0.6
    
if q_beds != 0:
    w_beds = 0.2
else:
    w_beds = 0
    
if q_price != 0:
    w_price = 0.2
else:
    w_price = 0
    
heap = list()
heapq.heapify(heap)
            
for doc_index in range(len(df)):
    
    d_price = dictionary3[doc_index][0]
    d_price = d_price.replace("$", "")
    d_price = float(d_price)
    
    try:
        d_beds = int(dictionary3[doc_index][1])
    except ValueError:
        d_beds = 0
        
    d_city = dictionary3[doc_index][2]
    d_city = d_city.lower()
    d_coord = cities_dictionary[d_city]

    if not math.isnan(q_coord[0]) and not math.isnan(q_coord[1]) and not math.isnan(d_coord[0]) and not math.isnan(d_coord[1]):
        dist = geodesic(q_coord, d_coord).km
        s_city = math.exp(-(dist/10))
    else:
        dist = float("nan")
        s_city = 0
        
    s_beds = math.exp(-(d_beds - q_beds)**2)
    s_price = math.exp(-(d_price - q_price)**2)
    
    score = w_city*s_city + w_beds*s_beds + w_price*s_price
    
    dictionary3[doc_index][5] = dist
    dictionary3[doc_index][6] = score
    
    dictionary3[doc_index] = score
    
    heapq.heappush(heap, score)
    
k = 10

top_k = heapq.nlargest(k, heap)

top_k_docs = list()

for i in range(len(top_k)):
    doc_index = list(dictionary3.keys())[list(dictionary3.values()).index(top_k[i])]
    top_k_docs.append(doc_index)
    del dictionary3[doc_index]
    
df_results = df.iloc[top_k_docs].drop(labels = ["Unnamed: 0", 
                                           "average_rate_per_night", 
                                           "bedrooms_count", 
                                           "date_of_listing", 
                                           "latitude", 
                                           "longitude"], axis = 1)
df_results.index = list(range(1, k+1))
df_results.index.name = 'ranking'
df_results


# <H2> Make a nice visualization! </H2>

# In[1]:


file_path = "/home/jagg/Data/airbnb_data"

from gvf import *

df = df.drop_duplicates(subset=["title", "description"], keep = False)
df.index = list(range(len(df)))


# In[2]:


coord_input = input()
coord_input = coord_input.split(" ")
init_lat = float(coord_input[0])
init_lon = float(coord_input[1])
km = float(input())


# In[3]:


coord = ("Lat:" + str(init_lat) + ", "+"Lon:" + str(init_lon))
coord, km


# In[4]:


distance = list()

for i in range(0, len(df)):
    
    if not math.isnan(df["latitude"][i]) and not math.isnan(df["longitude"][i]):
        distance.append(geodesic( (df["latitude"][i], df["longitude"][i]), (init_lat, init_lon) ).km)
    else:
        distance.append(float("nan"))
    
df["distance"] = distance
df


# In[5]:


df = df[df["distance"] < km]
df_results = df
df_results["distance"] = df_results["distance"].round(decimals = 2)
df_results


# In[6]:


m = folium.Map(location = [init_lat, init_lon], 
               width='100%', 
               height='100%', 
               left='0%', 
               top='0%', 
               position='relative',
               tiles='OpenStreetMap',
               API_key=None, 
               max_zoom=18, 
               min_zoom=6, 
               max_native_zoom=None, 
               zoom_start=10, 
               attr='ADM Group 23',
               min_lat=26, 
               max_lat=38,
               min_lon=-90, 
               max_lon=-108, 
               max_bounds=True, 
               crs='EPSG3857', 
               control_scale=True,              
               zoom_control=True
              )

# Second Raster layer from google maps - Terrain view
folium.raster_layers.TileLayer(
    tiles = 'http://{s}.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
    attr = 'Google satellite - ADM Group 23',
    name = 'Google Maps Terrain',
    max_zoom = 20,
    subdomains = ['mt0', 'mt1', 'mt2', 'mt3'],
    overlay = False,
    control = True,
).add_to(m)


# In[7]:


# Plugins for full screen view
plugins.Fullscreen(
    position='topleft',
    title='View Map in full screen',
    title_cancel='Exit',
    force_separate_button=True).add_to(m)


# In[8]:


# Circle
feature_group1 = folium.FeatureGroup("Airbnb query")

feature_group1.add_child(folium.Circle(location = [init_lat, init_lon],
                                       radius = km * 1000,
                                       popup = ('Search Radio: ' + str(km) + ' ' + 'Kms'),
                                       color = '#3050bb',
                                       fill = True,
                                       fill_color = '#3050bb'
                                       )
                         )
# Marker Start point
feature_group1.add_child(folium.Marker(location=[init_lat, init_lon],
                                       popup=coord,
                                       icon=folium.Icon(color = 'red')
                                       )
                        )


# In[9]:


# Markers Airbnb found
feature_group2 = folium.FeatureGroup("Airbnb availability")


# Creating the objects that will be on the map (Now from the Data Frame) 
for city, latitude, longitude in zip(df["city"], df["latitude"], df["longitude"]):
    feature_group2.add_child(folium.Marker(location=[latitude, longitude],
                                           popup=city,
                                           icon=folium.Icon(color='green', icon = 'university', prefix = 'fa')
                                          )
                            )    


# In[10]:


# Adds tools to the top right corner
m.add_child(MeasureControl())
m.add_child(feature_group1)
m.add_child(feature_group2)
m.add_child(folium.LatLngPopup())
m.add_child(folium.map.LayerControl(collapsed=True))


# In[11]:


m.save("map.html")

