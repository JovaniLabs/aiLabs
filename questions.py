import nltk
import sys
import os 
import string
import math
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import defaultdict, Counter


FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    
    # Initialize an empty dictionary to hold our files and their contents
    files = {}
    
    # Loop through each file in the specified directory
    for filename in os.listdir(directory):
        
        # If the file ends with ".txt", it's a text file
        if filename.endswith(".txt"):
            
            # Open the file in 'read' mode. The os.path.join() function is used to create the full file path
            with open(os.path.join(directory, filename), "r") as file:
                
                # Read the file's contents and add it to our dictionary
                # The key is the filename and the value is the file's contents
                files[filename] = file.read()
    
    # Return the dictionary containing our files and their contents
    return files


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    
    # The nltk function word_tokenize() breaks up the string into words and punctuation
    words = word_tokenize(document.lower())
    
    # List comprehension: compile a fresh list of items that
    # just the words that are not stopwords or punctuation string with a 
    # A pre-initialized string called "punctuation" contains all of the standard punctuation marks.
    # A list of frequently used English terms with little significance may be found by using the query 
    # stopwords.words("english").Prior to Natural Language Processing, 
    # and are frequently eliminated.
    words = [word for word in words if word not in string.punctuation and word not in stopwords.words("english")]
    
    # Return the list of words
    return words


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    
    # Create an empty dictionary to hold the IDF values for each word.
    idfs = dict()

    # Calculate the total number of documents
    num_documents = len(documents)

    # Use a set comprehension to get a list of all unique words across all documents
    words = set(word for sublist in documents.values() for word in sublist)
    
    # Loop through each unique word
    for word in words:
        # For each word, count the number of documents that contain the word
        f = sum(word in documents[title] for title in documents)
        
        # Calculate the IDF value for the word
        idf = math.log(num_documents / f)
        
        # Add the word and its IDF value to the idfs dictionary
        idfs[word] = idf
        
    # Return the idfs dictionary
    return idfs


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    
    # Create a defaultdict of type float, this will automatically assign 0.0 as default value for non-existent keys.
    tf_idfs = defaultdict(float)
    
    # Iterate through each word in the query
    for word in query:
        # If the word is present in the idfs dictionary (meaning it is in at least one of the documents)
        if word in idfs:
            # Then iterate through each file in the files dictionary
            for file in files:
                # Add the tf-idf score of the word for the current file to the total tf-idf score for this file.
                # The tf-idf score for a word in a file is calculated as the frequency of the word in the file times the idf of the word
                tf_idfs[file] += files[file].count(word) * idfs[word]
                
    # Sort the files by their tf-idf scores, in descending order
    sorted_files = sorted(tf_idfs.items(), key=lambda x: x[1], reverse=True)
    
    # Return the filenames of the top n files
    # Note that if there are fewer than n files, this will simply return all files
    return [file[0] for file in sorted_files[:n]]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    
    # Create an empty list to store sentences along with their associated IDF sum and query term density.
    sentence_values = []
    
    # Iterate over each sentence and its associated words
    for sentence, words in sentences.items():
        # Compute the sum of IDF values for each word in the query that also appears in the sentence
        idf_sum = sum(idfs[word] for word in query if word in words)
        
        # Compute the query term density which is the proportion of words in the sentence that are also in the query
        query_term_density = sum(word in query for word in words) / len(words)
        
        # Append a tuple containing the sentence, its IDF sum and its query term density to our list
        sentence_values.append((sentence, idf_sum, query_term_density))

    # Sort our list of tuples in descending order first by IDF sum and then by query term density
    sentence_values.sort(key=lambda x: (x[1], x[2]), reverse=True)
    
    # Return the 'n' top sentences. If there are fewer than 'n' sentences, it will return all sentences
    return [sentence[0] for sentence in sentence_values[:n]]

if __name__ == "__main__":
    main()
