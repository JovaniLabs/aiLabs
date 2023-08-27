import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    
    #Initialize dictionary to hold the probability distribution.
    prop_dist = {}

        #calculate number of pages in corpus and number of links on current page.
    dict_len = len(corpus)
    pages_len = len(corpus[page])

    #if there are no links on current page.
    if pages_len < 1:
            
        #Probability of transitioning to any page is equally likely
        for key in corpus:
            prop_dist[key] = 1 / dict_len

    else:
        #Otherwise, calculate two factors:
        #random_factor is the probability of jumping to a page not linked from the current one.
        #even_factor is the probability of following a link on the current page.
        random_factor = (1-damping_factor) / dict_len
        even_factor = damping_factor / pages_len
    
        #each page in corpus.
        for key in corpus:
            #if page is not linked from current page.
            if key not in corpus[page]:
                #Probability of transitioning to it is just the 'random_factor'.
                prop_dist[key] = random_factor 
            else:
                #if the page is linked from current page.
                #Probability of transitioning to it is the sum of even_factor and random_factor.
                prop_dist[key] = even_factor + random_factor

    #return the probability dispersal. 
    return prop_dist
    

def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    
    #Initialize a dictionary to keep track of the number of visits to each page 
    #begin with 0 for all pages.
    samples_dict = corpus.copy()
    for i in samples_dict:
        samples_dict[i] = 0
    sample = None

    #Perform 'n' steps of the simulation.
    for _ in range(n):
        if sample:
            #If currently on a page
            #decide the next page to visit based on the transition model.
            dist = transition_model(corpus, sample, damping_factor)
            dist_lst = list(dist.keys())
            dist_weights = [dist[i] for i in dist]
            #choose next page based on the probabilities.
            sample = random.choices(dist_lst, dist_weights, k=1)[0]
        else:
            #If not currently on page i.e., at the begining
            #choose a page at random to start from.
            sample = random.choice(list(corpus.keys()))

        #Increment the count of visits to the current page.
        samples_dict[sample] += 1

    #divide the count of visits to each page by 'n' to get the coherence 
    # which is estimate of PageRank of each page. 
    for item in samples_dict:
        samples_dict[item] /= n

    #outputs the final result after all computations.
    return samples_dict


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    
    #calculate total number of pages in corpus.
    pages_number = len(corpus)
    #initialize dictionaries to hold and rank values of each page.
    old_dict = {page: 1 / pages_number for page in corpus}
    new_dict = {}

    #Repeat the iteration until convergence.
    while True:
        #for each page in corpus.
        for page in corpus:
            #initialize a variable to accumulate the rank values of linking pages.
            temp = 0
            #for each potential linking page in corpus
            for linking_page in corpus:
                #if the current page is linked from the linking page.
                if page in corpus[linking_page]:
                    #Add the rank value of the linking page divided by its number of links
                    temp += (old_dict[linking_page] / len(corpus[linking_page]))
                #if the linking page has no outgoing links.
                if len (corpus[linking_page]) == 0:
                    #treat it as if it links to all pages in the corpus and add its rank value divided by the total number of pages.
                    temp += (old_dict[linking_page]) / pages_number

            #apply the damping factor and add the constant term to the accumulated rank value to get the new rank value.
            temp = damping_factor * temp + (1 - damping_factor) / pages_number
            #store new rank value in new dictionary.
            new_dict[page] = temp

        #calculate maximum difference between the old and new rank values 
        difference = max(abs(new_dict[x] - old_dict[x]) for x in old_dict)
        #if maximum difference is below the threshold, the ranks have converged and we break the loop.
        if difference < 0.001:
            break
        else:
            #otherwise, update the old rank values to be the new ones and repeat the iteration.
            old_dict = new_dict.copy()

    #after convergence, return final rank values as result.
    return old_dict

if __name__ == "__main__":
    main()