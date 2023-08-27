import sys
import copy

from crossword import *


class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        w, h = draw.textsize(letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self):
        """
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word.)
        """
        
        # create a deep copy of the domain dictionary to prevent modifications durring iteration.
        domain_copy = copy.deepcopy(self.domains)
        
        # iterate over each variable in the domain copy.
        for variable in domain_copy:
            
            # store the length of the variable.
            variable_length = variable.length
            
            # iterate over each word in the domain copy of the variable
            for word in domain_copy[variable]:
                
                # check if the length of the word is not equal to the variable length
                if len (word) != variable_length:
                    
                    #if the lengths do not match, remove the word from the original domain.
                    try:
                        self.domains[variable].remove(word)
                    except ValueError:
                        #handle the exception when the word is not found in the list.
                        print(f"word: '{word}' not found in domain of variable.")
                    
                    
    def revise(self, x, y):
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        """
        
        #identify overlap points between variable x and y in crossword puzzle.
        xoverlap, yoverlap = self.crossword.overlaps[x, y]
        
        #initialize revision flag as False.
        revision_made = False
        
        #make a deep copy of the domains to prevent modifications during iteration.
        domains_copy = copy.deepcopy(self.domains)
        
        #if x and y overlap, check consistency for words in their domains.
        if xoverlap:
            
            #iterate over each word in the domain of x.
            for xword in domains_copy[x]:
                
                #flag to check if matching value is found in domain y.
                match_value = False
                
                #iterate over each word in the domain of y.
                for yword in self.domains[y]:
                    
                    #if the overlapping characters in both words match, set flag to True and break loop.
                    if xword[xoverlap] == yword[yoverlap]: 
                        match_value = True 
                        break
                    
                #if no match is found, remove word from domain of x and set revision flag to True.
                if not match_value:
                    self.domains[x].remove(xword)
                    revision_made = True
                    
        #return True if a revision was made, False otherwise.
        return revision_made

    def ac3(self, arcs=None):
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with initial list of all arcs in the problem.
        Otherwise, use `arcs` as the initial list of arcs to make consistent.

        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.
        """
        
        #initialize an empty queue if no specific arcs are provided.
        if not arcs:
            
            #create a queue of arcs to revise.
            queue = []
            
            #for each variable, add its arcs with neighbors to the queue.
            for variable1 in self.domains:
                for variable2 in self.crossword.neighbors(variable1):
                    if self.crossword.overlaps[variable1, variable2] is not None:
                        queue.append((variable1, variable2))
                        
        #process all arcs until queue is empty                 
        while queue:
            #get the first arcs in the queue
            x, y = queue.pop(0)
            
            #try to revise the arc x, y.
            if self.revise(x, y):
                
                #if domain of x is empty after revision, return False
                if len(self.domains[x]) == 0:
                    return False
                
                #add arcs z, x for all neighbors z of x in the queue except y
                for neighbour in self.crossword.neighbors(x):
                    if neighbour != y:
                        queue.append((neighbour, x))
                        
            #of all arcs processed and no inconsistency found return True.
            return True

    def assignment_complete(self, assignment):
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """
        
        #iterate over each variable in the domains
        for variable in self.domains:
            
            #if a variable is not assigneda value in the assignment, return False
            if variable not in assignment:
                return False
            
        #if all variables in the domains have been assigned in the assignment , return True.
        return True

    def consistent(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        """
        
        #convert the values in the assignment to a list
        words = [*assignment.values()]
        
        #if there are duplicate words in the assignment, return False
        if len(words) != len(set(words)):
            return False
        
        #check if the length of the assigned word matches the length of the variable
        for variable in assignment:
            if variable.length != len(assignment[variable]):
                return False
            
        #check if the assigned words don't conflict at the overlap position.
        for variable in assignment:
            for neighbour in self.crossword.neighbors(variable):
                if neighbour in assignment:
                    x, y = self.crossword.overlaps[variable, neighbour]
                    
                    #if the overlapping characters of the assigned words don't match return False.
                    if assignment[variable][x] != assignment[neighbour][y]:
                        return False
                    
        #if all assigned words are unique of correct length and don't conflict return True.
        return True

    def order_domain_values(self, var, assignment):
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """
        
        #initialize a dictionary to store words in the domain and
        #their count of ruled out values
        word_dict = {}
        
        #fetch the neighbours of the variable 'var'
        neighbours = self.crossword.neighbors(var)
        
        #iterate over each word in the domain of var
        for word in self.domains[var]:
            
            #initialize count of ruled out values for word
            eliminated = 0
            
            #iterate over each neighbour of var.
            for neighbour in neighbours:
                
                #if the neighbour is already in the assignment skip to next neighbour
                if neighbour in assignment:
                    continue
                else:
                    #fetch the overlapping positions of var and neighbour.
                    xoverlap, yoverlap = self.crossword.overlaps[var, neighbour]
                    
                    #iterate over each word in the domain of neighbour.
                    for neighbour_word in self.domains[neighbour]:
                        
                        #if overlapping characters of word and neighbour_word 
                        #dont match increment eliminated count
                        if word[xoverlap] != neighbour_word[yoverlap]:
                            eliminated += 1
                            
            #update the eliminated count for word in the dictionary
            word_dict[word] = eliminated
            
        #sort the dictionary by the eliminated count in ascending order
        sorted_dict = dict(sorted(word.dict.items(), key=lambda item: item[1]))
        
        #return the words in the sorted order
        return [*sorted_dict]

    def select_unassigned_variable(self, assignment):
        """
        Return an unassigned variable not already part of `assignment`.
        Choose the variable with the minimum number of remaining values
        in its domain. If there is a tie, choose the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.
        """
        
        #initialize a dictionary to store unassigned variables and their domains
        choice_dict = {}
        
        #iterate over each variable in the domains
        for variable in self.domains:
            
            #if the variable is not assigned, add it to the choice_dict
            if variable not in assignment:
                choice_dict[variable] = self.domains[variable]
                
        #sort the dictionary by the size of each variable's domain
        sorted_list = [v for v, k in sorted(choice_dict.items(), key = lambda item: len(item[1]))]
        
        #return the variable with the smallest domain
        return sorted_list[0]

    def backtrack(self, assignment):
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.

        `assignment` is a mapping from variables (keys) to words (values).

        If no assignment is possible, return None.
        """
        
        #if all variables are assigned return the assignement
        if len(assignment) == len(self.domains):
            return assignment
        
        #select an unassigned variable
        variable = self.select_unassigned_variable(assignment)
        
        #iterate over each value in the domain of the selected variable
        for value in self.domains[variable]:
            
            #copy the current assignment and assign the selected value to 
            # the variable in the copy
            assignment_copy = assignment.copy()
            assignment_copy[variable] = value
            
            #if the new assignment is consistent, recursively continue to assign the next variable
            if self.consistent(assignment_copy):
                result = self.backtrack(assignment_copy)
                
                #if a complete assignment is found return it 
                if result is not None:
                    return result
                
        #if no valid assignment is found after trying all values 
        #in domain return None
        return None
                    
def main():

    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)

if __name__ == "__main__":
    main()