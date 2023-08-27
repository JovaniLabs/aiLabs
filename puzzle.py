from logic import *

AKnight = Symbol("A is a Knight")
AKnave = Symbol("A is a Knave")

BKnight = Symbol("B is a Knight")
BKnave = Symbol("B is a Knave")

CKnight = Symbol("C is a Knight")
CKnave = Symbol("C is a Knave")

# Puzzle 0
# A says "I am both a knight and a knave."
knowledge0 = And(
# Because A can't be both, he must be a knave.
    # If and only if A is not a knave, he is a knight.
    Biconditional(AKnight, Not(AKnave)),
    # If and only if A's proposition is True, then A is a knight.
    Biconditional(AKnight, And(AKnight, AKnave))
)

# Puzzle 1
# A says "We are both knaves."
# B says nothing.
knowledge1 = And(
# A cannot claim that they are both knaves, therefore making A a knave.
# B said nothing, and A's claim was proven false; B is a knight.
    # If and only if A and B are not knaves, they are knights.
    Biconditional(AKnight, Not(AKnave)),
    Biconditional(BKnight, Not(BKnave)),
    # If and only if A's proposition is True, then A is a knight.
    Biconditional(AKnight, And(AKnave, BKnave))
)

# Puzzle 2
# A says "We are the same kind."
# B says "We are of different kinds."
knowledge2 = And(
# According to A's remark, both A and B are either knights or knaves.
# B's claim implies that A and B are polar opposites, with A being a knight and B being a knave, and vice versa.
    # If and only if A and B are not knaves, they are knights.
    Biconditional(AKnight, Not(AKnave)),
    Biconditional(BKnight, Not(BKnave)),
    # If and only if A's proposition is true, A is a knight.
    Biconditional(AKnight, Or(And(AKnave, BKnave), And(AKnight, BKnight))),
    # If and only if B's proposition is true, B is a knight.
    Biconditional(BKnight, Or(And(AKnave, BKnight), And(AKnight, BKnave)))

)

# Puzzle 3
# A says either "I am a knight." or "I am a knave.", but you don't know which.
# B says "A said 'I am a knave'."
# B says "C is a knave."
# C says "A is a knight."
knowledge3 = And(
# It is implausible for A to claim that they are a knave, hence declaring B to be a knave. As a result,
# B's second assertion is untrue, making C a knight, which confirms the fact that 
# A is indeed a knight 
    # If and only if A, B, and C are not knaves, they are knights.
    Biconditional(AKnight, Not(AKnave)),
    Biconditional(BKnight, Not(BKnave)),
    Biconditional(CKnight, Not(CKnave)),
    # If and only if either B's first or C's only assertion is true, A is a knight.
    # According to the criteria of this problem, A must be a knight because they cannot clearly state otherwise.
    # State that they are Knaves
    Or(AKnight, AKnave),
    # If and only if the first sentence of B is true, B is a knight.
    Biconditional(BKnight, Biconditional(AKnight, AKnave)),
    # If and only if its second sentence is true, B is a knight.
    Biconditional(BKnight, CKnave),
    # If and only if C's proposition is true, C is a knight.
    Biconditional(CKnight, AKnight)

)


def main():
    symbols = [AKnight, AKnave, BKnight, BKnave, CKnight, CKnave]
    puzzles = [
        ("Puzzle 0", knowledge0),
        ("Puzzle 1", knowledge1),
        ("Puzzle 2", knowledge2),
        ("Puzzle 3", knowledge3)
    ]
    for puzzle, knowledge in puzzles:
        print(puzzle)
        if len(knowledge.conjuncts) == 0:
            print("    Not yet implemented.")
        else:
            for symbol in symbols:
                if model_check(knowledge, symbol):
                    print(f"    {symbol}")


if __name__ == "__main__":
    main()
