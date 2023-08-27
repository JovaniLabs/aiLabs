import csv
import sys
import pandas as pd

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    
    # Load the CSV data into a pandas DataFrame
    df = pd.read_csv(filename)

    # Define the list of column names that should have integer data type
    int_cols = [
        "Administrative",
        "Informational",
        "ProductRelated",
        "OperatingSystems",
        "Browser",
        "Region",
        "TrafficType",
    ]
    # Convert these columns to integer data type
    df[int_cols] = df[int_cols].astype(int)

    # Define the list of column names that should have float data type
    float_cols = [
        "Administrative_Duration",
        "Informational_Duration",
        "ProductRelated_Duration",
        "BounceRates",
        "ExitRates",
        "PageValues",
        "SpecialDay",
    ]
    # Convert these columns to float data type
    df[float_cols] = df[float_cols].astype(float)

    # Define the month abbreviations in order
    month_abbr = (
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "June",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    )
    # Convert the 'Month' column to integer data type by mapping the month abbreviations to their respective index
    df["Month"] = df["Month"].map({m: i for i, m in enumerate(month_abbr)}).astype(int)

    # Convert the 'VisitorType' column to integer data type
    # Map 'Returning_Visitor' to 1, and any other value (i.e., 'New_Visitor' and 'Other') to 0
    df["VisitorType"] = (
        df["VisitorType"].map({"Returning_Visitor": 1}).fillna(0).astype(int)
    )

    # Convert the 'Weekend' column to integer data type
    df["Weekend"] = df["Weekend"].astype(int)

    # Convert the 'Revenue' column to integer data type
    df["Revenue"] = df["Revenue"].astype(int)

    # Assign the 'Revenue' column as our labels
    labels = df["Revenue"].values.tolist()
    
    # Drop the 'Revenue' column from the DataFrame and use the rest as our evidence
    evidence = df.drop(columns="Revenue").values.tolist()

    return evidence, labels

def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    # Initialize a K-Nearest Neighbors (KNN) classifier
    # The number of neighbors to use (n_neighbors) is set to 1
    model = KNeighborsClassifier(n_neighbors=1)
    
    # Fit the model using the provided evidence and labels
    # This trains the model based on the input data
    model.fit(evidence, labels)
    
    # Return the trained model
    return model

def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    
    # Create a confusion matrix from the true labels and the predicted labels
    # This matrix gives us a summary of how well our classifier is performing
    matrix = confusion_matrix(labels, predictions)

    # Compute sensitivity (also known as the true positive rate)
    # It's the proportion of actual positive cases (in our context, purchases made)
    # that the classifier identified correctly
    # In the confusion matrix, it's the value at the second row and second column (1,1)
    # divided by the sum of the values in the second row
    sensitivity = matrix[1, 1] / sum(matrix[1])

    # Compute specificity (also known as the true negative rate)
    # It's the proportion of actual negative cases (in our context, no purchases made)
    # that the classifier identified correctly
    # In the confusion matrix, it's the value at the first row and first column (0,0)
    # divided by the sum of the values in the first row
    specificity = matrix[0, 0] / sum(matrix[0])

    # Return both sensitivity and specificity
    return sensitivity, specificity

if __name__ == "__main__":
    main()