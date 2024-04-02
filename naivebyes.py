import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load your dataset
def load_data(filename):
    data = pd.read_csv(filename)
    X = data.drop('label', axis=1)  # Assuming 'label' is the column containing the target labels
    y = data['label']
    return X, y

# Train Naive Bayes classifier
def train_naive_bayes(X_train, y_train):
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model

# Test the trained model
def test_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy

# Main function
def main():
    # Load your data
    X, y = load_data('testeer.csv')

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Naive Bayes classifier
    nb_model = train_naive_bayes(X_train, y_train)
    

    # Test the model
    accuracy = test_model(nb_model, X_test, y_test)
    print("Accuracy:", accuracy)



if __name__ == "__main__":
    main()
