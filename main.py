import csv, math

# Function imports a csv file and converts it to a useable type/format
def csv_file_import(filename):
    with open(filename) as imported_file:
        return list(csv.reader(imported_file))

# Function writes and exports a csv file based on the passed in predictions argument
def csv_file_export(filename, predictions):
    with open(filename, "w", newline = '') as exported_file:
        writer = csv.writer(exported_file)
        writer.writerows(predictions)

# Function imputes missing data by calculating what the most common value is for each column and assigning that value to any missing data
def impute(csv_data):
    # Loop through all columns excluding the Localization column (target)
    for i in range(len(csv_data[0]) - 1):
        value_counts = {}
        mode = None
        highest_count = -1

        # Loop through each row in the column
        for row in csv_data:
            value = row[i]
            
            # Increment count of value type if value is not missing
            if value != '?':
                if value in value_counts:
                    value_counts[value] += 1
                else:
                    value_counts[value] = 1

        # Calculate the highest frequency value for the column
        for value, count in value_counts.items():
            if count > highest_count:
                highest_count = count
                mode = value

        # Replace many missing data with the mode value for that specific column
        for row in csv_data:
            if row[i] == '?':
                row[i] = mode

    return csv_data

# Function merges our test data csv file and our keys txt file based upon matching the GeneIDs
def merge_csv_data(test_csv_data, keys_data):
    # Loop through each row in the csv data file
    for row in test_csv_data:
        # Set GeneID to be the column data that we want to use to match and merge
        gene_id = row[0]
        
        # If there is a ID match replace the Localization value with the Localization value from keys
        if gene_id in keys_data:
            row[-1] = keys_data[gene_id]
    
    return test_csv_data

# Function uses training data to train the naive baye model
def train_naive_baye(csv_data):
    class_frequency = {}
    prior_probabilities = {}
    likelihoods = {}

    # Count each class label's frequency
    for row in csv_data:
        class_label = row[-1]
        class_frequency[class_label] = class_frequency.get(class_label, 0) + 1

    # Calculate the prior probabilities for each class label
    for class_label, count in class_frequency.items():
        prior_probabilities[class_label] = count / len(csv_data)

    # Loop through all columns except for the Localization column (target)
    for i in range(len(csv_data[0]) - 1):
        likelihoods[i] = {}
        
        # Loop through each class in Localization
        for class_label in class_frequency:
            likelihoods[i][class_label] = {}
            counts = {}

            # Count frequency for a value given specific class label
            for row in csv_data:
                if row[-1] == class_label:
                    value = row[i]
                    counts[value] = counts.get(value, 0) + 1

            # Calculate likelihoods for each of the value given a specific class label
            for value, count in counts.items():
                likelihoods[i][class_label][value] = count / class_frequency[class_label]

    return prior_probabilities, likelihoods

# Function uses testing data to make predictions from the naive baye model
def use_naive_baye(prior_probabilities, likelihoods, row):
    posteriori_probabilities = {}
    
    # Loop through each of the class label to calculate the posterior probability
    for class_label, prior_probabilities in prior_probabilities.items():
        class_label_probability = math.log(prior_probabilities)
        
        # Loop through all columns except for the Localization column (target)
        for i in range(len(row) - 1):
            value = row[i]

            # Add the non-zero likelihoods to the probability
            if value in likelihoods[i][class_label]:
                class_label_probability += math.log(likelihoods[i][class_label][value])
            # If the likelihood is 0 then assign a very small probability to avoid 0
            else:
                class_label_probability += math.log(0.01)

        # Assign the final overall probability to the class label    
        posteriori_probabilities[class_label] = class_label_probability

    # Return the class label with the largest overall probability
    return max(posteriori_probabilities, key = posteriori_probabilities.get)

# Function tests the naive baye model and assigns a score to the results based on their accuracy
def test_and_score(test_csv_data, prior_probabilities, likelihoods):
    actual = []
    predictions = []
    correct_count = 0
    
    # Loop through rows of the csv data
    # Get predictions using use_naive_baye function
    for row in test_csv_data:
        gene_id = row[0]
        predicted_localization = use_naive_baye(prior_probabilities, likelihoods, row)
        
        predictions.append([gene_id, predicted_localization])
        actual.append(row[-1])
    
    # Loop through predictions and increment the correct count if the Localization prediction for a GeneID matches the actual correct categorization
    for i in range(len(predictions)):
        if predictions[i][1] == actual[i]:
            correct_count += 1

    # Calculate, format, and print the results
    print(f"Model Accuracy Percentage: {(correct_count / len(actual)) * 100 :.2f}%")

    return predictions

# Main function
def main():
    # Import and impute data
    training_data = impute(csv_file_import("Gene_Data/Genes_relation.data"))
    # Train model
    prior_probabilities, likelihoods = train_naive_baye(training_data)

    keys_data = {}

    # Loop through each row in the keys file and assign value to the GeneIDs
    for row in csv_file_import("Gene_Data/keys.txt"):
        keys_data[row[0]] = row[1]

    # Import and impute test data then merge it with the data from the key file
    testing_data = merge_csv_data(impute(csv_file_import("Gene_Data/Genes_relation.test")), keys_data)

    # Runs test data through the model, assigns an accuracy score to it, then writes GeneIDs and predicted Localizations to a csv
    predictions = test_and_score(testing_data, prior_probabilities, likelihoods)
    csv_file_export("predictions.csv", predictions) 

if __name__ == "__main__":
    main()