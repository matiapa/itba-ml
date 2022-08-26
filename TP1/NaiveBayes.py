def laplace_correction(n, N, k):
    return (n + 1) / (N + k)

def initialize_probabilities(data, class_header, laplace=False):
    classes = data[class_header].unique()
    prob_classes = {}
    prob_attr_per_class = {}

    # Class probabilties P(C)
    for c in classes:
        # count of rows in data that match class c
        count = data[data[class_header] == c].shape[0]
        # total number of rows in data
        total = data.shape[0]

        if laplace:
            prob_classes[c] = laplace_correction(count, total, len(classes))
        else:
            prob_classes[c] = count / total

    # Attribute probabilites per class P(a|C)
    for c in classes:
        total = data[data[class_header] == c].shape[0]

        # iterate over the columns of data
        for attr in data.columns:
            if attr == class_header:
                continue

            # count of rows in data that match class c and attribute a
            count = data[(data[class_header] == c) & (data[attr] == 1)].shape[0]
            
            # add the probability to the dictionary
            if laplace:
                prob_attr_per_class[(c, attr)] = laplace_correction(count, total, 2)
            else:
                prob_attr_per_class[(c, attr)] = count / total

    return classes, prob_classes, prob_attr_per_class


# Predict
def predict(x, data, class_header, laplace=False):
    mult_prob = {}

    classes, p_classes, p_attr_per_class = initialize_probabilities(data, class_header, laplace)

    # Calculate P(a|C) * P(C)
    for c in classes:
        mult_prob[c] = 1

        for attr in data.columns:
            if attr == class_header:
                continue

            if x[attr] == 1:
                mult_prob[c] *= p_attr_per_class[(c, attr)]
            else:
                mult_prob[c] *= (1 - p_attr_per_class[(c, attr)])

        mult_prob[c] *= p_classes[c]

    # Return the class with the highest hipothesis probability
    max_prob = max(mult_prob, key=mult_prob.get)

    # Get P(a) to then divide and get the posteriori probability
    # P(a) = P(a|C1) * P(C1) + P(a|C2) * P(C2)
    p_a = sum(mult_prob.values())

    # Transform the V of the hipotesis to a posteriori probability
    for c in classes:
        mult_prob[c] /= p_a

    return max_prob, mult_prob

