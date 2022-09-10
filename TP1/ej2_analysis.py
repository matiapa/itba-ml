from ej2 import *
from sklearn import metrics

categories = ['Nacional', 'Deportes', 'Entretenimiento', 'Economia']
df = noticias_argentinas_parser("data/Noticias_argentinas_no_extra_columns.csv", categories)

def best_block_analysis():
    ks = [2, 3, 4, 5, 6, 7, 8, 9, 10]

    # create csv file
    with open('ej2_best_block_size.csv', 'w') as csvfile:
        for k in ks:
            best_block, best_block_accuracy = cross_validation(df, k, categories)
            # save in csv file the result
            csvfile.write(str(k) + ',' + str(best_block_accuracy) + '\n')

def print_metrics(tp_metrics, fp_metrics, fn_metrics, tn_metrics):
    for category in categories:
        precision = tp_metrics[category]/(tp_metrics[category] + fp_metrics[category])
        recall = tp_metrics[category]/(tp_metrics[category] + fn_metrics[category])
        f1 = 2 * ((precision * recall) / (precision + recall))
        accuracy = (tp_metrics[category] + tn_metrics[category]) / (tp_metrics[category] + tn_metrics[category] + fp_metrics[category] + fn_metrics[category])
        print("Category: " + category)
        print("Precision: " + str(precision))
        print("Recall: " + str(recall))
        print("F1: " + str(f1))
        print("Accuracy: " + str(accuracy))

    # crear matriz de confusion
    for category in categories:
        matrix = [[tp_metrics[category], fn_metrics[category]], [fp_metrics[category], tn_metrics[category]]]
        df_cm = pd.DataFrame(matrix, index = ['Positivo', 'Negativo'], columns = ['Positivo', 'Negativo'])
        plt.figure(figsize = (5,3))
        sn.heatmap(df_cm, annot=True, fmt='g')

        # increase font size
        sn.set(font_scale=1.4)

        # save the figure
        plt.savefig('confusion_matrix_'+category+'.png')


def metrics_analysis():
    
    best_block, test, best_block_accuracy = cross_validation(df, 5, categories)
    dictionary, total_words_by_category, total_words = fill_dictionary(best_block)
    # tp_metrics, fp_metrics, fn_metrics, tn_metrics = get_metrics(categories, dictionary, total_words_by_category, total_words, test)
    # print_metrics(tp_metrics, fp_metrics, fn_metrics, tn_metrics)

    return dictionary, total_words_by_category, total_words, test

# best_block_analysis()

# metrics_analysis()

def roc_curve():
    # data, _ = divide_dataset(df, 0.2)
    best_block, test, best_block_accuracy = cross_validation(df, 5, categories)
    dictionary, total_words_by_category, total_words = fill_dictionary(best_block)

    tp_metrics = {}
    fp_metrics = {}
    fn_metrics = {}
    tn_metrics = {}

    umbrals = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    list_probabilities = []
    list_expected_category = []
    list_result_category = []

    for index, row in test.iterrows():
        category, probabilities = analyze_title(categories, row['title'], dictionary, total_words_by_category, total_words)
        list_probabilities.append(probabilities)
        list_expected_category.append(row['category'])
        list_result_category.append(category)

    fpr = {}
    tpr = {}

    for category in categories:
        fpr[category] = []
        tpr[category] = []

    for umbral in umbrals:
        print("Umbral: " + str(umbral))

        for category in categories:
            tp_metrics[category] = 0
            fp_metrics[category] = 0
            fn_metrics[category] = 0
            tn_metrics[category] = 0

        for index in range(len(list_probabilities)):
            # print(index)
            probabilities = list_probabilities[index]

            # analyze false negative, false positive, true negative and true positive based on the umbral

            for category in categories:
                if category == list_expected_category[index]:
                    if probabilities[category] >= umbral:
                        tp_metrics[category] += 1
                    else:
                        fn_metrics[category] += 1
                else:
                    if probabilities[category] >= umbral:
                        fp_metrics[category] += 1
                    else:
                        tn_metrics[category] += 1

        # calculate false positive rate and true positive rate
        
        for category in categories:
            if tp_metrics[category] + fn_metrics[category] == 0:
                tpr[category].append(0)
            else:
                tpr[category].append(tp_metrics[category]/(tp_metrics[category] + fn_metrics[category]))
            
            if fp_metrics[category] + tn_metrics[category] == 0:
                fpr[category].append(0)
            else:
                fpr[category].append(fp_metrics[category]/(fp_metrics[category] + tn_metrics[category]))

    # plot each point for each category
    for category in categories:
        # plot points and line
        plt.plot(fpr[category], tpr[category], label=category, marker='o')
        # calculate area under the curve
        auc = metrics.auc(fpr[category], tpr[category])
        print("AUC for " + category + ": " + str(auc))

    plt.legend()
    plt.xlabel('Tasa Falsos Positivos')
    plt.ylabel('Tasa Verdaderos Positivos')
    plt.title('Curva de ROC')
    plt.show()

roc_curve()

# train, test = divide_dataset(df, 0.8)
# dictionary, total_words_by_category, total_words = fill_dictionary(train)
# tp_metrics, fp_metrics, fn_metrics, tn_metrics = get_metrics(categories, dictionary, total_words_by_category, total_words, test)
# print_metrics(tp_metrics, fp_metrics, fn_metrics, tn_metrics)
