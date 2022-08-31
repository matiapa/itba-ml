from ej2 import *

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
    tp_metrics, fp_metrics, fn_metrics, tn_metrics = get_metrics(categories, dictionary, total_words_by_category, total_words, test)
    print_metrics(tp_metrics, fp_metrics, fn_metrics, tn_metrics)

# best_block_analysis()

# metrics_analysis()

train, test = divide_dataset(df, 0.8)
dictionary, total_words_by_category, total_words = fill_dictionary(train)
tp_metrics, fp_metrics, fn_metrics, tn_metrics = get_metrics(categories, dictionary, total_words_by_category, total_words, test)
print_metrics(tp_metrics, fp_metrics, fn_metrics, tn_metrics)