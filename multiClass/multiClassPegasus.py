import numpy as np
# import matplotlib.pyplot as plt
import sys
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.svm import SVC


def preprocessing():
    train_read = open('mnist_train.txt', 'r')
    test_read = open('mnist_test.txt', 'r')
    train_vectors = []
    train_answers = []
    test_vectors = []
    test_answers = []
    for line in train_read:
        new_vector = np.fromstring(line, dtype=int, sep=',')
        train_answers.append(new_vector[0])
        train_vectors.append([((2 * i) / 255) for i in new_vector[1:]])
    for line in test_read:
        new_vector = np.fromstring(line, dtype=int, sep=',')
        test_answers.append(new_vector[0])
        test_vectors.append([((2 * i) / 255) for i in new_vector[1:]])
    train_read.close()
    test_read.close()
    np.save('train_save.npy', train_vectors)
    np.save('train_ans_save.npy', train_answers)
    np.save('test_save.npy', test_vectors)
    np.save('test_ans_save.npy', test_answers)


def pegasos_svm_train(data, lam, label, trainingAnswers):
    print("Training classificator with label " + str(label))
    weightVector = np.zeros(784)
    objData = []
    t = 0
    for i in range(20):
        for vec, answer in zip(data, trainingAnswers):
            # all labels except for current training label are changed to -1
            if answer != label:
                answer = -1
            t = t + 1
            step = 1 / (t * lam)
            if (answer * np.dot(weightVector, vec) < 1):
                weightVector = np.add([k * (1 - step * lam) for k in weightVector], [j * step * answer for j in vec])
            else:
                weightVector = [k * (1 - step * lam) for k in weightVector]
    return weightVector


def pegasos_svm_test(testingAnswers, data, w):
    vectorCount = 0
    mislabel = 0
    for vec, answer in zip(data, testingAnswers):
        vectorCount += 1
        prediction_values = []
        # find dot product for each weight vector
        for wvec in w:
            prediction_values.append(np.dot(vec, wvec))
        prediction = np.argmax(prediction_values)
        # correctly classified
        if (prediction == answer):
            continue
        else:
            mislabel += 1
    return (mislabel / vectorCount)

def calculate_validation_errors(trainingVectors, trainingAns, lambdas):
    validationErs = []
    for l in lambdas:
        wvectors = []
        valErs = []
        folds = KFold(n_splits=5, random_state=None, shuffle=False)
        X = [x for x in range(20)]
        for train, test in folds.split(X):
            ktrain = []
            ktest = []
            testingAnswers = []
            trainingAnswers = []
            wvectors = []
            ktrain = np.concatenate(
                (trainingVectors[0:test[0]], trainingVectors[test[len(test) - 1]:len(trainingVectors)]),
                axis=0)
            ktest = trainingVectors[test[0]:test[len(test) - 1]]
            trainingAnswers = np.concatenate(
                (trainingAns[0:test[0]], trainingAns[test[len(test) - 1]:len(trainingVectors)]), axis=0)
            testingAnswers = trainingAns[test[0]:test[len(test) - 1]]

            for x in range(10):
                wvectors.append(pegasos_svm_train(ktrain, l, x, trainingAnswers))
            val = pegasos_svm_test(testingAnswers, ktest, wvectors)
            valErs.append(val)

        validationErs.append(np.average(valErs))
    return validationErs

def show_final_results(trainingVectors, trainingAnswers, testingVectors, testingAns):
    # Final Run with all testing set
    final_vectors = []
    for i in range(10):
        final_vectors.append(pegasos_svm_train(trainingVectors, 2 ** -5, i, trainingAnswers))
    print("Final Testing Error with lambda 2^-5", pegasos_svm_test(testingAns, testingVectors, final_vectors))

def main():
    preprocessing()

    trainingVectors = np.load('train_save.npy')
    trainingAns = np.load('train_ans_save.npy')
    weightVectors = []

    trainingAnswers = []
    testingAnswers = []

    # cross-validation with multiple values of lambda
    lambdas = [2 ** -5, 2 ** -4]

    # validationErs = calculate_validation_errors(trainingVectors, trainingAns, lambdas)

    # for ls, ve in zip(lambdas, validationErs):
    #     print("Lambda:", ls, "ValEr:", ve)

    trainingVectors = np.load('train_save.npy')
    trainingAns = np.load('train_ans_save.npy')
    testingVectors = np.load('test_save.npy')
    testingAns = np.load('test_ans_save.npy')

    show_final_results(trainingVectors, trainingAns, testingVectors, testingAns)


if __name__ == "__main__":
    main()



