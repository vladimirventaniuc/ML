import numpy as np

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
    W = np.zeros(784)
    epoch = 0
    for i in range(100):
        for X, Y in zip(data, trainingAnswers):
            if Y != label:
                Y = -1
            epoch = epoch + 1
            reg = 1 / (epoch * lam)
            if (Y * np.dot(W, X) < 1):
                W = np.add(W * (1 - reg * lam), X * (reg * Y))
            else:
                W = W * (1 - reg * lam)
    return W


def pegasos_svm_test(testingAnswers, data, w):
    vectorCount = 0
    mislabel = 0
    for vec, answer in zip(data, testingAnswers):
        vectorCount += 1
        prediction_values = []
        for wvec in w:
            prediction_values.append(np.dot(vec, wvec))
        prediction = np.argmax(prediction_values)
        if (prediction == answer):
            continue
        else:
            mislabel += 1
    return (mislabel / vectorCount)

def show_final_results(trainingVectors, trainingAnswers, testingVectors, testingAns):
    final_vectors = []

    lambdas = [2 ** -5, 2 ** -4, 2 ** -3, 2 ** -2, 2 ** -1, 2 ** 0, 2 ** 1]
    for lam in lambdas:
        print("Result for lambda = " + str(lam))
        for i in range(10):
            final_vectors.append(pegasos_svm_train(trainingVectors,lam, i, trainingAnswers))
        print("Final Testing Error with lambda 2^-5", pegasos_svm_test(testingAns, testingVectors, final_vectors))

def main():
    preprocessing()

    trainingVectors = np.load('train_save.npy')
    trainingAns = np.load('train_ans_save.npy')
    testingVectors = np.load('test_save.npy')
    testingAns = np.load('test_ans_save.npy')

    show_final_results(trainingVectors, trainingAns, testingVectors, testingAns)


if __name__ == "__main__":
    main()



