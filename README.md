Following the Udemy course, "Machine Learning Practical", by Kirill Eremenko, Hadelin de Ponteves, Dr. Ryan Ahmed, Ph.D., MBA SuperDataScience Team, Rony Sulca.

# Machine Learning
## Fashion Class Classification
A classification model with Convolutional Neural Network (CNN) to classify the images of fashion clothes into 10 categories. The fashion dataset consists of 70,000 images divided into 60,000 training and 10,000 testing samples. Dataset sample consists of 28x28 grayscale image, associated with a label from 10 classes. The 10 classes are as follows: 0 => T-shirt/top; 1 => Trouser; 2 => Pullover; 3 => Dress; 4 => Coat; 5 => Sandal; 6 => Shirt; 7 => Sneaker; 8 => Bag; 9 => Ankle boot. Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255.

    1/ Accuracy:              training (87%), test (87%)
    2/ Libraries:             numpy, pandas, pyplot, seaborn, random, keras, sklearn
    3/ Dataset:               Fashion-MNIST
    4/ Activation Function:   relu, sigmoid
    5/ Loss Function:         sparse_categorical_crossentropy
    6/ Optimizer:             adam

## Directing customers to subscription through app behavior analysis
A Logistic Regression model to predict which users will not subcribe to the paid membership based on app behavior analysis of the free products/services the company offered to the customers. So, that greater marketing efforts can go into trying to "convert" them to not-likely to subscribe users whom the company need to target with additional offers and promotions. Because of the costs of these offers, the company does not want to offer them to everybody, especially customers who were going to enroll anyways.

    1/ Accuracy:              training (78%), test (78%)
    2/ Libraries:             numpy, pandas, pyplot, seaborn, time, parser, sklearn
    3/ Dataset:               'appdata10.csv'
    4/ Feature Scaling:       StandardScaler
    5/ Evaluating:            k-Fold Cross Validation
    6/ Tuning:                GridSearch, Regularization

## Predicting the likelihood of e-signing a loan based on financial history
A machine learning model to predict for quality applicants by analyzing the financial history of the loan applicants, and choosing whether or not the applicant is too risky to be given a loan. The goal is to asses the quality of the applicants the company receives from an intermediary marketplace. The model will predict whether or not these applicants will complete the electronic signature phase of the loan application (a.k.a e_signed).

    1/ Accuracy:              64%
    2/ Models:                Logistic Regression / SVM (linear) / SVM (rbf) / Random Forest
    2/ Libraries:             numpy, pandas, pyplot, seaborn, time, random, sklearn
    3/ Dataset:               'financial_data.csv'
    4/ Feature Scaling:       StandardScaler
    5/ Evaluating:            k-Fold Cross Validation
    6/ Tuning:                GridSearch: Entropy / Gini
