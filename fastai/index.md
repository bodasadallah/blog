# FastAi 2020


### Lecture two

#### P value:

    determines if some numbers have realationship, or they are random (whether they are independat or dependant)

    suppose we have the temp and R (transmitity) values of a 100 cities in China and we want to see if there's a relation between them.

    then we generate many sets of random numbers for each parameter then we calculate the P value which would tell us what's the percentage this slope  is a random, and that ther's no relation

    A P-value is the probability of an observed result  assuming that the null hypothesis (there's no relation ) is true


    PS: P-value also is dependant on the size of the set u used, so they don't measure the importance of the result.

    so don't use P-values

    If the P value is > 0.5 then we sure that these daata have no ralation, and if  the p-value is so small, then there's a chance that the data have a relation

### Lecture three

In the course video and book, we built a bear classifier, using data from Microsoft Ping Api.

To build a deep learning model, we have first to gather the data, then we should prepare the data to be in the right format for the model, then we train the model and observe if we get satisfiable results, if not then we try to investigate to try to get better results. Finally we save our model and deploy it!

while gathering the data, notice that all the time the data would be biased, and in sometimes these biases would be severe that they can't be ignored

#### Race classifier

I have tried to rebuild the notebook and to make a Race classifier.

I got the dataset from here [Dataset](https://github.com/joojs/fairface), and then trained a small Resnet18 NN to classify images.

To deploy the app, I used Voila and MyBinder to make it available online here: [Race Classifier](https://hub.gke2.mybinder.org/user/bodasadalla98-fastai-c1-5e6j4ci5/voila/render/Race-classifier-voila.ipynb?token=RidonxHPRQmBgjyAy4RZKg)

Lastly, all the code can be found in this githun repo [here](https://github.com/BodaSadalla98/FastAi-C1/tree/main/Race-classifier)

```python

```

