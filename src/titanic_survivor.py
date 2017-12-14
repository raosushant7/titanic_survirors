import pandas as pd
# import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


class TrainingData:

    def __init__(self):

        self.survivors_df = pd.read_csv('D:/kaggle/titanic_survivors/data/input/train.csv')
        self.survivors_df.columns = self.survivors_df.columns.str.lower()
        self.x_for_reg = ['passengerid', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
        self.x_df = self.survivors_df[self.x_for_reg]
        self.y_df = self.survivors_df['survived']
        super(TrainingData, self).__init__()

    def cleaning_data(self):
        self.x_df['embarked'] = pd.factorize(self.x_df['embarked'])[0]
        self.x_df['sex'] = pd.factorize(self.x_df['sex'])[0]


class PredictionModels:

    def __init__(self):
        self.lr_model = LogisticRegression()
        super(PredictionModels, self).__init__()


class Prediction(PredictionModels, TrainingData):

    def __init__(self):
        TrainingData.__init__(self)
        PredictionModels.__init__(self)

    def survivor_predition(self):
        self.cleaning_data()
        self.lr_model.fit(self.x_df, self.y_df)

        return

if __name__ == '__main__':
    pred_obj = Prediction()
    pred_obj.survivor_predition()
