from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import RandomOverSampler,SMOTE
from sklearn.model_selection import train_test_split


class dataSampling:

    def __init__(self,df,y_column,test_size,random_state,stratify=True,shuffle=True):
        self.df = df
        self.random_state = 42
        self.y_column = y_column
        self.test_size =test_size
        self.random_state = random_state
        self.stratify = stratify
        self.shuffle = shuffle

        def data_train_test_split():
            self.X,self.y = self.df[list(set(self.df.columns).difference([self.y_column]))],self.df[self.y_column]
            if stratify == True:
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,self.y, test_size=self.test_size,shuffle=self.shuffle,stratify=self.y)
            else:
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,self.y, test_size=test_size,shuffle=shuffle)

        data_train_test_split()

    
    def get_data_without_sample(self):
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    # Methods to Oversample
    # NOTES: add ADASYN

    def get_data_smote(self):
        sm = SMOTE(random_state=self.random_state)
        X_sm, y_sm = sm.fit_resample(self.X_train, self.y_train)
        return X_sm,self.X_test,y_sm,self.y_test

    def get_data_random_over_sample(self):
        os =  RandomOverSampler(random_state=self.random_state)
        X_os,y_os = os.fit_resample(self.X_train,self.y_train) 
        return X_os,self.X_test,y_os,self.y_test
    
    # Methods to Undersample
    # NOTES: add cluster centroids, tomek links

    def get_data_near_miss(self,sampling_strategy=0.1):
        nm = NearMiss(sampling_strategy=sampling_strategy)
        X_nm,y_nm = nm.fit_resample(self.X_train,self.y_train)
        return X_nm,self.X_test,y_nm,self.y_test

    def print_class_percentage(self,X_train,X_test,y_train,y_test):
        print("=====Train class count=====")
        y_tr_cnts = y_train.value_counts()
        print(y_tr_cnts)
        print(y_tr_cnts/sum(y_tr_cnts))
        print("\n=====Test class count=====")
        y_ts_cnts = y_test.value_counts()
        print(y_ts_cnts)
        print(y_ts_cnts/sum(y_ts_cnts))
