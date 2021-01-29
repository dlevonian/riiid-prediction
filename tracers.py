# Classes to trace (keep the record) of the encountered sequences:
# - UserTracer: the most recent WINDOW trajectory (all features) for each User
# - QuestionTracer: all question ids ever encountered by each User
# - LectureTarcer: timesteps of the last relevan lecture (same Part) for each User
import numpy as np
import pandas as pd
import pickle
import time

class UserTracer(object):

    def __init__(self, window, po):
        
        self.window = window
        self.po=po 
        self.db = dict()


    def from_df(self, train_df): 

        unique_users = train_df.user_id.unique()
        user_idx = np.cumsum(train_df.user_id.value_counts().sort_index().values)
        split_features = np.split(train_df.values, user_idx[:-1])
        assert len(unique_users) == len(split_features)

        for i, user in enumerate(unique_users):
            
            uf = split_features[i]  # user features
            uf = uf[uf[:,self.po.timestamp].argsort()]   # sort by timestamp
            uf = uf[-self.window-1:,:] # take last (window+1) timesteps
            
            self.db[user] = uf # create entry

            assert len(uf)<=self.window+1
            assert user==int(uf[0,self.po.user_id])

        return self


    def add_row(self, row):
        """ Input: from the validation cycle, a 1D row
            If user exists in DB:   append the row to the user_window
                            else:   create a new key and add row as the starting user_window
        """
        expanded_row = np.array(row).reshape(1,-1)
        user = row[self.po.user_id]
       
        if user in self.db.keys():
            uf = self.db[user]
            assert row[self.po.timestamp] >= max(uf[:,self.po.timestamp])  # timestamp integrity
            self.db[user] = np.concatenate((uf, expanded_row), axis=0)[-self.window-1:]

        else:
            self.db[user] = expanded_row

    def get_window(self, user):
        """Returns the window of the shape (1, WINDOW, F_WIDTH)
           if ut-native feature window is shorter, pad with zeros
        """
        uf = self.db[user].copy()
        uf = prepare_features(uf, self.po)
        uf = uf[-self.window:]
        
        assert uf.shape[0]<=self.window

        # pad extra zeros to dimension_0 if necessary
        uf = np.pad(uf, ((0,self.window-uf.shape[0]),(0,0)))  
        uf = np.expand_dims(uf, axis=0)
       
        assert uf.ndim==3
        return uf


    def from_pickle(self, file_path):
        tic=time.time()
        with open(file_path, 'rb') as handle:
            self = pickle.load(handle)
        print(f'done in {time.time()-tic:.1f} sec')        
        return self


    def to_pickle(self, file_path):
        tic=time.time()
        with open(file_path, 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'done in {time.time()-tic:.1f} sec')        



class QuestionTracer(object):

    def __init__(self, po):
        
        self.po=po 
        self.eq = dict()  # eq: user's encountered questions

    def from_df(self, train_df): 
        unique_users = train_df.user_id.unique()
        user_idx = np.cumsum(train_df.user_id.value_counts().sort_index().values)
        split_features = np.split(train_df.question_id.values, user_idx[:-1])
        assert len(unique_users) == len(split_features)

        for i, user in enumerate(unique_users):
            self.eq[user] = split_features[i]   # encountered questions
        return self


    def update(self, user, question):
        if user in self.eq.keys():  
            self.eq[user] = np.concatenate((self.eq[user], [question]))
        else:
            self.eq[user]=np.array([question])


    def retrieve(self, user, question):
        
        return user in self.eq.keys() and question in self.eq[user]


    def from_pickle(self, file_path):
        tic=time.time()
        with open(file_path, 'rb') as handle:
            self = pickle.load(handle)
        print(f'done in {time.time()-tic:.1f} sec')        
        return self


    def to_pickle(self, file_path):
        tic=time.time()
        with open(file_path, 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'done in {time.time()-tic:.1f} sec')        



class LectureTracer(object):

    def __init__(self):
        self.db = dict()  # dictionary: db[user_id]={part_1 : latest_timestamp_1, ...}

    def from_df(self, latest_lectures): 
        for _,row in latest_lectures.iterrows():
            self.db[row.user_id]={k:row.iloc[k] for k in range(1,8)}
        return self

    
    def update(self, user, part, timestamp):
        if user not in self.db.keys():  self.db[user]=dict()
        self.db[user][part]=timestamp


    def retrieve(self, user, part, timestamp, MAXGAP=2e9):
        tsl = 0
        if (user in self.db.keys() and
                part in self.db[user].keys() and
                    not np.isnan(self.db[user][part])
            ):
            tsl = min(timestamp - self.db[user][part], MAXGAP)
        return tsl


    def from_pickle(self, file_path):
        tic=time.time()
        with open(file_path, 'rb') as handle:
            self = pickle.load(handle)
        print(f'done in {time.time()-tic:.1f} sec')        
        return self


    def to_pickle(self, file_path):
        tic=time.time()
        with open(file_path, 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'done in {time.time()-tic:.1f} sec')        

