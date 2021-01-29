
# Riiid: library of helper functions
import numpy as np
import pandas as pd

def process_df(df, remove_lectures=REMOVE_LECTURES):
    """Trim the unnecesary columns, leave only those to be processed into tf dataset
       Create a new column 'user_total': count of the interaction for the user
    """
    df = df[['user_id','content_id','content_type_id','answered_correctly','timestamp']]
    df = df.rename(columns={'content_id':'question_id', 'answered_correctly':'y_true'})

    if remove_lectures: 
        df = df[df.content_type_id==0].drop('content_type_id',axis=1)
    
    user_total=[1+np.arange(c) for c in df.user_id.value_counts().sort_index()]
    df['user_total'] = np.concatenate(user_total)
  
    return df


def add_repeat_questions(df):
    """Adds repeat indicator - whether the user has already answered this question before
    """
    user_idx = np.cumsum(df.user_id.value_counts().sort_index().values)
    split_qids = np.split(df.question_id.values, user_idx[:-1])

    unique_users = df.user_id.nunique()
    assert unique_users==len(split_qids)
    
    tic = time.time()
    repeat = []
    for user, arr in enumerate(split_qids):
        rq = np.zeros_like(arr, dtype=int)
        for i, el in enumerate(arr):
            if el in arr[:i]: rq[i]=1
        repeat.append(rq)
        print(f'\rprocessed {100*user/unique_users:.1f}%   {time.time()-tic:.1f} sec',end='')        

    df['repeat']=np.concatenate(repeat)
    return df