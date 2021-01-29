# Class IterValid to simulate Kaggle env generator
# forked from https://www.kaggle.com/its7171/time-series-api-iter-test-emulator

class IterValid(object):

    def __init__(self, df, max_user=1000):
        """
        ITERATOR --> df (test_df) -->  MODEL --> sample_df (sample_prediction_df)
        """
        # test_df, the yield of the iterator
        df = df [['row_id','timestamp','user_id','content_id','content_type_id','task_container_id',
                  'user_answer', 'answered_correctly',
                  'prior_question_elapsed_time','prior_question_had_explanation']]
        df = df.reset_index(drop=True)
        self.df = df

        self.user_answer = df['user_answer'].astype(str).values
        self.answered_correctly = df['answered_correctly'].astype(str).values

        # 2 new columns in the iterator yield (=the input to the model)
        df['prior_group_responses'] = "[]"
        df['prior_group_answers_correct'] = "[]"
        
        # submission format: 2 columns: 
        #  row_id        answered_correctly (y_hat probability)
        self.sample_df = df[df['content_type_id'] == 0][['row_id']]
        self.sample_df['answered_correctly'] = 0.65

        self.len = len(df)
        self.user_id = df.user_id.values
        self.task_container_id = df.task_container_id.values
        self.content_type_id = df.content_type_id.values
        
        self.max_user = max_user
        self.current = 0
        self.pre_user_answer_list = []
        self.pre_answered_correctly_list = []

    def __iter__(self):
        return self

    def fix_df(self, 
               user_answer_list, answered_correctly_list, 
               pre_start   # previous group's start position (first row)
               ):
        
        df= self.df[pre_start:self.current].copy()

        sample_df = self.sample_df[pre_start:self.current].copy()

        df.loc[pre_start,'prior_group_responses'] = '[' + ",".join(self.pre_user_answer_list) + ']'
        df.loc[pre_start,'prior_group_answers_correct'] = '[' + ",".join(self.pre_answered_correctly_list) + ']'
        
        df.drop(['user_answer', 'answered_correctly'],axis=1, inplace=True)

        self.pre_user_answer_list = user_answer_list
        self.pre_answered_correctly_list = answered_correctly_list
        return df, sample_df

    def __next__(self):

        added_user = set()
        pre_start = self.current
        pre_added_user = -1
        pre_task_container_id = -1
        pre_content_type_id = -1
        user_answer_list = []
        answered_correctly_list = []        
        
        while self.current < self.len:
            crr_user_id = self.user_id[self.current]
            crr_task_container_id = self.task_container_id[self.current]
            crr_content_type_id = self.content_type_id[self.current]

            # Each group will contain interactions from many different users, 
            # but no more than one task_container_id of questions from any single user. 
            if crr_user_id in added_user and (crr_user_id != pre_added_user or (crr_task_container_id != pre_task_container_id and crr_content_type_id == 0 and pre_content_type_id == 0)):
                # known user(not prev user or (differnt task container and both question))
                return self.fix_df(user_answer_list, answered_correctly_list, pre_start)
            
            # Each group has between 1 and 1000 users.
            if len(added_user) == self.max_user:
                if  crr_user_id == pre_added_user and (crr_task_container_id == pre_task_container_id or crr_content_type_id == 1):
                    user_answer_list.append(self.user_answer[self.current])
                    answered_correctly_list.append(self.answered_correctly[self.current])
                    self.current += 1
                    continue
                else:
                    return self.fix_df(user_answer_list, answered_correctly_list, pre_start)

            added_user.add(crr_user_id)
            pre_added_user = crr_user_id
            pre_task_container_id = crr_task_container_id
            pre_content_type_id = crr_content_type_id
            user_answer_list.append(self.user_answer[self.current])
            answered_correctly_list.append(self.answered_correctly[self.current])
            self.current += 1
        
        if pre_start < self.current:
            return self.fix_df(user_answer_list, answered_correctly_list, pre_start)
        else:
            raise StopIteration()            
