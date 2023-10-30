from glob import glob
import os
import pandas as pd
import numpy as np

from aif360.datasets import StandardDataset

# TextLoader
from gensim.models import Word2Vec
from tqdm import tqdm


class DataLoader:
    def __init__(self, data_dir=None):
        if not data_dir:
            self.directory = os.getcwd()
            self.files = glob(os.getcwd())
        else:
            self.directory = data_dir
            self.files = glob(os.path.join(data_dir, '*.*'))
            
    def Set_BaseData(self, data=None):
        if not data:
            self.data = self.files
        else:
            self.data = data
    
    def Get_annotation(self, input_data, colnames=[], method=None, **params):
        self.colnames = colnames
        
        if not method:
            self.annotation = np.zeros([len(input_data)])
        else:
            self.annotation = method(input_data, params)
        
        return len(self.annotation)
    
    def Append_annotation(self, input_data, colnames=[], method=None, **params):
        result = []
        
        if not method:
            result = np.zeros([len(input_data)])
            self.annotation.append(result)
        else:
            result = method(input_data, params)
            #print(result)
            
            new_anno = []
            for i, point in enumerate(result):
                if type(point) == list and type(self.annotation[i]) == list:
                    temp = self.annotation[i] + point
                elif type(point) == list and type(self.annotation[i]) != list:
                    point.insert(0, self.annotation[i])
                    temp = point
                elif type(point) != list and type(self.annotation[i]) == list:
                    #print(point, self.annotation[i])
                    self.annotation[i].append(point)
                    temp = self.annotation[i]
                else:
                    temp = [self.annotation[i], point]
                
                new_anno.append(temp)
            #print(new_anno)
            self.annotation = new_anno
            
        # Append colnames
        self.colnames.extend(colnames)
        
        idx = len(self.colnames) + 1
        while len(self.colnames) < len(self.annotation):
            name = f"anno_{idx}"
            self.colnames.append(name)
            idx += 1
                
        return len(self.annotation)
    
    def Drop_annotation(self, colnames):
        # Drop on annotation table
        new_anno = np.array(self.annotation)
        t_idxs = []
        for col_name in colnames:
            idx = self.colnames.index(col_name)
            t_idxs.append(idx)
        
        new_anno = np.delete(new_anno, t_idxs, 1)
        
        self.annotation = new_anno
        
        # Drop on colnames list
        for col_name in colnames:
            self.colnames.remove(col_name)
        
        return len(self.annotation)
    
    def convert_to_DataFrame(self):
        df_input = {"Data": self.data}
        
        anno = np.array(self.annotation)
        if anno.ndim > 1:
            if anno.shape[0] == len(self.data):
                anno = anno.transpose()

            for i, an in enumerate(anno):
                try:
                    key = self.colnames[i]
                except IndexError:
                    key = f"anno_{i+1}"
                df_input[key] = an
        else:
            if not self.colnames:
                key = "anno_1"
            else:
                key = self.colnames[0]
            df_input[key] = anno
            
        df = pd.DataFrame(df_input)
        
        return df
    
    
class TextLoader(DataLoader):
    def __init__(self, corpus, tokenizer, model=None):
        self.corpus = corpus
        self.tokenized_corpus = [tokenizer(sent) for sent in corpus]

        # Document vectorizing: average Word2Vec
        if model:
            self.model = model
        else:
            print('Word2Vec modeling...')
            print('size={size}, workers={workers}, skipgram={sg}'.format(size=100, workers=4, sg=1))
            self.model = Word2Vec(self.tokenized_corpus, size=100, workers=4, sg=1)
            print('Modeling successed!', end='\n\n')

        print('Vectorizing document...')
        doc_vec = []
        for s in tqdm(self.tokenized_corpus):
            wv_list = [self.model.wv.get_vector(w) for w in s if w in self.model.wv.index2word]

            if not wv_list:
                dv = np.zeros(100)
            else:
                dv = sum(wv_list) / len(wv_list)

            doc_vec.append(dv)
        print('Vectorized finished!')

        self.vectorized_df = pd.DataFrame(doc_vec, columns=['wv_{dim}'.format(dim=i) for i in range(100)])

    
    def bootstrap(self, key_group, threshold):
        result = []
        for word in key_group:
            for w, s in self.model.wv.most_similar(word):
                if s > threshold:
                    result.append(w)
                    
        result = key_group + result
        
        return result
    
    
    def bias_labeling(self, privilege_words, unprivilege_words):
        ## Refining the lists
        result = []
        for word in privilege_words:
            if not word in unprivilege_words:
                result.append(word)
        privilege_group = result
        
        result = []
        for word in unprivilege_words:
            if not word in privilege_words:
                result.append(word)
        unprivilege_group = result
        
        # Check the list
        bias_list = []
        for sent in self.tokenized_corpus:
            pri_cnt = 0
            unpri_cnt = 0
            for word in sent:
                if word in privilege_group:
                    pri_cnt += 1
                elif word in unprivilege_group:
                    unpri_cnt += 1

            if pri_cnt >= unpri_cnt:
                bias = 1
            else:
                bias = 0
            bias_list.append(bias)
        
        return bias_list
    
    
    def Make_Data(self, privilege_keys, unprivilege_keys, threshold):
        # One line perform full course
        self.Set_BaseData(self.corpus)
        
        # Make bias
        pr_group = self.bootstrap(privilege_keys, threshold)
        unpr_group = self.bootstrap(unprivilege_keys, threshold)
        
        bias = self.bias_labeling(pr_group, unpr_group)
        
        # Set colnames
        self.colnames = ['bias']
        
        # Make annotation
        anno = []
        for point in bias:
            anno.append([point])
        self.annotation = anno
        
        print("Ready to make dataframe. But there is no label. So you need to append label column using Append_annotation method.")
        
    
class ImageLoader(DataLoader):
    def make_images(self):
        pass

class SoundLoader(DataLoader):
    def make_sounds(self):
        pass
    
class VideoLoader(DataLoader):
    def make_videos(self):
        pass