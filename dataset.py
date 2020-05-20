import warnings
warnings.simplefilter(action='ignore',category=FutureWarning)
import pandas as pd
from torch.utils.data.dataset import Dataset
import numpy as np
import math
import datetime
from dateutil.relativedelta import *


class CustomDataset(Dataset):
    def __init__(self,args,data_root,file_name,idxes_file_name,for_test=False,transform=None):
        # Initialize
        self.args = args
        self.for_test = for_test
        self.transform =transform
        self.random_length = args.random_length
        self.before_lim = self.args.before_lim
        self.after_lim = self.args.after_lim
        self.idxes_file_name = idxes_file_name

        # Read the csv file
        data = pd.read_csv(data_root+file_name)#지역 + 에너지 + 날씨관측 + 날씨예보1,2,3 + 날씨통계
        self.cols = data.columns.tolist()
        self.info_cols = self.cols[0:2]+self.cols[3:8]
        self.data_cols = self.cols[8:]
        self.info = data[self.info_cols].astype("float32")
        self.info['day_in_month'] = pd.to_datetime(data['date']).dt.days_in_month
        self.data = data[self.data_cols].astype("float32")

        self.weather_ob_col = self.data_cols[5:13]
        self.weather_fo1_col = self.data_cols[13:21]
        self.weather_fo2_col = self.data_cols[21:29]
        self.weather_fo3_col = self.data_cols[29:37]
        self.weather_st_col = self.data_cols[37:]
        self.weather_after_col = ["weather1","weather2","weather3","weather4","weather5","weather6","weather7","weather8"]
        #사용가능 idxes 리스트만들기
        if not for_test: # 테스트용은 충분히 긴 같은리스트이지만, 학습용은 길이에 따라 다르게 리스트 생성
            self.idxes_file_name = str(self.before_lim)+self.idxes_file_name
        try:
            self.idxes = pd.read_csv(self.args.usable_idx_path+self.idxes_file_name)
            print("Load!!")
        except:
            self.idxes=self.info["before_err"] > self.before_lim
            self.idxes= self.idxes[self.idxes==True].index
            self.idxes = self.idxes.to_frame(index=False)
            self.idxes = self.idxes.sample(frac=1).reset_index(drop=True)
            self.idxes.to_csv(self.args.usable_idx_path+self.idxes_file_name,index=False,header=False)
            print("Make indxes file!!({})".format(self.idxes_file_name))
        # Normalize
        self.normalize_factor = pd.read_csv(args.normalize_factor_path)
        self.normalization()
        self.data = self.data.astype("float32")

        self.data_len = len(self.idxes)
        print("length : {}".format(len(self.idxes)))

    def normalization(self):
        for i, header in enumerate(list(self.normalize_factor.columns)):
            if i > 0 and i < 4:
                self.data.at[:, header] = (self.data[header] - self.normalize_factor.loc[1][header]) / (self.normalize_factor.loc[0][header] - self.normalize_factor.loc[1][header])

            elif i > 3:
                tmp_header = ['obse_weather', 'fore_weather1_', 'fore_weather2_', 'fore_weather3_', 'stat_weather']
                for th in tmp_header:
                    self.data.at[:, th+'{}'.format(i-3)] = (self.data[th+'{}'.format(i-3)] - self.normalize_factor.loc[1][header]) / (self.normalize_factor.loc[0][header] - self.normalize_factor.loc[1][header])
        self.data.at[:, 'month'] = (self.data['month'] - 1) / (12-1)
        self.data.at[:, 'dayweek'] = self.data['dayweek'] / 6

    def __getitem__(self, from_index):
        self.after_lim =int(from_index%100)
        self.before_lim = int((from_index/100)%1000)
        '''
        ex) 전체 200개 중 사용가능한 데이터가 맨뒤 50개 라고 할때
            from_index는 0~49로 사용가능한 갯수내 범위 index는 150~199으로 매칭됨
            idxes는 사용가능한 길이 범위 내에서 뽑는 것 dataset의 사용가능 인덱스에 매칭하여 반환
        '''
        from_index =int(from_index/100000)
        index = self.idxes.iloc[from_index].item()
        while(True):
            #데이터 정상치 길이가 충분한지 고려
            if self.info.iloc[index]["before_err"]<=self.before_lim or self.info.iloc[index]["after_err"]<self.after_lim:
                from_index = np.random.randint(0, self.data_len)
                index = self.idxes.iloc[from_index].item()
            # 총 길이가 31일이 안될 때, 기준일 앞에 정상인 데이터 + 예측일이 31일 이상이 되도록 선택
            elif not 31 - self.after_lim < self.info.iloc[index]["before_err"]:
                from_index = np.random.randint(0,self.data_len)
                index = self.idxes.iloc[from_index].item()
            #테스트시 2월 인데 31일을 예측 할 필요가 없다.
            elif self.after_lim>self.info.iloc[index]['day_in_month'] and self.for_test:
                try:
                    from_index = from_index + self.args.test_batch_size * 31
                    index = self.idxes.iloc[from_index].item()
                except:
                    from_index = np.random.randint(0, self.data_len)
                    index = self.idxes.iloc[from_index].item()
            else:
                break

        inputs = np.asarray(self.data.iloc[index-self.before_lim:index+self.after_lim])
        info = np.asarray(self.info.iloc[index])

        # 입력 + 출력이 31일이 안됬을 때를 대비해, 기준일 이전 31일치를 미리 뽑아서 저장
        front_lim = 31-self.after_lim
        front_data = np.asarray(self.data.iloc[index - front_lim:index+1][['ele', 'wat', 'gas']])
        if self.transform is not None:
            inputs = self.transform(inputs)

        return inputs,info,self.before_lim,self.after_lim, front_data

    def __len__(self):
        return self.data_len

