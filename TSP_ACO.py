import numpy as np
import pandas as pd
import random
import matplotlib as mpl

class TSP():
    def __init__(self, file,m,alpha,beta,rho,Q,iter,iter_max):
        self.file = file
        self.m = m  #蚂蚁信息
        self.alpha = alpha  #信息素重要程度因子
        self.beta = beta    #启发函数重要程度因子
        self.rho = rho  #信息挥发因子
        self.Q = Q  #常系数
        self.iter = iter    #迭代次数初值
        self.iter_max = iter_max    #最大迭代次数
    def init(self):
        self.city = self.load_city(self.file)   #加载数据
        self.n = self.city.shape[0] #城市数量
        self.d = self.cal_distance(self.city)   #计算城市之间距离
        self.eta =np.ones((self.city.shape[0],self.city.shape[0])) / self.d #启发函数
        self.tau = np.ones((self.n,self.n))    #信息素矩阵
        self.route_best = np.zeros((self.iter_max,self.n),dtype='int64')   #各代最佳路径
        self.length_best = np.zeros((self.iter_max,1))   #各代最佳路径长度
        self.length_ave = np.zeros((self.iter_max,1))    #各代路径的平均长度

    def run(self):
        for iter in range(self.iter_max):
            #随机产生各蚂蚁的起点
            table = self.init_start(self.m,self.n)
            #构建解析空间
            citys_index = np.arange(0,self.n)
            #逐个蚂蚁路径选择
            for i in range(0,self.m):
                #逐个城市路径选择
                for j in range(1,self.n):
                    tabu = table[i,0:j]    #已访问的城市集合（禁忌表）
                    allow = np.array(list(set(citys_index).difference(set(tabu))))  #待访问城市集合
                    P = allow.copy().astype('float64')
                    #计算城市之间转移的概率
                    for k in range(0,allow.shape[0]):
                        P[k] =self.tau[tabu[-1],allow[k]]**self.alpha * self.eta[tabu[-1],allow[k]]**self.beta
                    P= P/np.sum(P)
                    #轮盘赌注选择下个访问城市
                    Pc = np.cumsum(P)
                    target_index = np.argwhere(Pc >= np.random.rand())
                    target = allow[target_index[0]]
                    table[i,j] = target
            #计算各蚂蚁之间路径距离
            length = self.cal_length(table,self.m,self.n,self.d)
            #计算最短路径距离及平均距离
            self.select_best(table,iter,length)
            #更新信息素：table蚂蚁群,length蚂蚁路径距离,Q常系数,rho信息挥发因子,tau信息素矩阵
            self.update_tau(table,length)
            #重置蚂蚁空间
            table = np.zeros((self.m, self.n))
            print("第",iter,"次迭代最佳距离",self.length_best[iter],'\n',"-----------------------")

    def load_city(self, file):
        '''导入数据'''
        city = pd.read_excel(file,header=None).values
        return city

    def cal_distance(self,city):
        '''计算城市之间互相距离'''
        n = city.shape[0]
        d = np.zeros([n,n])
        for i in range(0,n):
            for j in range(0,n):
                if i != j:
                    d[i,j] = (np.sqrt((city[i,0]-city[j,0])**2 +
                                 (city[i,1]-city[j,1])**2))
                else:
                    d[i,j] = 1e-4
        return d

    def cal_length(self,table,m,n,d):
        '''计算各蚂蚁之间路径距离'''
        length = np.zeros(m)
        for i in range(0,m):
            route = table[i,:]
            for j in range(0,n-1):
                length[i] = length[i] + d[route[j],route[j+1]]
            length[i] =length[i] + d[route[n-1],route[0]]
        return length

    def init_start(self,m,n):
        '''初始化蚂蚁'''
        start = np.zeros((self.m,self.n),dtype='int64')
        for i in range(0,m):
            temp = np.random.randint(0, n)
            start[i,0] = temp
        return np.array(start)

    def select_best(self,table,iter,length):
        '''计算最短路径及平均距离'''
        if iter ==0:
            min_index = np.argmin(length)
            min_length = np.min(length)
            self.length_best[iter] = min_length
            self.length_ave[iter] = np.mean(length)
            self.route_best[iter] = table[min_index,:]
            pass
        else:
            min_index = np.argmin(length)
            min_length = np.min(length)
            self.length_best[iter] = np.min([self.length_best[iter-1],min_length])
            self.length_ave[iter] = np.mean(length)
            if self.length_best[iter] == min_length:
                self.route_best[iter] = table[min_index]
            else:
                self.route_best[iter] = self.route_best[iter-1]

    def update_tau(self,table,length):
        # 更新信息素
        delta_tau = np.zeros((self.n, self.n))
        # 逐个蚂蚁计算
        for i in range(self.m):
            # 逐个城市计算
            for j in range(self.n - 1):
                delta_tau[table[i, j], table[i, j + 1]] = delta_tau[table[i, j], table[i, j + 1]] + self.Q / length[i]
            delta_tau[table[i, self.n - 1], table[i, 0]] = delta_tau[table[i, self.n - 1], table[i, 0]] + self.Q / \
                                                           length[i]
        self.tau = (1 - self.rho) * self.tau + delta_tau


def main():
    file = 'city.xlsx'
    m = 50  #蚂蚁信息
    alpha = 1   #信息素重要程度因子
    beta = 5    #启发函数重要程度因子
    rho = 0.1   #信息挥发因子
    Q = 1   #常系数
    iter = 1 #迭代次数初值
    iter_max = 200  #最大迭代次数
    tsp = TSP(file,m,alpha,beta,rho,Q,iter,iter_max)
    tsp.init()
    tsp.run()


if __name__ == '__main__':
    main()
# list(set(a).intersection(set(b)))  # 交集
# list(set(a).union(set(b)))  # 并集
# list(set(b).difference(set(a)))  # 差集