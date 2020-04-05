# -*- coding: utf-8 -*-
import urllib.request
from datetime import datetime as dt
import time
import pandas as pd
from tqdm import trange
import getpass

agora = int(dt.timestamp(dt.now()))

class yahoo:
    def __init__(self,tickers=[],sp=False):
        self.tickers = tickers
        self.sp = sp
        self.path = 'C:\\Users\\{}\\Desktop\\base_equities\\'.format(getpass.getuser())
        self.n_ativos = len(self.tickers)
        
    def _download_files(self):
        """
        Baixa todos os arquivos que serão utilizados posteriormente para gerar o dataframe
        """
        erros=[]
        start_time = time.time()
        for i in trange(self.n_ativos):
            ticker = self.tickers[i]
            if ticker=='^BVSP': url = 'https://query1.finance.yahoo.com/v7/finance/download/%5EBVSP?period1=0&period2={0}&interval=1d&events=history'.format(agora)
            else: url = 'https://query1.finance.yahoo.com/v7/finance/download/{0}?period1=0&period2={1}&interval=1d&events=history'.format(ticker.upper()+'.SA', agora)
            try:
                urllib.request.urlretrieve(url, self.path + ticker.upper() + '.csv')
                if self.sp==True: print('Download de {0} concluído'.format(ticker.upper()))
            except:
                erros.append(ticker.upper())
        if self.sp==True: print("Download feito em {0:.4f}s".format(time.time() - start_time))
        return erros
    
    def _get_df(self,ticker):
        """
        Gera um dataframe do ticker inputado
        """
        return pd.read_csv(filepath_or_buffer=(self.path + ticker.upper() + '.csv'),sep=',',index_col=0,parse_dates=True).dropna()
        
    def _consolidate_dfs(self, column='Adj Close'):
        """
        Retorna uma matriz com todos os tickers presentes no arquivo base, 532 líquidos
        """
        dados_original = self._get_df(self.tickers[0])[column]
        df_consolidado = pd.DataFrame(data=list(dados_original.values),columns={self.tickers[0].upper()}).set_index(dados_original.index)
        for i in range(1,self.n_ativos):
            dados_soma = self._get_df(self.tickers[i])[column]
            df_soma = pd.DataFrame(data=list(dados_soma.values),columns={self.tickers[i].upper()}).set_index(dados_soma.index)
            df_consolidado = pd.merge(df_consolidado,df_soma,on='Date')
        return df_consolidado
        
    def _all_tickers(self,file='acoes_momentum.txt'):
        """
        Retorna uma matriz com todos os tickers presentes no arquivo base, 532 líquidos
        """
        f = open(file,'r')
        ativos = [line.replace('\n','') for line in f.readlines()]
        f.close()
        return ativos
