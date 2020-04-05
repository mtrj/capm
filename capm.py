# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lib_yahoo import yahoo
from baixa_curvas import bmf
from datetime import date
from pandas.tseries.offsets import BDay

class capm:
    def __init__(self, tickers):
        self.tickers = list(tickers)
        self.shift_beta = 252
        self.bench = '^BVSP'
        if self.bench not in tickers: self.tickers.append(self.bench)
        sp = str(date.today()-BDay(1))[0:10].split('-')
        self.cdi = float(bmf(val_date=date(int(sp[0]),int(sp[1]),int(sp[2])))._baixa_pre()[:1]['taxas252'])
        
    def _betas(self, download=False, shift=252, periods=1):
        """
        Calcula os betas das ações dados o shift de dias úteis para uso de dados e o slice de períodos
        """
        obj = yahoo(self.tickers)
        if download==True:
            obj._download_files()
        self.df = obj._consolidate_dfs()[::periods]
        self.df = np.log(self.df/self.df.shift(1))
        self.avg_returns = [self.df[-shift:][self.tickers[i]].mean()*252
                            for i in range(len(self.tickers)-1)]
        self.betas = []
        for i in range(len(self.tickers)):
            if self.tickers[i]!=self.bench:
                self.betas.append(self.df[-shift:].cov()[self.bench][self.tickers[i]]/
                            self.df[-shift:][self.bench].var())
        return self.betas
    
    def kes(self, shift=252, periods=1):
        """
        Calcula os KEs das ações solicitadas
        """
        self._betas(shift=shift, periods=periods)
        kes = []
        for i in range(len(self.tickers)-1):
            kes.append(self.cdi+self.betas[i]*(self.avg_returns[i]-self.cdi))
        return kes
    
    def simulacao_betas(self, n_max=500):
        """
        Cria uma simulação dos betas com n_max períodos diferentes de corte nos dados
        diário=1
        semanal=5
        mensal=22
        """
        lista_betas = [self._betas(periods=i) for i in range(1,n_max+1)]
        colunas = [self.tickers[i] for i in range(len(self.tickers)-1)]
        dados = pd.DataFrame(data=lista_betas, columns=colunas)
        dados.index+=1
        return dados
