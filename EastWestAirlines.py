# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 17:14:35 2019

@author: Ganesh
"""

import pandas as pd

Airlines = pd.read_excel(r"F:\\R\\files\\EastWestAirlines.xlsx")

Airlines.head()

Airlines.info()

Airlines.isna().sum()


