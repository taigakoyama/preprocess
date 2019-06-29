# preprocess
機械学習でよく使う前処理をまとめます。

addXXX → dataframeにXXXを追加します。
getXXX → XXXをgetします。
showXXX → XXXをshowします。

# modules
使っているモジュールは下記の通り

import re
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statistics import mean, median,variance,stdev
from yellowbrick.target import BalancedBinningReference
from yellowbrick.features import Rank1D,Rank2D
