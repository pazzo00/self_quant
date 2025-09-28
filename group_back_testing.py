import pandas as pd
import bisect
from datetime import timedelta
from joblib import Parallel, delayed
import re


class GroupBackTesting:
    def __init__(self, data, signal, bins=5, weights=None):
        self.data = data
        self.trade_days = data.index.tolist()
        self.signal = signal
        self.bins = bins
        self.weights = weights

    def __call__(self):
        # group_ret = []
        # for i in range(len(self.signal)):
        #     group_ret.append(self.cal_period_pnl(i))

        group_ret = Parallel(n_jobs=16, verbose=10)(
            delayed(self.cal_period_pnl)(idx)
            for idx in range(len(self.signal))
        )

        group_ret = pd.concat(group_ret)
        group_ret.set_index('datetime', inplace=True)
        group_ret = (1 + group_ret).cumprod()
        return group_ret

    def cal_period_pnl(self, idx):
        signal = self.signal.iloc[idx].dropna()
        signal = signal.sort_values()
        total = len(signal)
        start_time = signal.name
        start_idx = bisect.bisect_left(self.trade_days, start_time)

        daily_pnl = []
        index_day = []
        if idx == 0:
            daily_pnl.append([0] * self.bins)
            index_day.append(start_time)

        if idx < len(self.signal) - 1:
            end_time = self.signal.iloc[idx+1].name
            end_idx = bisect.bisect_right(self.trade_days, end_time)
        else:
            end_idx = len(self.trade_days)

        for i in range(start_idx + 1, end_idx):
            index_day.append(self.trade_days[i])
            tmp_pnl = []
            tmp_rtn = self.data.iloc[i]
            start = 0
            for end_pos in range(1, self.bins + 1, 1):
                end = int(total * end_pos / self.bins)
                tmp_codes = signal.iloc[start: end].index.tolist()
                start = end
                if self.weights is not None:
                    weight = self.weights[end_pos-1].iloc[idx]
                    tmp_signal = tmp_rtn.reindex(tmp_codes).fillna(0) * weight.reindex(tmp_codes).fillna(0)
                else:
                    tmp_signal = tmp_rtn.reindex(tmp_codes).mean()
                tmp_pnl.append(tmp_signal.sum())
            daily_pnl.append(tmp_pnl)
        daily_pnl = pd.DataFrame(daily_pnl, columns=['group_{}'.format(_) for _ in range(self.bins)])
        daily_pnl['datetime'] = index_day
        return daily_pnl


if __name__ == '__main__':
    x = [1,2,3,4,5,6,10]
    print(bisect.bisect_right(x, 2))
