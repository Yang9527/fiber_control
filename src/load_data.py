import xlrd
import traceback
import numpy as np
from tf_agents.trajectories.trajectory import from_episode

def load_data(filename):
    try:
        data = xlrd.open_workbook(filename)
    except:
        print(traceback.print_exc())
    sheets = data.sheets()
    tbl = sheets[1]
    rlts = []
    rlt = []
    pred_row_values = None
    for i in range(tbl.nrows):
        # print(tbl.row_types(i))
        row_values = tbl.row_values(i)
        row_types = tbl.row_types(i)
        
        id, ts, d, v, ct, t, l = row_values
        if row_types[0] != 2:
            continue
#        print(id, ts, d, d>0, v, v>0, ct, t, l)
        if d > 0 and v > 0:
            if pred_row_values is None:
                rlt = []
                pred_row_values = row_values
            else:
                tmp_row_values = [x for x in row_values]
                row_values[3] -= pred_row_values[3]
                pred_row_values = tmp_row_values
                rlt.append(row_values)
        else:
            if rlt and pred_row_values is not None:
                rlts.append(rlt)
            pred_row_values = None
    return rlts

def experience_to_traj(rlt):
    rlt = np.array(rlt)
    d = rlt[:,2]
    v = rlt[:,3] / 0.001
    v = v.astype(np.int32)
    
    discount = np.ones_like(v) * 0.99
    policy_info=rlt[:,4]
    reward = -np.abs(d-1.22)
    traj = from_episode(
        observation=d,
        action=v,
        reward=reward,
        discount=discount,
        policy_info=policy_info
    )
    return traj

if __name__ == "__main__":
    rlts = load_data("data/玻璃棒拉丝数据.xlsx")
    trajs = [experience_to_traj(rlt) for rlt in rlts]
    for traj in trajs:
        print(traj)