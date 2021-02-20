import numpy as np

def Generate_flow(traj_line,drop,W=38,H=36):
    in_traj_list,out_traj_list={},{}
    X=[[] for x in range(5)]
    for name,data in traj_line.iterrows():
        inflow=eval(data['trajectory_inflow'])
        outflow=eval(data['trajectory_outflow'])
        for m,(ins,out) in enumerate(zip(inflow,outflow)):
            Flows=np.zeros([2,W,H])
            for keys,values in ins.items():
                if keys not in drop:
                    if keys not in in_traj_list.keys():in_traj_list[keys]=[]
                    in_traj_list[keys]+=[{name:m}]
                    for grid in values:
                        if grid!=-1:
                            x,y=int(grid/H),int(grid%H)
                            Flows[0,x,y]+=1
            for keys,values in out.items():
                if keys not in drop:
                    if keys not in out_traj_list.keys():out_traj_list[keys]=[]
                    out_traj_list[keys]+=[{name:m}]
                    for grid in values:
                        if grid!=-1:
                            x,y=int(grid/H),int(grid%H)
                            Flows[1,x,y]+=1
            X[name].append(Flows)
    #X=[np.concatenate(x,axis=0) for x in X]
    traj_list={'inflow':in_traj_list,'outflow':out_traj_list}
    return X,traj_list