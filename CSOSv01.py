import numpy as np
from sklearn.cluster import KMeans

def get_problem_info(func):
    func = func.lower()
    # output format: Lb, Ub, globalMin
    switcher={
            'zakharov'          : [-5, 10, 0],
            'schumer_steiglitz' : [-100, 100, 0],
            'powell_sum'        : [-100, 100, 0],
            'sum_squares'       : [-10, 10, 0],
            'bent_cigar'        : [-100, 100, 0],
            'schwefel_12'      : [-100, 100, 0],
            'chung_reynolds'   : [-100, 100, 0],
            'brown'             : [-1, 4, 0],
            'alpine_1'          : [-10, 10, 0],
            'rastrigin'         : [-5.12, 5.12, 0],
            'csendes'           : [-1, 1, 0],
            'quartic'           : [-1.28, 1.28, 0],
            'exponential'       : [-1, 1, -1],
            'griewank'          : [-600, 600, 0],
            'ackley_1'          : [-32, 32, 0],
         }
    return switcher.get(func,"invalid function name")

def calc_obj (func):
    func = func.lower()
    switcher={
            'zakharov'          : zakharov,
            'schumer_steiglitz' : schumer_steiglitz,
            'powell_sum'        : powell_sum,
            'sum_squares'       : sum_squares,
            'bent_cigar'        : bent_cigar,
            'schwefel_12'       : schwefel_12,
            'chung_reynolds'   : chung_reynolds,
            'brown'             : brown,
            'alpine_1'          : alpine_1,
            'rastrigin'         : rastrigin,
            'csendes'           : csendes,
            'quartic'           : quartic,
            'exponential'       : exponential,
            'griewank'          : griewank,
            'ackley_1'          : ackley_1,
         }
    return switcher.get(func,"invalid function name")

def zakharov(dat):
    sum1, sum2, sum3 = 0, 0, 0
    for i in range( len(dat) ):
        sum1 += dat[i]**2
        sum2 += 0.5*(i+1)*dat[i]
        sum3 += 0.5*(i+1)*dat[i]
    y = sum1+ (sum2)**2 + (sum3)**4
    return y

def schumer_steiglitz(dat):
    sum1 = 0
    for i in range( len(dat) ):
        sum1 += (dat[i])**4
    y = sum1
    return y

def powell_sum(dat):
    sum1 = 0
    for i in range( len(dat) ):
        sum1 += abs(dat[i])**(i+1)
    y = sum1
    return y

def sum_squares(dat):
    sum1 = 0
    for i in range( len(dat) ):
        sum1 += (i+1)*(dat[i])**2
    y = sum1
    return y

def bent_cigar(dat):
    sum1, sum2 = 0, 0
    for i in range( len(dat) ):
        sum1 += (dat[i])**2
        for j in range(1, len(dat)):
            sum2 += (dat[j])**2
    y = sum1 + sum2*10e6
    return y

def schwefel_12(dat):
    sum1, sum2 = 0, 0
    for i in range( len(dat) ):
        for j in range(0, i) :
            sum2 += dat[i]
        sum1 += sum2**2
        sum2 = 0
    y = sum1
    return y

def chung_reynolds(dat):
    sum1 = 0
    for i in range(len(dat)):
        sum1 += dat[i]**2
    y = sum1**2
    return y

def brown(dat):
    sum1 = 0
    for i in range((len(dat)-1)):
        sum1 += (dat[i]**2)**(1+(dat[i+1]**2)) + (dat[i+1]**2)**(1+(dat[i]**2))
    y = sum1
    return y

def alpine_1(dat):
    sum1 = 0
    for i in range(len(dat)):
        sum1 += dat[i]*(np.sin(dat[i])) + 0.1*dat[i]
    y = abs(sum1)
    return y

def rastrigin(dat):
    sum1 = 0
    for i in range(len(dat)):
        sum1 += (dat[i]**2 - 10*(np.cos(2*(np.pi)*dat[i])) +10)
    y = sum1
    return y
    
def csendes(dat):
    sum1 = 0
    for i in range(len(dat)):
        sum1 += (dat[i]**6) * (2 + np.sin(1/(dat[i]+1e-12)))
    y = sum1
    return y

def quartic(dat):
    sum1 = 0
    for i in range(len(dat)):
        sum1 += (i+1)*((dat[i])**4)
    y = sum1 + np.floor( np.random.uniform(0,1) )
    return y

def exponential(dat):
    sum1 = 0
    for i in range(len(dat)):
        sum1 += dat[i]**2
    y = - np.exp(-0.5*(sum1))
    return y

def griewank(dat):
    sum1, sum2 = 0, 1
    for i in range(len(dat)):
        sum1 += dat[i]**2
        sum2 *= np.cos( dat[i]/np.sqrt(i+1) )
    y = 1 + (sum1/4000) - sum2
    return y

def ackley_1(dat):
    sum1, sum2 = 0, 0
    n = len(dat)
    for i in range(len(dat)):
        sum1 += dat[i]**2
        sum2 += np.cos(2*np.pi*(i+1))
    y = -20*np.exp(-0.2*np.sqrt(sum1/n)) - np.exp(sum2/n) + 20 + np.e
    return y

def solve(n_population, MaxIter, Tol, function, nd, Lb, Ub, GM):
    Xhist = {}
    X = np.random.uniform( low=Lb, high=Ub, size=(nd*n_population) ).reshape((n_population, nd))
    X_val = np.apply_along_axis(calc_obj(function), 0, X.T)
    kmeans = KMeans(n_clusters = np.int(n_population/2), random_state=0).fit(X)
    X_id = kmeans.labels_

    # initial clustering  
    while True:
        uniqueIDs, memberCount = np.unique(X_id, return_counts=True)
        if np.min(memberCount)<=1:
            candidate = uniqueIDs[np.where(memberCount<=1)]
            selected  = np.random.choice(candidate, 1, replace=False)[0]
            centroid = np.vstack(np.array([(i, X[X_id==i].mean(axis=0)) for i in uniqueIDs]).T[1])

            cc = centroid[uniqueIDs==selected]
            odr = np.argsort(np.sum(np.square(cc-centroid), axis=1))[:2][1]
            X_id[X_id==selected] = uniqueIDs[odr]
        else:
            break
    uniqueIDs, memberCount = np.unique(X_id, return_counts=True)
    temp = [(i, X[X_id==i].mean(axis=0)) for i in uniqueIDs]
    c_center = np.vstack(np.array(temp).T[1])

    # merging
    while True:
        c1, c2, d = [], [], []
        for i in range(0, len(uniqueIDs)-1):
            for j in range(i+1, len(uniqueIDs)):
                c1 = np.append(c1, uniqueIDs[i])
                c2 = np.append(c2, uniqueIDs[j])
                d = np.append(d, np.sqrt(np.sum(np.square(c_center[i]-c_center[j]))))
        if np.min(d) <= np.mean(d)/2 :
            loc = np.where(d==np.min(d))
            X_id[X_id==c1[loc][0]] = np.int(c2[loc][0])

            uniqueIDs, memberCount = np.unique(X_id, return_counts=True)
            temp = [(i, X[X_id==i].mean(axis=0)) for i in uniqueIDs]
            c_center = np.vstack(np.array(temp).T[1])
        else :
            old_cid = np.array([])
            new_cid = np.array([])
            odr = 1
            for i in range(n_population):
                x = X_id[i]
                if np.isin(x, old_cid) == 0:
                    old_cid = np.append(old_cid, x)
                    new_cid = np.append(new_cid, odr)
                    odr +=1
                if np.isin(x, old_cid) == 1:
                    X_id[i] = new_cid[old_cid==x][0]

            uniqueIDs, memberCount = np.unique(X_id, return_counts=True)
            temp = [(i, X[X_id==i].mean(axis=0)) for i in uniqueIDs]
            c_center = np.vstack(np.array(temp).T[1])
            break

    FE = 0
    for mx in range(MaxIter):
        if np.min(X_val) <= GM+Tol :
            Xhist[mx] = X_val.copy()
            break
        Xhist[mx] = X_val.copy()
        for c in uniqueIDs:
            if np.min(X_val) <= GM+Tol :
                break
            memberCluster = np.where(X_id==c)[0]
            for i in memberCluster:
                if np.min(X_val) <= GM+Tol :
                    break
                # mutualism
                localBest_id = memberCluster[np.argmin(X_val[memberCluster])]
                localBest_X = X[localBest_id]
                j = np.random.choice(memberCluster[memberCluster!=i], 1)[0] # selected solutions for interaction
                MV = np.mean(X[[i,j]], axis=0)
                BF = np.random.choice(2, 2, replace=False) + 1
                new_X = X[[i,j]] + np.random.rand(1)[0]*(localBest_X-MV*BF[:, None])
                new_val = np.apply_along_axis(calc_obj(function), 0, new_X.T)
                if new_val[0] < X_val[i] :
                    X[i] = new_X[0]
                    X_val[i] = new_val[0]
                if new_val[1] < X_val[j] :
                    X[j] = new_X[1]
                    X_val[j] = new_val[1]
                FE += 2

                # commensalism
                if np.min(X_val) <= GM+Tol :
                    break
                localBest_id = memberCluster[np.argmin(X_val[memberCluster])]
                localBest_X = X[localBest_id]
                j = np.random.choice(memberCluster[memberCluster!=i], 1)[0] # selected solutions for interaction
                new_X = X[i] + np.random.rand(1)[0]*(-1)*(localBest_X-X[j])
                new_val = np.apply_along_axis(calc_obj(function), 0, new_X.T)
                if new_val < X_val[i] :
                    X[i] = new_X
                    X_val[i] = new_val
                FE += 1

            # parasitism
            if np.min(X_val) <= GM+Tol :
                break
            memberCluster_c = np.where(X_id==c)[0]
            localBest_id_c = memberCluster_c[np.argmin(X_val[memberCluster_c])]
            localBest_X_c = X[localBest_id_c]

            d = np.random.choice(uniqueIDs[uniqueIDs!=c], 1)[0]
            memberCluster_d = np.where(X_id==d)[0]
            localBest_id_d = memberCluster_d[np.argmin(X_val[memberCluster_d])]
            localBest_X_d = X[localBest_id_d]

            new_X_d = localBest_X_c*np.random.randint(2, size=nd) + np.random.uniform(low=Lb, high=Ub, size=nd)
            new_val_d = np.apply_along_axis(calc_obj(function), 0, new_X_d.T)

            if new_val_d < X_val[localBest_id_d]:
                X[localBest_id_d] = new_X_d
                X_val[localBest_id_d] = new_val_d
            FE += 1
        
    return(Xhist, FE, mx)