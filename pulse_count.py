import cv2
import numpy as np

def delta(line1, line2):
    x = line1.shape[0]
    print("len", x)
    kmin = 100000
    dmin = 0
    for delta in range(5):
        print("delta:", delta, x-delta)
        print(line1.shape)
        d1 = line1[delta:(x-delta), :, :]
        print("d1shape", d1.shape, d1[0])
        len1 = x - 2  * delta
        for delta2 in range(5):
            print(delta2, delta2 + len1)
            if delta2 + len1 > x :
                break
            d2 = line2[delta2 : delta2 + len1, :, :]
            print(d2.shape,d2[0])
            diff = np.abs(d2 - d1)
            print(diff[0])
            k = np.sum(diff)/np.size(diff)
            print("diff:",k)
            if k < kmin:
                kmin = k
                dmin = delta2
            #return k
        for delta2 in range(5):
            #delta2 = -delta2
            
            print(x - delta2 - len1, x - delta2)
            if x - delta2 - len1 < 0 :
                break
            d2 = line2[x - delta2 - len1 : x - delta2, :, :]
            print(d2.shape,d2[0])
            diff = np.abs(d2 - d1)
            print(diff[0])
            k = np.sum(diff)/np.size(diff)
            print("diff:",k)
            if k < kmin:
                kmin = k
                dmin = - delta2
            #return k
        #return dmin
    return dmin

if __name__ == "__main__":
    image = "webcams2.webmpulse1.png"
    m1 = cv2.imread(image)
    dlist = []
    l1 = int(m1.shape[0] / 2)
    for k in range(m1.shape[1]):
        line1 = m1[:l1,k:k+1,:]
        line2 = m1[:l1,k+1:k+2,:]
        d1 = delta(line1, line2)
        print(d1)
        dlist.append(d1)
    print(dlist)

    m2 = np.zeros((50,len(dlist)+1,3),np.uint8)
    t0 = 25
    idx = 0
    cv2.circle(m2,(idx,t0),3,(0,0,255),-1)

    for dt in dlist:
        if abs(dt) > 5 :
            t0 = 25
            continue
        t0 = t0 + dt
        if t0 > m2.shape[0] :
            t0 = 25
            continue
        idx = idx + 1
        cv2.circle(m2,(idx*3,t0),3,(0,0,255))
        print(t0, end=" ")
    cv2.imwrite("pulse_sin1.jpg", m2)



