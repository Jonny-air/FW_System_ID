import numpy as np

q = [0.687,  0.598,  0.311,  0.271]

R = np.empty([3,3])

aa = q[0]*q[0]
ab = q[0]*q[1]
ac = q[0]*q[2]
ad = q[0]*q[3]
bb = q[1]*q[1]
bc = q[1]*q[2]
bd = q[1]*q[3]
cc = q[2]*q[2]
cd = q[2]*q[3]
dd = q[3]*q[3]

roll = np.arctan2(2*(cd+ab), 1.0-2.0*(bb + cc)) #atan2(2.0 * (d * c + a * b) , 1.0 - 2.0 * (q.q1 * q.q1 + q.q2 * q.q2))
pitch = np.arcsin(2*(ac-bd))
yaw = np.arctan2(2*(bc+ad), -1.0+2.0*(aa + bb))

R[0,0] = aa+bb-cc-dd
R[0,1] = 2*(bc-ad)
R[0,2] = 2*(bd+ac)
R[1,0] = 2*(bc+ad)
R[1,1] = aa-bb+cc-dd
R[1,2] = 2*(cd-ab)
R[2,0] = 2*(bd-ac)
R[2,1] = 2*(cd+ab)
R[2,2] = aa-bb-cc+dd

vec = np.array([0.0, 1.0, 0.0])
print(R @ vec)