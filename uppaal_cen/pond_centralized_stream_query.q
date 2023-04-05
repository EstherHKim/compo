strategy opt = minE (2*st_c + w1 + w2 + w3) [<=4*60]: <> (t==240.0)

simulate [<=60+1; 1] { t,rain,S_UC,w1,c1,Open1,o1,Rain.rainLoc,st_w,st_c,st_o,w2,Open2,o2,c2,w3,Open3,o3,c3 } under opt
