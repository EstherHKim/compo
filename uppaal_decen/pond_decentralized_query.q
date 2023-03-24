strategy opt = minE (wmax) [<=4*60]: <> (t==240.0)

simulate 1 [<=60+1] { t,rain,S_UC,w,c,Open,o,Rain.rainLoc,st_w,st_c,st_o } under opt
