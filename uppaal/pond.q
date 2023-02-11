strategy opt = minE (wmax) [<=4*60]: <> (t==240.0)

simulate 1 [<=60+1] { t,rain,S_UC,w,c,Open,o,Rain.rainLoc } under opt
