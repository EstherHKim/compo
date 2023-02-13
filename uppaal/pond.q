strategy opt = minE (c) [<=2*60]: <> (t==120.0)

simulate [<=60+1; 1] { t,rain,S_UC,w,c,Open,o,Rain.rainLoc } under opt
