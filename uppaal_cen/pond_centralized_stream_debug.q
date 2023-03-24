//This file was generated from (Academic) UPPAAL 4.1.20-stratego-11 (rev. 323D987339A98B21), December 2022

/*

*/
// Random control

/*
Set in options -> statistical parameters the 'discretization step for hybrid systems' to 
0.5 to speed up calculation.
*/
//simulate 10 [<=72*h] {w, t}

/*
Set in options -> statistical parameters the 'discretization step for hybrid systems' to 
0.5 to speed up calculation.
*/
//E[<=72*h; 100](max:o)

/*

*/
//NO_QUERY

/*

*/
// Optimal control

/*
Notice that with o being monotonically increasing, the []o <= 0 equals <>o <= 0. Therefore, we can combine the synthesis 
of a safe and optimal strategy a single query.
Set in options -> statistical parameters the 'discretization step for hybrid systems' to 
0.5 to speed up calculation.
Set in options -> learning parameters the first four parameters to 40, 100, 20, 20 (in that order) 
to speed up the calculation.
3000 runs in approximately 2.5 minutes.
*/
strategy opt = minE (2*st_c + c1 + c2 + c3) [<=72*h]: <> (t==72*h)

/*

*/
E[<=72*h; 1000] (max:st_o) under opt

/*
Approximately 2480.
Set in options -> statistical parameters the 'discretization step for hybrid systems' to 
0.5 to speed up calculation.
*/
E[<=72*h; 1000] (max:st_c) under opt

/*

*/
E[<=72*h; 1000] (max:c1) under opt

/*

*/
E[<=72*h; 1000] (max:c2) under opt

/*

*/
E[<=72*h; 1000] (max:c3) under opt

/*
Set in options -> statistical parameters the 'discretization step for hybrid systems' to 
0.5 to speed up calculation.
*/
simulate [<=72*h;10] {st_w, w1, w2, w3} under opt
