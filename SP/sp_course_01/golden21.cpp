/*  PES   */
#include "mlisp.h"
extern double a/*2*/ ;
	 extern double b/*2*/ ;
	 double fun/*3*/ (double x);
	 double golden__section__search/*7*/ (double a, double b);
	 double golden__start/*15*/ (double a, double b);
	 extern double mphi/*25*/ ;
	 double __PES__try/*26*/ (double a, double b
	 , double xa, double ya
	 , double xb, double yb);
	 bool close__enough_Q/*44*/ (double x, double y);
	 extern double tolerance/*46*/ ;
	 extern double total__iterations/*47*/ ;
	 extern double xmin/*48*/ ;
	 //________________ 
double a/*2*/ = 2.;
	 
double b/*2*/ = 3.;
	 
double fun/*3*/ (double x){
 x = (x - (21. / 22.));
	 return
 (x - expt((x - 2.)
	 , 3.)
	  - atan(x) - 1.);
	 }

double golden__section__search/*7*/ (double a, double b){
 {
 double xmin(((a < b)
	? golden__start(a
	 , b)
	 
	: true
	? golden__start(b
	 , a)
	 
	: _infinity));
	 newline();
	 return
 xmin;
	 }
}

double golden__start/*15*/ (double a, double b){
 total__iterations = 0.;
	 {
 double xa((a + (mphi * (b - a)))),
	 xb((b + (- (mphi * (b - a)))));
	 return
 __PES__try(a
	 , b
	 , xa
	 , fun(xa)
	 , xb
	 , fun(xb))
	 ;
	 }
}

double mphi/*25*/ = ((3. - sqrt(5.)) * (1. / 2.0));
	 
double __PES__try/*26*/ (double a, double b
	 , double xa, double ya
	 , double xb, double yb){
 return
 (close__enough_Q(a, b)
	? ((a + b) * 0.5)
	: true
	? display("+"),
	 total__iterations = (total__iterations + 1.),
	 ((ya < yb)
	? b = xb,
	 xb = xa,
	 yb = ya,
	 xa = (a + (mphi * (b - a))),
	 __PES__try(a
	 , b
	 , xa
	 , fun(xa)
	 , xb
	 , yb)
	 
	: true
	? a = xa,
	 xa = xb,
	 ya = yb,
	 xb = (b - (mphi * (b - a))),
	 __PES__try(a
	 , b
	 , xa
	 , ya
	 , xb
	 , fun(xb))
	 
	: _infinity)
	: _infinity);
	 }

bool close__enough_Q/*44*/ (double x, double y){
 return (abs((x - y)) < tolerance);
	 }

double tolerance/*46*/ = .001;
	 
double total__iterations/*47*/ = 0.;
	 
double xmin/*48*/ = 0.;
	 int main(){
 display("Calculations!");
	 newline();
	 xmin = golden__section__search(a
	 , b)
	 ;
	 display("Interval=\t[");
	 display(a);
	 display(" , ");
	 display(b);
	 display("]\n");
	 display("Total number of iteranions=");
	 display(total__iterations);
	 newline();
	 display("xmin=\t\t");
	 display(xmin);
	 newline();
	 display("f(xmin)=\t");
	 display(fun(xmin));
	 newline();
	 std::cin.get();
	 return 0;
	 }

