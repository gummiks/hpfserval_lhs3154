## module cubicSpline
''' k = curvatures(xData,yData).
    Returns the curvatures of cubic spline at its knots.

    y = evalSpline(xData,yData,k,x).
    Evaluates cubic spline at x. The curvatures k can be
    computed with the function 'curvatures'.
From:
    http://www.dur.ac.uk/physics.astrolab/py_source/kiusalaas/v1_with_numpy/cubicSpline.py
Example
from numpy import *
import cubicSpline
x=arange(10)
y=sin(x)
xx=arange(9)+0.5

k=cubicSpline.curvatures(x,y)
yy=cubicSpline.evalSpline(x,y,k,xx)
yy
'''
#from numarray import zeros,ones,Float64,array
import numpy
import LUdecomp3
from numpy import logical_and, asarray,zeros_like,floor,append
from numpy.core.umath import sqrt, exp, greater, less, cos, add, sin, \
     less_equal, greater_equal
import spl_int


def spl_c(x,y):
   '''
   spline curvature coefficients
   input: x,y data knots
   returns: tuple (datax, datay, curvature coeffients at knots)
   '''
   n = numpy.size(x)
   if numpy.size(y)==n:
      return spl_int.spl_int(x,y,n)


def spl_ev(xx,xyk):
   '''
   evalutes spline at xx
   '''
   #import pdb; pdb.set_trace()
   #x,y,k=xyk
   #return spl_int.spl_ev(x,y,k,numpy.size(x),xx,numpy.size(xx))
   return spl_int.spl_ev(xyk[0],xyk[1],xyk[2],numpy.size(xyk[0]),xx,numpy.size(xx))

def spl_cf(x,y):
   '''
   USING
   returns a xyk tuple with x,y, y', y'', y

   INPUT:
   - x - input x array
   - y - input y array

   OUTPUT
   A tuple with the following arrays:
   - x   - original x array
   - y   - original y array
   - y'  - first derivative
   - y'' - second derivative
   - y   - ????

   EXAMPLE:
    x = np.linspace(0,10,100)
    y = x**3.
    plt.plot(x,y)
    x0, y0, y1, y2, y3 = spline_cv(x,y)
    plt.plot(x,y)
    plt.plot(x0,y0)
    plt.plot(x0[1:],y1)
    plt.plot(x0,y2)
    plt.plot(x0[1:],y3)
   
   NOTES:
   - See also spl_evf
   '''
   n = numpy.size(x)
   if numpy.size(y)==n:
      return spl_int.spl_intf(x,y,n)

def spl_evf(xx,xyk):
   """
   Evaluate the spline at locations xx

   INPUT:
   xx - input array
   xyk - the output 5-tuple from spl_cf()

   OUTPUT:
   yy - the y values evaluated at xx

   EXAMPLE:
    x = np.linspace(0,10,100)
    y = x**3.
    plt.plot(x,y)
    x0, y0, y1, y2, y3 = spline_cv(x,y)

    xx = np.linspace(0,10,1000)
    yy = spline_ev(xx,(x0,y0,y1,y2,y3))
    plt.plot(xx,yy)
   """
   return spl_int.spl_evf(xyk[0],xyk[1],xyk[2],xyk[3],xyk[4],numpy.size(xyk[0]),xx,numpy.size(xx))



def spl_eq_c(x,y):
   '''
   spline curvature coefficients
   input: x,y data knots
   returns: tuple (datax, datay, curvature coeffients at knots)
   '''
   n = numpy.size(x)
   if numpy.size(y)==n:
      return spl_int.spl_eq_int(x,y,n)

def spl_eq_ev(xx,xyk):
   '''
   evalutes spline at xx
   xyk tuple with data knots and curvature from spl_c
   '''
   return spl_int.spl_eq_ev(xyk[0],xyk[1],xyk[2],numpy.size(xyk[0]),xx,numpy.size(xx))


def curva(x,y):
   # faster
   # Compute the hi and bi
   # h,b = ( x[i+1]-xi,y[i+1]-y[i] for i,xi in enumerate(x[:-1]))
   x = x[1:] - x[:-1]  # =h =dx
   y = (y[1:] - y[:-1]) / x # =b =dy
   # Gaussian Elimination
   #u[1:],v[1:] = (u for a,b in zip(u,v) )
   u = 2*(x[:-1] + x[1:])  # l        #c
   v = 6*(y[1:] - y[:-1])  # alpha    #d
   #for i,h in enumerate(x[1:-1]):
     #u[i+1] -= h**2/u[i]
     #v[i+1] -= h*v[i]/u[i]
   u[1:] -= x[1:-1]**2/u[:-1]
   v[1:] -= x[1:-1]*v[:-1]/u[:-1]
   # Back-substitution
   y[:-1] = v/u
   y[-1] = 0
   #for i in range(len(y[1:])):
      #y[-i-2] -= (x[-i-1]*y[-i-1]/u[-i-1])
   y[-2::-1] = y[-2::-1]-(x[:0:-1]*y[:0:-1]/u[::-1])
   #print y[1:0]
   return append(0,y)

def curva_slow(x,y):
   # Compute t,he hi and bi
   # h,b = ( x[i+1]-xi,y[i+1]-y[i] for i,xi in enumerate(x[:-1]))
   x = x[1:] - x[:-1]  # =h =dx
   y = (y[1:] - y[:-1]) / x # =b =dy
   # Gaussian Elimination
   #u[1:],v[1:] = (u for a,b in zip(u,v) )
   #x = 2*(x[:-1] + x[1:])   # u = 2*(h[:-1] + h[1:]) = 2* dh =2 ddx
   #y = 6*(y[:-1] - y[1:])   # v = 6*(b[:-1] - b[1:])   ~ ddy
   #u[1]=2*(x[0]+x[1])
   #v[1]=2*(y[1]-y[0])
   u = 2*(x[:-1] + x[1:])  # l        #c
   v = 6*(y[1:] - y[:-1])  # alpha    #d
   for i,h in enumerate(x[1:-1]):
     # if i==0: print 'slo_here'
      u[i+1] = u[i+1] - h**2/u[i]
      v[i+1] -= h*v[i]/u[i]
   # Back-substitution
   y[:-1] = v/u
   y[-1] = 0
   for i in range(len(y[1:])):
      y[-i-2] -= (x[-i-1]*y[-i-1]/u[-i-1])
   return append(0,y)

def curvatures(xData,yData):
    n = len(xData) - 1
    c = numpy.zeros((n),dtype=float)
    d = numpy.ones((n+1),dtype=float)
    e = numpy.zeros((n),dtype=float)
    k = numpy.zeros((n+1),dtype=float)
    c[0:n-1] = xData[0:n-1] - xData[1:n]
    d[1:n] = 2.0*(xData[0:n-1] - xData[2:n+1])
    e[1:n] = xData[1:n] - xData[2:n+1]
    k[1:n] = 6.0*(yData[0:n-1] - yData[1:n]) /c[0:n-1] \
            -6.0*(yData[1:n] - yData[2:n+1]) /e[1:n]
    LUdecomp3.LUdecomp3(c,d,e)
    LUdecomp3.LUsolve3(c,d,e,k)
    return k

def curvatures_org(xData,yData):
    n = len(xData) - 1
    c = numpy.zeros((n),dtype=float)
    d = numpy.ones((n+1),dtype=float)
    e = numpy.zeros((n),dtype=float)
    k = numpy.zeros((n+1),dtype=float)
    c[0:n-1] = xData[0:n-1] - xData[1:n]
    d[1:n] = 2.0*(xData[0:n-1] - xData[2:n+1])
    e[1:n] = xData[1:n] - xData[2:n+1]
    k[1:n] =6.0*(yData[0:n-1] - yData[1:n]) \
                 /(xData[0:n-1] - xData[1:n]) \
             -6.0*(yData[1:n] - yData[2:n+1])   \
                 /(xData[1:n] - xData[2:n+1])
    LUdecomp3.LUdecomp3(c,d,e)
    LUdecomp3.LUsolve3(c,d,e,k)
    return k


#def evalSpline_new(xData,yData,k,xx):
   #global m,iLeft,iRight
   #iLeft = 0
   #iRight = len(xData)- 1
   #m=-1
   #xn=xData[1:]
   #d=xn-xData[:-1]
   #def do(x):
    #global m,iLeft,iRight
    #m+=1
    #h = d[i]
    #A = (x - xn[i])/h
    #B = (x - xData[i])/h
    #return ((A**3 - A)*k[i] - (B**3 - B)*k[i+1])/6.0*h*h   \
         #+ (yData[i]*A - yData[i+1]*B)
   #for x in xx;while  x >= xData[iLeft] and iLeft<iRight: iLeft += 1
    #i=iLeft-1]
   #return map( lambda x: do(x),  xx)

def evalSpline_for(xData,yData,k,xx):
   # very slow
   h = xData[1]-xData[0]
   n=numpy.arange(len(xData)-1)
   for i,x in enumerate(xx):
      a = (x - (i+1))/h
      b = a+1
      #print i
      xx[i]=((a-a**3)*k[i] - (b-b**3)*k[i+1])/6.0*h*h   \
         - (yData[i]*a - yData[i+1]*b)
   return xx

def evalSpline_vec(xData,yData,k,xx):
   h = xData[1]-xData[0]
   n=numpy.arange(len(xx))
   AA = (xx - (n+1))/h
   BB = AA+1
   return ((AA-AA**3)*k[:-1] - (BB-BB**3)*k[1:])/6.0*h*h   \
         - (yData[:-1]*AA - yData[1:]*BB)

def evalSpline_gen(xData,yData,k,xx):
   # generator expression
   h = xData[1]-xData[0]
   n=numpy.arange(len(xx))
   AA = (xx - (n+1))/h
   BB = AA+1
   return (((AA[i]-AA[i]**3)*k[i] - (BB[i]-BB[i]**3)*k[i+1])*h*h/6.0  \
         - (yData[i]*AA[i] - yData[i+1]*BB[i]) for i in numpy.arange(len(xx)))


def evalSpline(xData,yData,k,xx):
   # generator expression
   # S(x)=a_i+b_i(x-x_i)+c_i(x-x_i)^2+d_i(x-x_i)^2
   # a_i=y_i
   # b_i=-h_i/6*z_(i+1)-h_i/3*z_i+ (y_(i+1)-y_i)/h_i
   # c_i=z_i/2
   # d_i= (z_(i+1)-z_i)/6/h_i
   # x=x-x_i=x-i = > S=a+x*(b+x*(c+x*d))
   h = xData[1]-xData[0]
   n=numpy.arange(len(xData)-1)
   AA = (xx - (n+1))/h
   BB = AA+1
   return ( yData[i]+x*( -k[i+1]/6. - k[i]/3 + (yData[i+1]-yData[i]) + x*(k[i]/2 +x *(k[i+1]-k[i])/6))  for i,x in enumerate(xx-numpy.arange(len(xx))) )

#def evalSpline(xData,yData,k,xx):
   ## generator expression
   #h = xData[1]-xData[0]
   #n=numpy.arange(len(xData)-1)
   #AA = (xx - (n+1))/h
   #BB = AA+1
   #return (((a-a**3)*k0 - (b-b**3)*k1)/6.0*h*h   \
         #- (y0*a - y1*b) for a,b,k0,k1,y,y1 in zip(AA,BB,k[:-1],k[1:],yData[:.1],yData[1:]))


def evalSpline_old2(xData,yData,k,xx):
   y = numpy.empty_like(xx)
   iLeft = 0
   iRight = len(xData)- 1
   m=-1
   for x in xx:
    m+=1
    while  x >= xData[iLeft] and iLeft<iRight: iLeft += 1
    i=iLeft-1
    h = xData[i] - xData[i+1]
    A = (x - xData[i+1])/h
    B = (x - xData[i])/h
    y[m]= ((A**3 - A)*k[i] - (B**3 - B)*k[i+1])/6.0*h*h   \
         + (yData[i]*A - yData[i+1]*B)
   return y

   

def evalSpline_old(xData,yData,k,xx):

   def findSegment(xData,x,i):
        iLeft = i
        iRight = len(xData)- 1
        while 1:    #Bisection
            if (iRight-iLeft) <= 1: return iLeft
            i =(iLeft + iRight)/2
            if x < xData[i]: iRight = i
            else: iLeft = i
   yy = []
   i = 0
   for x in xx:
    i = findSegment(xData,x,i)
    h = xData[i] - xData[i+1]
    y = ((x - xData[i+1])**3/h - (x - xData[i+1])*h)*k[i]/6.0 \
      - ((x - xData[i])**3/h - (x - xData[i])*h)*k[i+1]/6.0   \
      + (yData[i]*(x - xData[i+1])                            \
       - yData[i+1]*(x - xData[i]))/h
    yy.append(y)
    if i<10: print(i,y,x, k[i],x - xData[i])
   return numpy.array(yy)



def cubic(x):
    """A cubic B-spline.

This is a special case of `bspline`, and equivalent to ``bspline(x, 3)``.
"""
    ax = abs(asarray(x))
    res = zeros_like(ax)
    cond1 = less(ax, 1)
    if cond1.any():
        ax1 = ax[cond1]
        res[cond1] = 2.0 / 3 - 1.0 / 2 * ax1 ** 2 * (2 - ax1)
    cond2 = ~cond1 & less(ax, 2)
    if cond2.any():
        ax2 = ax[cond2]
        res[cond2] = 1.0 / 6 * (2 - ax2) ** 3
    return res



def csp_eval(cj, newx, dx=1.0, x0=0):
    """Evaluate a spline at the new set of points.

`dx` is the old sample-spacing while `x0` was the old origin. In
other-words the old-sample points (knot-points) for which the `cj`
represent spline coefficients were at equally-spaced points of:

oldx = x0 + j*dx j=0...N-1, with N=len(cj)

Edges are handled using mirror-symmetric boundary conditions.

"""
    newx = (asarray(newx) - x0) / float(dx)
    res = zeros_like(newx)
    if res.size == 0:
        return res
    N = len(cj)
    cond1 = newx < 0
    cond2 = newx > (N - 1)
    cond3 = ~(cond1 | cond2)
    # handle general mirror-symmetry
    res[cond1] = csp_eval(cj, -newx[cond1])
    res[cond2] = csp_eval(cj, 2 * (N - 1) - newx[cond2])
    newx = newx[cond3]
    if newx.size == 0:
        return res
    result = zeros_like(newx)
    jlower = floor(newx - 2).astype(int) + 1
    for i in range(4):
        thisj = jlower + i
        indj = thisj.clip(0, N - 1) # handle edge cases
        result += cj[indj] * cubic(newx - thisj)
    res[cond3] = result
    return res
