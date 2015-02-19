

import numpy as np
import scipy as sp
import pylab as pp
import pdb

from scipy import special
from scipy import stats

sqrt2 = np.sqrt( 2.0 )
sqrt_2pi = np.sqrt( 2.0 * np.pi )
log_sqrt_2pi = np.sqrt(sqrt_2pi)

# import numpy as np
# import scipy as sp
# import pylab as pp
#from sobol import * 

import pdb

sqrt_2pi = np.sqrt(2.0*np.pi)

def logsumexp(x,dim=0):
    """Compute log(sum(exp(x))) in numerically stable way."""
    #xmax = x.max()
    #return xmax + log(exp(x-xmax).sum())
    if dim==0:
        xmax = x.max(0)
        return xmax + np.log(np.exp(x-xmax).sum(0))
    elif dim==1:
        xmax = x.max(1)
        return xmax + np.log(np.exp(x-xmax[:,newaxis]).sum(1))
    else: 
        raise 'dim ' + str(dim) + 'not supported'
        
class Sampler( object ):
  def __init__( self, dim, x0 ):
    self.dim = dim
    self.x   = np.zeros( self.dim, dtype = float )
    
    self.x = x0.copy()
      
  def set_state( self, x ):
    self.x = x
    
  def step( self, target ):
    raise NotImplementedError
    
  def x_not_i( self, x, i ):
    x_not_i = np.hstack( (x[:i], x[i+1:]))
    
    print x, i, x_not_i
    # return scalar if we have dim x_not_i == 1
    if len(x_not_i) == 1:
      return x_not_i[0]
    
    return x_not_i
    

class GibbsSampler( Sampler ):
  def step( self, target ):
    for i in xrange( self.dim ):
      xi = target.gibbs_rand( i, self.x_not_i( self.x, i ) )
      self.x[i] = xi
    return self.x.copy()
      
class OrderOverrelaxed( Sampler ):
  def __init__( self, dim, x0, K, S ):
    super(OrderOverrelaxed, self).__init__( dim, x0 )
    self.K = K
    
    # is S =0, then this is Overrelaxed by Neal
    self.S = S
    
    
  def step( self, target ):
    for i in xrange( self.dim ):
      xi = target.gibbs_rand( i, self.x_not_i( self.x, i ), self.K[i] )
      xi = np.hstack( (xi, self.x[i]))
      I = xi.argsort()
      r = pp.find(I==self.K[i])
      
      xi = xi[ I[self.K[i]-self.S[i]-r] ]
      
      self.x[i] = xi
      #print i, xi
    return self.x.copy()

class OrderOverrelaxedWithU( Sampler ):
  def __init__( self, dim, x0, K, S ):
    super(OrderOverrelaxedWithU, self).__init__( dim, x0 )
    self.K = K
    
    # is S =0, then this is Overrelaxed by Neal
    self.S = S
    
    self.F  = stats.norm(0,1).cdf
    self.iF = stats.norm(0,1).ppf
    
  def step( self, target ):
    u = np.zeros(self.dim)
    for i,S,K in zip( xrange( self.dim ), self.S, self.K ):
      
      m,vr = target.gibbs_post(i, self.x_not_i( self.x, i ) )
      xi = self.x[i]
      #self.F  = stats.norm( (xi-m)/np.sqrt(v))
      
      u = self.F( (xi-m)/np.sqrt(vr)  )
      #print xi, u, m, np.sqrt(v)
      #pdb.set_trace()
      r = np.random.binomial( K, u )
      
      if r > K - r:
        v = np.random.beta( K-r+1, 2*r-K )
        u = u*v
      elif r < K - r:
        v = np.random.beta( r+1, K-2*r )
        u = 1.0 - (1.0-u)*v
      else:
        pass # keep it the same
        
      self.x[i] = m + np.sqrt(vr)*self.iF( u )
      
      
    return self.x.copy()
        
def sticky_rand( pp, zz = None ):
  if zz is None:
    zz = np.random.rand();


  if np.random.rand() < (1-pp):
    zz = np.random.rand()


  uu = zz
  return uu

def norm( x ):
  return np.sqrt( np.dot(x,x) )
  
def norminv( u, mu, stdev ):
  x = sqrt2 * sp.special.erfinv( 2*u - 1.0 )
  return mu + x*stdev 
  
def normcdf( x, mu, stdev):
  u = 0.5*(1.0 + sp.special.erf( (x-mu)/(sqrt2*stdev)))
  return u
  
def log_prob( x, mu, cov ):
  d = len(x)
  iCov = np.linalg.pinv(cov)
  logprob = -d*sqrt_2pi-0.5*np.log(np.linalg.det(cov)) - 0.5*np.dot( np.dot( (x-mu), iCov ), x-mu )
  return logprob
  
def log_prob_grad_dx( x, mu, cov ):
  iCov = np.linalg.pinv(cov)
  return -np.dot( x-mu, iCov )*0.1
  
def show_1( mu, cov ):
  XRR = np.random.multivariate_normal( mu,C, 1000)
  pp.figure()
  pp.plot( XRR[:,0], XRR[:,1], 'bo', mec='k',ms=10, alpha=0.01)
  
  # x=np.linspace( -3*C[0,0], 3*C[0,0], 50)
  #   y=np.linspace( -3*C[1,1], 3*C[1,1], 50)
  
  XX,YY = pp.meshgrid( np.linspace( -3*C[0,0], 3*C[0,0], 40), np.linspace( -3*C[1,1], 3*C[1,1], 40))
  #pdb.set_trace()
  Z=[]
  dZ=[]
  dX = []
  dY = []
  for xrow, yrow in zip( XX, YY ):
    dx_row = []
    dy_row = []
    z_row  = []
    for xi,yi in zip( xrow, yrow ):
      xx = np.array( [xi,yi])
    
      zi = log_prob( xx, mu, cov )
      #sif zi > -10:
      #Z.append( zi )
      dZ.append( log_prob_grad_dx( xx, mu, cov ) )
      dxx = dZ[-1]
      dx_row.append( np.sign(dxx[0])*min( np.exp(dxx[0]),1) )
      dy_row.append( np.sign(dxx[1])*min( np.exp(dxx[1]),1))
      log_acc = min( log_prob( xx - dxx, mu, cov )-zi,0)
      z_row.append( np.exp( log_acc ) )
    dX.append( dx_row )    
    dY.append( dy_row )
    Z.append( z_row)
        #print xx, dxx
        #pp.arrow( xi, yi, dxx[0], dxx[1])
  dY = np.array( dY )
  dX = np.array( dX )
  Z = np.array( Z )
  pp.quiver( XX, YY, dX, dY )
  print Z
  levels = [0.9,0.75,0.5,0.25,0]
  cc=pp.contour(XX, YY,Z, lw=2,levels=levels)
  pp.colorbar(cc)
  pp.show()
      
  
def DetMCMC(shift,T,mux,muy,cxx,cyy,cxy,u_init):

    # %
    # % S = DetMCMCgauss(shift,T,mux,muy,cxx,cyy,cxy,perfect_samples)
    # %
    # %Input: 
    # % shift: shift vecor for the uniform variante u on a circle
    # % T: nr. of iterations of algorithm
    # % mux: mean of gaussian density in x direction
    # % muy: mean of gaussian density in y direction
    # % cxx: covariance of xx-component of gaussian density
    # % cyy: covariance of yy-component of gaussian density
    # % cxy: covariance of xy-component of gaussian density
    # % perfect_samples: 1 if you want IID samples
    # %Output:
    # % S: 2xT matrix with samples 

    # % ---initial values x,y,u,S
    x = np.random.randn()
    y = np.random.randn()
    u = u_init
    #u = np.random.rand()

    zz = None
    pp = 0.1
    zz = sticky_rand(pp, zz )

    S = np.zeros((2,T))
    R = np.zeros(T)
    U = [u]

    shifts = [np.pi, np.pi]

    plotT = 1000 #% plot every 100 samples

    s = rand_sobol( 1000, 1, np.int(np.random.rand()*100 ))
    si = 0

    for t in range(T):


        #% -- compute conditional mean and covariance x | y and update x,u
        mu_xgy = mux + (cxy / cyy) * (y - muy) #%conditional mean
        # mu_xgy2 = mux + (cxy / cyy) * (y2 - muy)

        sg_xgy = np.sqrt(cxx - (cxy**2)/cyy) #%conditional sigma
        #sg_xgy = np.sqrt( cxx - (cxy**2)/cyy ) #%conditional covariance
        ii = np.int(np.mod(t,2))
        u = np.mod( normcdf( x, mu_xgy, sg_xgy) + shifts[ii], 1) #%update for u. Note shift and modulus.
        zz = sticky_rand( pp, zz )
        u = np.mod( u + zz, 1) #%update for u. Note shift and modulus.
        #u = np.mod( u + shifts[ii], 1) #%update for u. Note shift and modulus.
        x_new = norminv(u,mu_xgy,sg_xgy) #% update for x

        #u = np.mod( u + shift, 1) #%update for u. Note shift and modulus.
        #u = np.mod( s[si] + shift, 1 )
        #u = 1.0 - u
        #u = s[si]
        si+=1
        U.append(u)



        x = x_new;
        #% -- compute conditional mean and covariance y | x and update y,u
        mu_ygx = muy + (cxy / cxx) * (x - mux); #% conditional mean

        #sg_ygx = np.sqrt( cyy - (cxy**2)/cxx ); #% conditional covariance
        sg_ygx = np.sqrt(cyy - (cxy**2)/cxx) #% conditional sigma

        #v1 = s[si]
        #si+=1
        y_new = norminv(u,mu_ygx,sg_ygx); #% update for y

        u = normcdf( y,mu_ygx,sg_ygx ) 

        #u = np.mod( normcdf( y, mu_xgy, sg_xgy) + shift, 1)

        #u = np.mod( np.random.rand()*0.1 + s[si] + shift, 1 )
        #si+=1

        #U.append(u)

        y = y_new

        S[:,t] = np.array([x,y]) #%record sample



    return S, np.array(U)
      
def DetMCMCgauss4d(shift,T,mux,muy,cxx,cyy,cxy,u_init):

  # %
  # % S = DetMCMCgauss(shift,T,mux,muy,cxx,cyy,cxy,perfect_samples)
  # %
  # %Input: 
  # % shift: shift vecor for the uniform variante u on a circle
  # % T: nr. of iterations of algorithm
  # % mux: mean of gaussian density in x direction
  # % muy: mean of gaussian density in y direction
  # % cxx: covariance of xx-component of gaussian density
  # % cyy: covariance of yy-component of gaussian density
  # % cxy: covariance of xy-component of gaussian density
  # % perfect_samples: 1 if you want IID samples
  # %Output:
  # % S: 2xT matrix with samples 

  # % ---initial values x,y,u,S
  x = np.random.randn()
  y = np.random.randn()
  u = u_init
  #u = np.random.rand()

  zz = None
  pp = 0.9
  zz = sticky_rand(pp, zz )

  S = np.zeros((2,T))
  R = np.zeros(T)
  U = [u]
  
  shifts = [np.pi, np.pi]
  shifts = [0.17,0.17]

  plotT = 100 #% plot every 100 samples

  s = rand_sobol( 1000, 1, np.int(np.random.rand()*100 ))
  s = sticky_sobol( 1000, 1, np.int(np.random.rand()*100 ), pp)
  si = 0
  us =[u,u]
  for t in range(T):
      
      
      #% -- compute conditional mean and covariance x | y and update x,u
      mu_xgy = mux + (cxy / cyy) * (y - muy) #%conditional mean
      # mu_xgy2 = mux + (cxy / cyy) * (y2 - muy)
    
      sg_xgy = np.sqrt(cxx - (cxy**2)/cyy)  #%conditional sigma
      ii = np.int(np.mod(t,2))
      #u = np.mod( normcdf( x, mu_xgy, sg_xgy) + shifts[ii], 1) #%update for u. Note shift and modulus.
      zz = sticky_rand( pp, zz )
      #u = np.mod( u + zz, 1) #%update for u. Note shift and modulus.
      #u = np.mod( u + shifts[ii], 1) #%update for u. Note shift and modulus.
      print "\t u = " + str(us[0]) + " sobol " + str(s[si] )
      
      #us[0] = np.random.rand()
      
      us[0] = np.mod( us[0] + s[si] + 1*shifts[0], 1 )
      x_new = norminv(us[0],mu_xgy,sg_xgy) #% update for x
      print "x: " + str(x) + " to " + str(x_new) + " using u = " + str(us[0])
      
      #u = np.mod( u + shift, 1) #%update for u. Note shift and modulus.
      
      #u = 1.0 - u
      #u = s[si]
      si+=1
      
      
      us[0] = normcdf( x, mu_xgy, sg_xgy ) 
      

      x = x_new;
      #% -- compute conditional mean and covariance y | x and update y,u
      mu_ygx = muy + (cxy / cxx) * (x - mux); #% conditional mean
    
      sg_ygx = np.sqrt(cyy - (cxy**2)/cxx)  #% conditional sigma
    
      #v1 = s[si]
      #si+=1
      us[0] = np.mod( us[0] + s[si] + 1*shifts[0], 1 )
      #us[0] = np.random.rand()
      y_new = norminv(us[0],mu_ygx,sg_ygx); #% update for y
      print "y: " + str(y) + " to " + str(y_new) + " using u = " + str(us[0])
    
      us[0] = normcdf( y,mu_ygx,sg_ygx ) 
      
      si+=1
      #u = np.mod( normcdf( y, mu_xgy, sg_xgy) + shift, 1)
      
      #u = np.mod( np.random.rand()*0.1 + s[si] + shift, 1 )
      #si+=1
      
      #U.append(u)
      U.append(us[0])
      y = y_new
     
      S[:,t] = np.array([x,y]) #%record sample
    
      
          
  return S, np.array(U)
        
# if __name__ == "__main__":
#   #pp.close('all')
#
#   shift =  np.mod( np.sqrt(2.0), 1.0 )
#   T = 50
#   mux = 0
#   muy = 0
#   cxx = 1.0
#   cyy = 1.0
#   cxy = 0.998
#   mu= np.array( [mux,muy])
#
#   u_init = 0.02 #np.pi/4
#
#   seed = np.int(np.random.rand()*1000)
#
#   C = np.array( [[cxx,cxy],[cxy,cyy]])
#   XR = multi_randn( 1000, C, seed )
#   XS = multi_sobol( 1000, C, seed )
#   XRR = np.random.multivariate_normal( [mux,muy],C, 1000)
#
#   S,U = DetMCMCgauss4d(shift,T,mux,muy,cxx,cyy,cxy, u_init )
#   RND = S
#   #MUR = S[1]
#   #    if np.mod(t,plotT)==0:
#   pp.figure()
#   #pp.clf()
#   pp.plot( XRR[:,0], XRR[:,1], 'bo', mec='b',ms=10, alpha=0.1)
#   #pp.plot( XR[:,0], XR[:,1], 'go')
#   #pp.plot( XS[:,0], XS[:,1], 'mo')
#   # for x0,y0,x1,y1 in zip(RND[:,:-1],RND[:,1:]):
#   #     pp.plot(x0,y0, ,'r-')
#   pp.plot(RND[0,:],RND[1,:],'m-')
#   #pp.plot(RND[0,:],RND[1,:],'ro')
#   #pp.plot(RND[0,:],RND[1,:],'ro')
#   pp.plot( RND[0,:],RND[1,:], 'ro', mec='r',ms=5, alpha=0.5)
#
#   pp.title( "u init = " + str(u_init))
#
#   #pp.figure()
#   #show_1( mu, C )
#   #pp.plot(MUR[0,:],MUR[1,:],'bo')
#
#   # pp.figure(2)
#   #   pp.clf()
#   #   pp.plot(R,lw=2)
#
#   pp.show()
#
#
#


class TargetDensity( object ):
  
  # log pdf
  def logp( self, x ):
    raise NotImplementedError
    
  def neglogp( self, x ):
    return -self.logp(x)
    
  def rand( N = 1 ):
    raise NotImplementedError
    
  # gradient of pdf
  def g( self, x ):
    raise NotImplementedError
    
  # gradient of log pdf
  def g_logp( self, x ):
    raise NotImplementedError
    
  def neg_g_logp( self, x ):
    return -self.g_logp(x)
      
  # pdf
  def p( self, x ):
    return np.exp( self.logp(x) )
    
  # a sample from gibbs posterior of variable i, conditioned on other variables
  def gibbs_rand( self, i, x_not_i, N = 1):
    raise NotImplementedError
    
  # a gibbs posterior parameters of variable i, conditioned on other variables
  def gibbs_post( self, i, x_not_i ):
    raise NotImplementedError
    
  def contour( self ):
    raise NotImplementedError
    
    
    
class PencilGaussian( TargetDensity ):
  def __init__( self, mu = np.array([0,0]), cov = np.array([[1.0,0.98],[0.98,1.0]]) ):
    self.mu     = mu
    self.cov    = cov
    self.icov   = np.linalg.inv( self.cov )
    self.det    = np.linalg.det(self.cov)
    self.logdet = np.log( self.det )
    self.dim    = 2
    
  def logp( self, x ):
    d = x-self.mu
    try:
      lp = -self.dim*log_sqrt_2pi - 0.5*self.logdet - 0.5*np.sum( np.dot( d, self.icov) * d, 1 )
    except:
      lp = -self.dim*log_sqrt_2pi - 0.5*self.logdet - 0.5*np.sum( np.dot( d, self.icov) * d )
    return lp
    
  def g( self, x ):
    return self.pdf(x)*self.g_logp(x)
    
  def g_logp( self, x ):
    d = x-self.mu
    return np.dot( self.icov, -d )
 
    
  def gibbs_post( self, i, x_not_i ):
    if i == 0:
      j=1
    elif i == 1:
      j=0
    else:
      raise Exception
      
    mui = self.mu[i]
    cii = self.cov[i,i]
    cjj = self.cov[j,j]
    cij = self.cov[i,j]
    cji = cij
    
    # xi | xj ~ N( ui + cij*cjj^-1*(xj-mui), cii - cij*cjj^-1*cji)
    mu = mui + cij*(x_not_i-mui)/cjj
    var = cii - cij*cji/cjj
    
    return mu, var
    
  def gibbs_rand( self, i, x_not_i, N = 1):
    
    mu, var = self.gibbs_post( i, x_not_i )
    if N == 1:
      return mu + np.random.randn()*np.sqrt(var)
    else:
      return mu + np.random.randn(N)*np.sqrt(var)
      
  def rand( self, N = 1 ):
    return np.random.multivariate_normal( self.mu, self.cov, N )
    
  def contour( self ):
    nspace = 50
    gridspace = np.linspace( -3, 3, nspace )
    X1,X2 = np.meshgrid( gridspace, gridspace ); x1 = X1.flatten(); x2 = X2.flatten();x=np.vstack((x1,x2)).T
    pdf = self.p( x )
    
    PDF = pdf.reshape((nspace,nspace))
    #y = self.rand(100000)
    #pp.hexbin(y[:,0], y[:,1],gridsize=30,cmap=pp.cm.winter)
    
    pp.contourf( X1, X2, PDF, 10, cmap=pp.cm.winter )

def run( x0, T, sampler, target ):
  X = [x0]
  for t in range( T ):
    xt = sampler.step( target )
    X.append(xt)
    
  X = np.array(X)
  
  return X
      
if __name__ == "__main__":
  #np.random.seed(1)
  pp.close('all')
  T = 10
  C = np.array([[1.0,0.985],[0.985,1.0]])
  target = PencilGaussian(cov = C )
  mu0 = np.zeros(2)
  x0 = np.array([3.0,-3.0])
  dim = len(x0)
  K = [100,100]
  S = [0,0]
  L=5
  epsilon=0.1
  width = 0.001
  nhistory = 1
  
  TRUE = np.random.multivariate_normal(mu0,C,T )
  gibbs = GibbsSampler( dim, x0 )
  order = OrderOverrelaxed( dim, x0, K, S )
  order2 = OrderOverrelaxedWithU( dim, x0, K, S )
  #hmc = HybridMonteCarlo( dim, x0, L, epsilon )
  #nrhmc = NoRejectHybridMonteCarlo( dim, x0, L, epsilon, demo=False )
  #print "WARNING nrhmc IS NOT WORKING!!!"
  #phmc = PenalizedHMC( dim, x0, L, epsilon, width, nhistory, demo=False )
  
  pp.figure()
  XX = []
  samplers = [gibbs, order,order2]
  #samplers = [phmc,hmc]
  i=1
  for sampler in samplers:
    np.random.seed(3)
    X = run( x0, T, sampler, target )
    #print "reject rate = %0.2f"%( sampler.rejections / float(T) )
    
    #pp.figure()
    pp.subplot(4,len(samplers), i )
    target.contour()
    pp.plot( X[:,0], X[:,1], 'r-' )
    pp.plot( X[:,0], X[:,1], 'wo', ms=4 )

    pp.subplot(4,len(samplers), i + len(samplers))
    #target.contour()
    pp.plot( X )
    
    pp.subplot(4,len(samplers), i + 2*len(samplers))
    pp.hist( TRUE[:,0], 20, normed = True, color='r',alpha=0.5 )
    #pp.hist( TRUE[:,1], 20, normed = True, color='r',alpha=0.5 )
    pp.hist( X[:,0], 20, normed = True, color='b',alpha=0.5 )
    
    pp.subplot(4,len(samplers), i + 3*len(samplers))
    pp.hist( TRUE[:,1], 20, normed = True, color='r',alpha=0.5 )
    #pp.hist( TRUE[:,1], 20, normed = True, color='r',alpha=0.5 )
    pp.hist( X[:,1], 20, normed = True, color='b',alpha=0.5 )
    i+=1
    XX.append(X)
  
  pp.draw()
  pp.show()