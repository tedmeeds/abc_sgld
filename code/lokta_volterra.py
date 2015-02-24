# 2 species
h = function(x, pars) {
   hazs = numeric(length(pars))
   hazs[1] = pars[1]*x[1]
   hazs[2] = pars[2]*x[1]*x[2]
   hazs[3] = pars[3]*x[2]
   return(hazs)
}

smat = matrix(0,nrow=2,ncol=3)
smat[1,1] = 1; smat[1,2] = -1
smat[2,2] = 1; smat[2,3] = -1
rownames(smat) = c("Prey", "Predator")

initial = c(100, 100)

pars = c(0.5,0.0025,0.3)

model = create_model(smat, h, initial, pars)


f = get_f = function(x, pars)
{
   fmat = matrix(0, nrow=3, ncol=2)
   fmat[1,1] = pars[1]
   fmat[2,1] = pars[2]*x[2]
   fmat[2,2] = pars[2]*x[1]
   fmat[3,2] = pars[3]
   return(fmat)
}
model = create_model(smat, h, initial, pars, f)
l2 = lna(model, maxtime=50, ddt=0.1, TRUE)
plot(g[,1], g[,2], type="l")


# 3 species
h = function(x, pars) {
   hazs = numeric(length(pars))
   hazs[1] = pars[1]*x[1]
   hazs[2] = pars[2]*x[1]*x[2]
   hazs[3] = pars[3]*x[2]*x[3]
   hazs[4] = pars[4]*x[3]
   return(hazs)
}

smat = matrix(0,nrow=3,ncol=4)
smat[1,1] = 1; smat[1,2] = -1
smat[2,2] = 1; smat[2,3] = -1
smat[3,3] = 1; smat[3,4] = 1
rownames(smat) = c("Prey", "Predator","Other")

initial = c(10, 100,10)

pars = c(0.5,0.0025,0.3, 0.6)

model = create_model(smat, h, initial, pars)
g = gillespie(model, maxtime=50)

f = get_f = function(x, pars)
{
   fmat = matrix(0, nrow=3, ncol=2)
   fmat[1,1] = pars[1]
   fmat[2,1] = pars[2]*x[2]
   fmat[2,2] = pars[2]*x[1]
   fmat[3,2] = pars[3]
   return(fmat)
}
model = create_model(smat, h, initial, pars, f)
l2 = lna(model, maxtime=50, ddt=0.1, TRUE)
plot(g[,1], g[,2], type="l")








import numpy as np
import scipy as sp
import pylab as pp

def simulator(self, theta):
    dt = self.dt

    pred = self.pred_start
    prey = self.prey_start

    c1 = np.exp(theta[0])
    c2 = np.exp(theta[1])
    c3 = np.exp(theta[2])

    output = np.zeros((self.T, 2))

    def lotvol(dt, pred, prey, c1, c2, c3):
        d_prey = c1 * prey - c2 * prey * pred
        d_pred = c2 * prey * pred - c3 * pred

        return pred + d_pred * dt, prey + d_prey * dt

    for i in range(self.T):
        for j in range(int(1.0 / self.dt)):
            pred, prey = lotvol(dt, pred, prey, c1, c2, c3)

        output[i, 0] = pred + distr.normal.rvs(0, self.noise)
        output[i, 1] = prey + distr.normal.rvs(0, self.noise)

    return output
    

def lv2(x,y,a,b,c,d,T):
  
  X = [x]
  Y = [y]
  
  dt = 0.001
  for t in range(T):
    dx = a*x - b*x*y
    dy = c*x*y - d*y
    
    x = x + dx*dt
    y = y + dy*dt
    
    X.append(x)
    Y.append(y)
    
  return np.array(X), np.array(Y)
  
def lv( r, x, A, T ):
  # r: N-dimensional linear growth rate
  # x: N-dimensional population size
  # A: N-N dim sparse species interaction matrix
  
  N = len(r)
  X = np.zeros( (N,T))
  X[:,0] = x
  
  for t in range(T-1):
    
    dx = r*x*(1.0 - np.dot( A, x ) )
    
    x = x + dx
    
    X[:,t+1] = x
    
  return X
  
  
if __name__ == "__main__":
  pp.close('all')
  
  N = 2
  
  r = 3*np.ones(N)
  x = np.random.rand(N)
  A = np.eye(N)
  
  a = 1.0
  b = 1.0
  c = 1.0
  d = 1.0
  x = 10
  y = 5
  
  A[0,1] = 1
  A[1,0] = 1
  
  T=500000
  
  #X = lv( r, x, A, T )
  
  X,Y = lv2(x,y,a,b,c,d,T)
  
  pp.figure(1)
  pp.clf()
  pp.plot(X)
  pp.plot(Y)
  
  pp.show()