


class layer( object ):
  def __init__( self, nbr_input_units, nbr_hidden_units ):
    self.nbr_input_units  = nbr_input_units
    self.nbr_hidden_units = nbr_hidden_units
    
  def forward( self, inputs ):
    N,D = inputs.shape
    
    A = np.dot( inputs, self.W ) + self.b
    
    I = pp.find( A >= 0 )
    
    H = np.zeros( (N,n) )
    
    H[I] = 1
    return H
    
    
class heavyside_nn(object):
  def __init__( self ):
    self.layers = []
    self.nbr_parameters = 0
    
  def add_layer( self, layer ):
    self.layers.append( layer )
    self.nbr_parameters += layer.nbr_input_units*layer.nbr_hidden_units + layer.nbr_hidden_units
    
  def forward( self, inputs ):
    
    for layer in self.layers:
      outputs = layer.forward( inputs )
      inputs = outputs
      
    return outputs
    
if __name__ == "__main__":