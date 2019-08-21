import tensorflow as tf

"""
This is a short tutorial of tensorflow. After this tutorial, you should know the following concepts:
1. constant,
2. operations
3. variables 
4. gradient calculation 
5. optimizer 
"""

def regression_graph(print_info=False):

    # a linear model (w, b)
    w = tf.constant(1.6, name='weight')
    b = tf.constant(0.9, name='bias')

    # a data point (x, y)
    x = tf.constant(0.8, name='feature')
    y = tf.constant(0.2, name='label')

    # calculate the function value 
    # calculations like +, -, *, /, are overloaded by tensorflow, so `w * x' is the same as  `tf.multiply(w, x)' 
    #f = w * x + b  
    f = tf.add(tf.multiply(w, x, name='inner_product'), b, name='score') 

    if print_info:
        print('Msg from the function: we can evaluate any value in the graph with tf.Session')
        print('Msg from the function: the value of f = w * x + b is ' + str(tf.Session().run(f)))


    loss = tf.square(tf.subtract(f, y, name='difference'), name='squared_loss')
    
    return loss 


def regression_graph_vectorized(print_info=False):
    
    # weight matrix has size 2
    w = tf.constant(1.6, shape=[2], name='weight')
    b = tf.constant(0.9, name='bias')

    x = tf.constant([[0.3, 0.5], [0.4, 0.4], [0.1, 0.7]], name='feature')
    y = tf.constant([[0.1], [0.2], [0.3]], name='label')

    if print_info:
        print('By w.get_shape(), we get the shape of the tensor w: ' + str(w.get_shape()))
        print('With tf.shape(w), we get the shape of w as a one-element tensor: ' + str(tf.shape(w)))

    # For matrix multiplication, we need to make w a two-dimensional tensor
    w = tf.expand_dims(w, axis=1)

    # TODO: please check the document of tf.matmul and calculate w * x through matrix multiplication. Name the operation as "inner_product"
    inner_prod = None 

    # after the inner product, we get a two-dimensional matrix with the second dimension as 1, but we want a vector
    if print_info:
        print('The shape of tf.matmul(x, w) is: ' + str(inner_prod.get_shape()))
    
    # squeeze it
    inner_prod_squeezed = tf.squeeze(inner_prod)

    if print_info:
        print('The shape of tf.squeeze(tf.matmul(x, w)) is: ' + str(inner_prod_squeezed.get_shape()))

    f = tf.add(inner_prod_squeezed, b, name='score')

    instance_losses = tf.square(f - y)

    loss = tf.reduce_sum(instance_losses)
    
    return loss 


def regression_graph_with_placeholder(x, y, print_info=False):
    
    w = tf.constant(1.6, shape=[2], name='weight')
    b = tf.constant(0.9, name='bias')

    # TODO: please complete this function so it can calculate squared loss in the same way as the two functions above
    # Actually, tensorflow treats place holders in the same way as constant tensors in almost all operations 
    loss = None
   
    return loss


def regression_graph_with_variable(w, b, print_info=False):
    
    x = tf.constant([[0.3, 0.5], [0.4, 0.4], [0.1, 0.7]], name='feature')
    y = tf.constant([[0.1], [0.2], [0.3]], name='label')

    # TODO: please complete this function so it can calculate squared loss in the same way as the first two functions
    # Tensorflow treats variables also in the same way as constant tensors in almost all operations 
    loss = None
     
    return loss


def regression_graph_full(x, y, w, b, print_info=False):

    # TODO: please implement this function to calculate squared loss from these inputs. Then optimize w and b in ipython notebook.
    loss = None
   
    return loss



