import numpy as np
from os import listdir
from os.path import join

import tensorflow as tf

def restore_graph(graph_dir):
    tf.reset_default_graph()
    sess=tf.Session()    
    
    meta_file = sorted([f for f in listdir(graph_dir) if f.endswith('.meta')])[-1]
    saver = tf.train.import_meta_graph(join(graph_dir, meta_file))
    saver.restore(sess, tf.train.latest_checkpoint(graph_dir))
    
    graph = tf.get_default_graph()
    
    return graph, sess

def predict(image, labels, graph, sess):
    model = {
        "bottleneck_tensor": graph.get_tensor_by_name('flatten/flatten/Reshape:0'),
        "bottleneck_input": graph.get_tensor_by_name('bottleneck/InputPlaceholder:0'), 
        "images": graph.get_tensor_by_name("placeholders_variables/input:0"),
        "labels": graph.get_tensor_by_name('placeholders_variables/labels:0'),
        "keep" : graph.get_tensor_by_name('placeholders_variables/dropout_keep:0')
    }
    
    bottleneck = graph.get_tensor_by_name("bottleneck/InputPlaceholder:0")
    tensor = sess.run(model["bottleneck_tensor"], 
                          feed_dict={model["images"]:image, 
                                     model["bottleneck_input"]:np.zeros((1, 43264)), 
                                     model["labels"]:np.array([1,0,0,0,0,0,0,0,0]).reshape(1,-1), 
                                     model["keep"]:1.0})
    
    logits = graph.get_tensor_by_name("fully_conn/logits:0")
    prediction = tf.argmax(logits, 1)
    
    result = sess.run(prediction, feed_dict={bottleneck:tensor})
    
    return labels[np.squeeze(result)]