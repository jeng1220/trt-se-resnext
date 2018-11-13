import numpy as np
import graphsurgeon as gs
import tensorflow as tf
from pdb import set_trace

def extract_tensor_content(input_obj):
    try:
        dim = input_obj.attr["value"].tensor.tensor_shape.dim
        nb_dim = len(dim)
        if nb_dim != 4: 
            raise ValueError    
        shape = (dim[0].size, dim[1].size, dim[2].size, dim[3].size)

        if "kernel" in input_obj.name:
            if input_obj.attr["value"].tensor.float_val[0] == 0:
                wts = np.zeros(shape, dtype=np.float32)
            else: 
                raise ValueError
        elif "W" in input_obj.name or "concat" in input_obj.name:
            wts = np.frombuffer(input_obj.attr["value"].tensor.tensor_content, dtype=np.float32)
            wts = wts.reshape(shape)
        else: 
            raise ValueError

        return wts
    except ValueError:
        print("unknown network pattern")
        set_trace()

def fix_bug1_concat_const(dynamic_graph):
    node_list = dynamic_graph.find_nodes_by_op("ConcatV2")

    for node in node_list:

        input_str_list = node.input
        input_obj_list = []

        for input_str in input_str_list:
            input_obj = dynamic_graph.find_nodes_by_path(input_str)[0]
            if input_obj.op == "Const":
                input_obj_list.append(input_obj)

        if len(input_obj_list) == 3:
            try:
                input_obj = input_obj_list[0]
                wts0 = extract_tensor_content(input_obj)

                input_obj = input_obj_list[1]
                wts1 = extract_tensor_content(input_obj)

                input_obj = input_obj_list[2]
                if "axis" in input_obj.name:
                    axis = input_obj.attr["value"].tensor.int_val[0]
                else:
                    raise ValueError

                if len(wts0.shape) == 4 and len(wts1.shape) == 4 and axis == 2:
                    new_wts = np.concatenate((wts0, wts1), axis=2)
                    node.op = "Const"
                    node.attr.clear()
                    node.attr["dtype"].CopyFrom(tf.AttrValue(type=tf.float32.as_datatype_enum))
                    node.attr["value"].CopyFrom(tf.AttrValue(tensor=tf.make_tensor_proto(new_wts, dtype=tf.float32, shape=new_wts.shape)))

                    for input_obj in input_obj_list:
                        dynamic_graph.remove(input_obj)
                        print("remove", node.name, ":", input_obj.name)
                else:
                    raise ValueError
            except ValueError:
                print("unknown network pattern")
                set_trace()

def preprocess(dynamic_graph):
    fix_bug1_concat_const(dynamic_graph)
