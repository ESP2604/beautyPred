import tensorflow as tf
import tf2onnx
from tensorflow.python.framework import graph_io
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile

# 加载冻结的 TensorFlow 图
def load_graph(pb_file):
    with tf.io.gfile.GFile(pb_file, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.compat.v1.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")
    return graph

# 指定模型路径
model_path = "nsfw_mobilenet.pb"
graph = load_graph(model_path)

# 获取输入和输出节点名称（请根据您的模型进行修改）
input_names = ["input_1:0"]  # 例如，输入节点名称为 "input_1:0"
output_names = ["dense_3/Softmax:0"]  # 例如，输出节点名称为 "dense_3/Softmax:0"

# 将冻结的图转换为 ONNX 格式
onnx_model_path = "nsfw_mobilenet.onnx"
tf2onnx.convert.from_graph_def(
    graph.as_graph_def(),
    input_names=input_names,
    output_names=output_names,
    output_path=onnx_model_path,

)
