from models.SmaAt_UNet import *  # assuming your class is in this file
import torch
import torch.fx as fx
import torch.nn as nn
import json
from torchsummary import summary

def get_weights_shape(module):
    if hasattr(module, "weight") and module.weight is not None:
        return list(module.weight.shape)
    return None

def get_activation_type(node):
    target = node.target
    if isinstance(target, str):
        if "relu6" in target.lower():
            return "RELU6"
        if "relu" in target.lower():
            return "RELU"
        if "sigmoid" in target.lower():
            return "SIGMOID"
        if "swish" in target.lower():
            return "SWISH"
    return "0"

def get_layer_type(node, module):
    if isinstance(module, nn.Conv2d):
        if module.groups == module.in_channels == module.out_channels:
            base_name = "depthwise_conv_2d"
            conv_type = "dw"
        else:
            base_name = "conv_2d"
            if module.kernel_size == (1,1) or module.kernel_size == 1:
                conv_type = "pw"
            else:
                conv_type = "s"
        return base_name, conv_type
    if isinstance(module, nn.Linear):
        return "linear", None
    if node.op == "call_function":
        if node.target == torch.add:
            return "add", None
        if node.target == torch.mul:
            return "mul", None
    if node.op == "placeholder":
        return "input", None
    if node.op == "output":
        return "output", None
    return "other", None

def get_stride(module):
    if hasattr(module, "stride"):
        return module.stride if isinstance(module.stride, int) else module.stride[0]
    return None

def get_ifms_ofms_shapes(node, example_input, module=None):
    if node.op == "placeholder":
        shape = list(example_input.shape[1:])
        return shape, shape
    return None, None  # For all others, fallback to hook-based values

def extract_graph(model, only_shrinking, input_shape=(1, 3, 224, 224)):
    model.eval()
    example_input = torch.randn(input_shape)
    traced = fx.symbolic_trace(model)

    node_to_id = {node: i for i, node in enumerate(traced.graph.nodes)}
    id_to_node = {i: node for node, i in node_to_id.items()}

    parents_map = {i: [] for i in node_to_id.values()}
    children_map = {i: [] for i in node_to_id.values()}

    for node in traced.graph.nodes:
        this_id = node_to_id[node]
        for inp_node in node.all_input_nodes:
            if inp_node in node_to_id:
                inp_id = node_to_id[inp_node]
                parents_map[this_id].append(inp_id)
                children_map[inp_id].append(this_id)

    graph = []
    modules = dict(model.named_modules())

    # Shape tracking via forward hooks
    shapes = {}

    def hook(module, input, output):
        in_shape = list(input[0].shape[1:]) if isinstance(input, (tuple, list)) else list(input.shape[1:])
        out_shape = list(output.shape[1:]) if hasattr(output, 'shape') else None
        shapes[module] = {"ifm": in_shape, "ofm": out_shape}

    hooks = []
    for name, mod in modules.items():
        hooks.append(mod.register_forward_hook(hook))

    # Run a forward pass to capture shapes
    with torch.no_grad():
        model(example_input)

    for h in hooks:
        h.remove()

    last_hw = None
    for node in traced.graph.nodes:
        this_id = node_to_id[node]
        module = modules.get(node.target, None) if node.op == "call_module" else None

        _, layer_type = get_layer_type(node, module)
        activation = "0"
        if layer_type in ("s", "dw", "linear"):
            activation = "0"

        weights_shape = get_weights_shape(module) if module else None
        strides = get_stride(module) if module else None

        # Shape logic
        ifms_shape, ofms_shape = None, None
        if node.op == "placeholder":
            ifms_shape, ofms_shape = get_ifms_ofms_shapes(node, example_input, module)
        elif module in shapes:
            ifms_shape = shapes[module].get("ifm")
            ofms_shape = shapes[module].get("ofm")

        if only_shrinking and ofms_shape and len(ofms_shape) == 3 and isinstance(module, nn.Conv2d):
            _, h, w = ofms_shape
            if last_hw is not None:
                last_h, last_w = last_hw
                if h > last_h and w > last_w:
                    # Spatial size is not shrinking anymore → stop
                    break
            last_hw = (h, w)

        graph.append({
            "id": this_id,
            "name": str(node),
            "activation": activation,
            "type": layer_type,
            "weights_shape": weights_shape,
            "ifms_shape": ifms_shape,
            "ofms_shape": ofms_shape,
            "strides": strides,
            "parents": parents_map[this_id],
            "children": children_map[this_id]
        })

    return graph

def export_to_onnx(model, input_shape):
    dummy_input = torch.inp
    torch.onnx.export(
        model,
        dummy_input,
        "lvunet.onnx",
        input_names=["input"],
        output_names=["output"],
        opset_version=14,  # ONNX version
        do_constant_folding=True
    )

def export_to_onnx(model, input_shape):
    dummy_input = torch.randn(*input_shape)
    torch.onnx.export(
        model,
        dummy_input,
        "lvunet.onnx",
        input_names=["input"],
        output_names=["output"],
        opset_version=14,  # ONNX version
        do_constant_folding=True
    )
    
# Run
if __name__ == "__main__":
    model = SmaAt_UNet(n_channels=3, n_classes=21)
    summary(model, (3, 256, 256))
    #export_to_onnx(model, (1, 3, 256, 256))
    graph = extract_graph(model, True, input_shape=(1, 3, 256, 256))
    with open("model_dag.json", "w") as f:
        json.dump(graph, f, indent=4)

    print("✅ Saved graph to mobilenet_graph_final.json")
