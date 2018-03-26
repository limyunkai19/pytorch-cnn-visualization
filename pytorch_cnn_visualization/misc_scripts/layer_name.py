import json
import torch, torchvision


available_model = json.load(open('../data/available_model.json', 'r'))

def get_layer_name(model):
    layer_name = []
    for module in model.named_modules():
        layer_name.append(module[0])
    return layer_name

def get_conv_layer_name(model):
    conv_layer_name = []
    for module in model.named_modules():
        if isinstance(module[1], torch.nn.Conv1d) or \
           isinstance(module[1], torch.nn.Conv2d) or \
           isinstance(module[1], torch.nn.Conv3d):
                conv_layer_name.append(module[0])
    return conv_layer_name

all_layer_name = {}
all_conv_layer_name = {}
for model_name in available_model:
    print("Extracting layer name of: ", model_name)
    model = torchvision.models.__dict__[model_name](pretrained=False)

    all_layer_name[model_name] = get_layer_name(model)
    all_conv_layer_name[model_name] = get_conv_layer_name(model)

json.dump(all_layer_name, open('../data/layer_name.json', 'w'));
json.dump(all_conv_layer_name, open('../data/conv_layer_name.json', 'w'));
