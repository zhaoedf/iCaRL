import copy
import torch
from torch import nn

import sys
sys.path.append("..")
from feature_extractor.cifar_resnet import resnet32
from feature_extractor.resnet import resnet18, resnet34, resnet50
from feature_extractor.ucir_cifar_resnet import resnet32 as cosine_resnet32
from feature_extractor.ucir_resnet import resnet18 as cosine_resnet18
from feature_extractor.ucir_resnet import resnet34 as cosine_resnet34
from feature_extractor.ucir_resnet import resnet50 as cosine_resnet50
from fc.linears import SimpleLinear, SplitCosineLinear, CosineLinear


def get_convnet(convnet_type, pretrained=False):
    name = convnet_type.lower()
    if name == 'resnet32':
        return resnet32()  # [modified]
    elif name == 'resnet18':
        return resnet18(pretrained=pretrained)
    elif name == 'resnet34':
        return resnet34(pretrained=pretrained)
    elif name == 'resnet50':
        return resnet50(pretrained=pretrained)
    elif name == 'cosine_resnet18':
        return cosine_resnet18(pretrained=pretrained)
    elif name == 'cosine_resnet32':
        return cosine_resnet32()
    elif name == 'cosine_resnet34':
        return cosine_resnet34(pretrained=pretrained)
    elif name == 'cosine_resnet50':
        return cosine_resnet50(pretrained=pretrained)
    else:
        raise NotImplementedError('Unknown type {}'.format(convnet_type))


class BaseNet(nn.Module):

    def __init__(self, convnet_type, pretrained):
        super(BaseNet, self).__init__()

        self.convnet = get_convnet(convnet_type, pretrained)#backbone/feature_extractor
        self.fc = None

    @property #use Class.function as Class property(attri)
    def feature_dim(self):
        return self.convnet.out_dim

    def extract_vector(self, x): #feature vector
        return self.convnet(x)['features']

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x['features'])
        '''@Author:defeng
            what's in side x: (details can be found in convs folder)
        {
            'fmaps': [x_1, x_2, ..., x_n],
            'features': features
        }
        '''
        out.update(x)
        '''@Author:defeng
            notice: out(*i.e., self.fc(...)*) is not a tensor but a python dict(*see linears.py*) for details.
            so the .update() method is actually a python dict method.
            see for details: http://c.biancheng.net/view/4386.html
        '''

        return out

    def update_fc(self, nb_classes):#nb stands for number, so nb_classes is equal to numOfclasses. e.g., nb_output = self.fc.out_features
        pass

    def generate_fc(self, in_dim, out_dim):
        pass

    def copy(self):
        return copy.deepcopy(self)# copy self and return

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self
    '''@Author:defeng
        easy to understand freeze.
    '''


class IncrementalNet(BaseNet):

    def __init__(self, convnet_type, pretrained, gradcam=False):
        super().__init__(convnet_type, pretrained)
        self.gradcam = gradcam
        if hasattr(self, 'gradcam') and self.gradcam:
            self._gradcam_hooks = [None, None] # cos there are only two hooks, forward and backward hooks.
            self.set_gradcam_hook()

    def update_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias
            '''@Author:defeng
                like I expected, when increasing nb_classes, you need to modify out_features of the classifier to match the nb_classes.
                In order to do that, you will have to:
                    1. using generate_fc() to generate a new fc.
                    2. using "slicing" to move the params in old fc to the new fc.
                    3. delete the old fc from main memory.
            '''

        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)

        return fc

    def forward(self, x):
        x = self.convnet(x)
        # print(type(self.fc))
        out = self.fc(x['features'])
        out.update(x)
        # print(out.keys())
        if hasattr(self, 'gradcam') and self.gradcam:
            out['gradcam_gradients'] = self._gradcam_gradients
            out['gradcam_activations'] = self._gradcam_activations

        return out

    def unset_gradcam_hook(self):
        self._gradcam_hooks[0].remove()
        self._gradcam_hooks[1].remove()
        self._gradcam_hooks[0] = None
        self._gradcam_hooks[1] = None
        self._gradcam_gradients, self._gradcam_activations = [None], [None] # len=1

    def set_gradcam_hook(self):
        self._gradcam_gradients, self._gradcam_activations = [None], [None] #Variables to store grad and activations

        def backward_hook(module, grad_input, grad_output):
            self._gradcam_gradients[0] = grad_output[0] # grad_output[0].detach() is preferable!
            return None

        def forward_hook(module, input, output):
            self._gradcam_activations[0] = output #output, i.e., activation
            return None

        self._gradcam_hooks[0] = self.convnet.last_conv.register_backward_hook(backward_hook)
        self._gradcam_hooks[1] = self.convnet.last_conv.register_forward_hook(forward_hook)
        '''@Author:defeng
            forward_hook: get activation map A
            backward_hook: get grad from network classifier output(logits) to the activation map A
            see for details: Grad-CAM paper.   
        '''

        '''@Author:defeng
            info about torch register_XX_hook fucntion can be found in: https://blog.csdn.net/u011995719/article/details/97752853
        '''


class CosineIncrementalNet(BaseNet):

    def __init__(self, convnet_type, pretrained, nb_proxy=1):
        super().__init__(convnet_type, pretrained)
        self.nb_proxy = nb_proxy

    def update_fc(self, nb_classes, task_num):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            if task_num == 1: # increment task_num, task_num == 1 means only the first increments.
                fc.fc1.weight.data = self.fc.weight.data
                fc.sigma.data = self.fc.sigma.data
            else:
                prev_out_features1 = self.fc.fc1.out_features
                fc.fc1.weight.data[:prev_out_features1] = self.fc.fc1.weight.data
                fc.fc1.weight.data[prev_out_features1:] = self.fc.fc2.weight.data
                fc.sigma.data = self.fc.sigma.data

        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        if self.fc is None:# flag for determing whether is is the first increment or not.
            fc = CosineLinear(in_dim, out_dim, self.nb_proxy, to_reduce=True)
        else:
            prev_out_features = self.fc.out_features // self.nb_proxy
            # prev_out_features = self.fc.out_features
            fc = SplitCosineLinear(in_dim, prev_out_features, out_dim - prev_out_features, self.nb_proxy)

        return fc
        '''@Author:defeng
            from the condition:"if self.fc is None:", we can know that
        '''

# -----------------------------------------------------------------
class BiasLayer(nn.Module):
    def __init__(self):
        super(BiasLayer, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1, requires_grad=True))
        self.beta = nn.Parameter(torch.zeros(1, requires_grad=True))

    def forward(self, x, low_range, high_range):
        ret_x = x.clone()
        ret_x[:, low_range:high_range] = self.alpha * x[:, low_range:high_range] + self.beta
        return ret_x

    def get_params(self):
        return (self.alpha.item(), self.beta.item())


class IncrementalNetWithBias(BaseNet):
    def __init__(self, convnet_type, pretrained, bias_correction=False):
        super().__init__(convnet_type, pretrained)

        # Bias layer
        self.bias_correction = bias_correction
        self.bias_layers = nn.ModuleList([])
        self.task_sizes = []

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x['features'])
        if self.bias_correction:
            logits = out['logits']
            for i, layer in enumerate(self.bias_layers):
                logits = layer(logits, sum(self.task_sizes[:i]), sum(self.task_sizes[:i+1]))
            out['logits'] = logits

        out.update(x)

        return out

    def update_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)
        self.bias_layers.append(BiasLayer())

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)

        return fc

    def get_bias_params(self):
        params = []
        for layer in self.bias_layers:
            params.append(layer.get_params())

        return params

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True
