import torch
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
from PIL import Image
import matplotlib.cm as Pltcolormap

from . import utils

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM)
    Get a coarse heatmap of activation highlighting important location in
    input image based on the gradient of (last) convolutional layer to
    achieve "visual explanation" for CNN prediction

    Reference: https://arxiv.org/abs/1610.02391
    """

    def __init__(self, model, transform, target_layer, num_classes=1000, cuda=False):
        self.model = model
        self.model.train(False)
        self.cuda = cuda
        if self.cuda:
            self.model.cuda()
        self.transform = transform
        self.target_layer = target_layer
        self.num_classes = num_classes

        # define hook function
        def forward_hook(module, input, output):
            self.feature_maps = output.data

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].data

        # register hook function
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(forward_hook)
                module.register_backward_hook(backward_hook)

    def forward(self, img):
        """ The forward pass

        Argument:
            img (PIL Image) - the (unprocessed) input image

        Return:
            Tensor/Dict - the output of the model
        """

        # preprocess the PIL image first
        img_tensor = self.transform(img)
        img_tensor.unsqueeze_(0) # this add a dimension as a dummy "batch"
        img_variable = Variable(img_tensor)
        self.output = self.model(img_variable)
        return self.output.data

    def backward(self, idx, sorted_idx=False):
        self.model.zero_grad()
        self.output.backward(gradient=utils.one_hot_tensor(idx, self.num_classes), retain_graph=True)

    def get_gradcam_intensity(self, idx, sorted_idx=False, backward=True):
        """ The (partial) backward pass and generate GradCAM intensity value for each pixel

        Argument:
            idx (int) - the idx of the class to be localize by GradCAM
            sorted_idx (bool) - if sorted_idx==True, the idx[0] will be the class with highest score,
                idx[1] will be the class with second highest score and so on
            backward (bool) - perform backward pass or not,
                                under normal usecase, it should be true

        Return:
            Tensor (size == kernal size of target layer)
                - GradCAM intensity value for each pixel
        """

        # implement sorted_idx !!!
        if backward:
            self.backward(idx)
        # self.feature_maps # 1x2048x7x7
        # self.gradients # 1x2048x7x7

        # GAP = torch.nn.AvgPool2d(self.gradients.size()[2:])
        weights = F.avg_pool2d(Variable(self.gradients), kernel_size=self.gradients.size()[2:]).data

        gradCAM_intensity = torch.FloatTensor(self.feature_maps.size()[2:]).zero_()

        for feature_map, weight in zip(self.feature_maps[0], weights[0]):
            gradCAM_intensity += feature_map * weight

        #relu
        gradCAM_intensity.clamp_(min=0)

        return gradCAM_intensity

    @staticmethod
    def apply_color_map(intensity, img):
        """ Apply the color map on the original image with GradCAM intensity value generated
            by GradCAM.backward()

        Argument:
            intensity (Tensor) - GradCAM intensity value generated by GradCAM.backward()
            img (PIL image) - The image that GradCAM intensity were to be apply to,
                suppose to be the original image

        Return:
            PIL image - The img with GradCAM intensity applied to
            Numpy array - The intensity same size as img (range: [0-1])
        """

        # normalize
        intensity = utils.normalize(intensity)

        # use PIL bilinear resize interpolation
        # note: *255 -> resize -> /255.0 (divide for heat map input[0,1]) is === resize
        pil = Image.fromarray(intensity.cpu().numpy())
        pil = pil.resize(img.size, resample=Image.BILINEAR)
        intensity = np.asarray(pil)

        # get the color map from matplotlib
        color_map = Pltcolormap.get_cmap('jet')
        heat_map = color_map(intensity)
        heat_map[:,:,3] /= 2.0
        heat_map *= 255

        original_img = np.asarray(img)

        return Image.fromarray(np.uint8((heat_map[:,:,:3]+original_img)/2.0)), intensity

class Backpropagation:
    """
    Vanilla Backpropagation
    Backpropagate to input then get the gradient at input
    """

    def __init__(self, model, transform, num_classes=1000, cuda=False):
        self.model = model
        # self.model = model
        self.model.train(False)
        self.cuda = cuda
        if self.cuda:
            self.model.cuda()
        self.transform = transform
        self.num_classes = num_classes

    def forward(self, img):
        """ The forward pass

        Argument:
            img (PIL Image) - the (unprocessed) input image

        Return:
            Tensor/Dict - the output of the model
        """

        img_tensor = self.transform(img)
        img_tensor.unsqueeze_(0) # this add a dimension as a dummy "batch"
        img_variable = Variable(img_tensor, requires_grad=True)

        self.input = img_variable
        self.output = self.model(img_variable)
        return self.output.data

    def backward(self, idx):
        self.model.zero_grad()
        self.output.backward(gradient=utils.one_hot_tensor(idx, self.num_classes), retain_graph=True)

    def get_input_gradient(self, idx, sorted_idx=False, backward=True):
        """ The backward pass and return the gradient at input image

        Argument:
            idx (int) - the idx of the class to be localize by GradCAM
            sorted_idx (bool) - if sorted_idx==True, the idx[0] will be the class with highest score,
                idx[1] will be the class with second highest score and so on
            backward (bool) - perform backward pass or not,
                                under normal usecase, it should be true

        Return:
            PIL image - The RGB gradient images generated based on the gradient value
            Numpy array (nxnxc) - The gradient value for each pixel
        """

        #implement sorted_idx !!!

        if backward:
            self.backward(idx)

        # 1x3x224x224 -> 224x224x3
        gradient = self.input.grad.data.cpu().numpy()[0].transpose(1, 2, 0)

        gradient_img_arr = (utils.normalize(gradient)*255).astype('uint8')
        return Image.fromarray(gradient_img_arr), gradient

class GuidedBackpropagation(Backpropagation):
    """
    Guided Backpropagation

    x.grad or img.grad is what we wanted
    GuidedBackprop: input>0 * gradin>0 * gradin on relu.backward
    but original relu had implemented relu gradin = input>0 * gradin
    thus we only need to add gradin>0 * relu gradin

    Reference: https://arxiv.org/abs/1412.6806

    NOTE:
        The gradient on back propagation of the model will be modify, if this is
        not the desired behaeviour, construct this class with a deepcopy of model
    """

    def __init__(self, model, transform, num_classes, cuda=False):
        super().__init__(model, transform, num_classes, cuda)

        # define hook function
        def backward_hook(module, grad_input, grad_output):
            # Guided Backpropagation
            # Only allows positive gradient to backflow
            return (torch.clamp(grad_input[0], min=0.0),)

        # register hook function on relu module
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.ReLU):
                module.register_backward_hook(backward_hook)

class GuidedGradCAM:
    """
    Guided Grad-CAM
    Use the heatmap of Grad-CAM to produce a localized guided-backprop saliency

    Reference: https://arxiv.org/abs/1610.02391

    NOTE:
        This class is present for completeness and consistancy.
        For general visualization usage, please use the Visualize wrapper class
    """
    def __init__(self, model, transform, target_layer, cuda=False):
        self.model = model
        self.cuda = cuda
        self.transform = transform
        self.target_layer = target_layer

        self.GradCAM = GradCAM(model, transform, target_layer, cuda)
        self.GuidedBackprop = GuidedBackpropagation(copy.deepcopy(model),
                                                    transform, cuda)

    def forward(self, img):
        """ The forward pass
        Argument:
            img (Tensor) - the (unprocessed) input image
        Return:
            Tensor/Dict
        """
        pass
