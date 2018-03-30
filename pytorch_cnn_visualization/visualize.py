import json, copy

from PIL import Image
from matplotlib import pyplot as PLT

import torch, torchvision
from torchvision import transforms
from torch.autograd import Variable

from . import utils
from .core import Backpropagation, GuidedBackpropagation, GradCAM

class Visualize:
    """
    Warpper class of all the visualizatiom for efficiency and consistancy

    NOTE:
        I know some part of this class looks weird and redundant but my aims
        is to make different classes share the same model for efficiency
    """

    def __init__(self, model, transform, target_layer, num_classes=1000, retainModel=True, cuda=False):
        self.model = model
        self.cuda = cuda
        self.transform = transform
        self.target_layer = target_layer
        self.num_classes = num_classes

        if retainModel:
            self.model = copy.deepcopy(model)

        self.gradCAM = GradCAM(self.model, self.transform,
                               self.target_layer, self.num_classes, self.cuda)
        self.vBackprop = Backpropagation(self.model, self.transform, self.num_classes, self.cuda)
        self.gBackprop = GuidedBackpropagation(
            # guided backprop will change the gradient on back prop
            # need a copy of model
            copy.deepcopy(model),
            self.transform,
            self.num_classes,
            self.cuda
        )

    def input_image(self, image):
        """ Supply input image for "visual explanation"

        Argument:
            img (PIL Image) - the (unprocessed) input image

        Return:
            None
        """

        self.image = image
        img_tensor = self.transform(image)
        img_tensor.unsqueeze_(0) # this add a dimension as a dummy "batch"
        img_variable = Variable(img_tensor, requires_grad=True)

        self.input = img_variable
        self.ginput = copy.deepcopy(img_variable)

        self._forwarded = False
        self._forwarded_gbackprop = False
        self._model_idx = None
        self._gmodel_idx = None
        self._gradcam_intensity = None
        self._gradcam_intensity_idx = None
        self._vbackprop_gradient = None
        self._vbackprop_gradient_idx = None
        self._gbackprop_gradient = None
        self._gbackprop_gradient_idx = None

        # self.output = self.model(img_variable)
        # return self.output.data


    def get_prediction_output(self):
        """ Get the prediction output of the model

        Argument:
            None

        Return:
            Tensor - the prediction output of the model
        """

        if self._forwarded:
            return self.output.data
        if self._forwarded_gbackprop:
            return self.output_gbackprop.data

        self.output = self.model(self.input)
        self.gradCAM.input = self.input
        self.vBackprop.input = self.input
        self._forwarded = True

        return self.output.data

    def get_gradcam_intensity(self, idx, sorted_idx=False):
        """ Get the GradCAM intensity value for each pixel

        Argument:
            idx (int) - the idx of the class to be localize by GradCAM
            sorted_idx (bool) - if sorted_idx==True, the idx[0] will be the class with highest score,
                idx[1] will be the class with second highest score and so on

        Return:
            Tensor (size == kernal size of target layer)
                - GradCAM intensity value for each pixel
        """

        if sorted_idx:
            # implement sorted_idx at here to keep idx consistancy
            # idx = sorted_idx(idx)
            pass

        if self._gradcam_intensity_idx == idx:
            return self._gradcam_intensity

        if not self._forwarded:
            self.output = self.model(self.input)
            self.gradCAM.input = self.input
            self.vBackprop.input = self.input
            self._forwarded = True

        if self._model_idx != idx:
            self.model.zero_grad()
            self.output.backward(gradient=utils.one_hot_tensor(idx, self.num_classes), retain_graph=True)
            self._model_idx = idx

        self._gradcam_intensity = self.gradCAM.get_gradcam_intensity(idx, backward=False)
        self._gradcam_intensity_idx = idx;

        return self._gradcam_intensity

    def get_gradcam_heatmap(self, idx, sorted_idx=False):
        """ Get the GradCAM heatmap on the original input image
            and intensity value that is same size as original input image

        Argument:
            idx (int) - the idx of the class to be localize by GradCAM
            sorted_idx (bool) - if sorted_idx==True, the idx[0] will be the class with highest score,
                idx[1] will be the class with second highest score and so on

        Return:
            PIL image - The original image with GradCAM intensity applied to
            Numpy array - The intensity same size as original image (range: [0-1])
        """

        intensity = self.get_gradcam_intensity(idx, sorted_idx)
        return self.gradCAM.apply_color_map(intensity, self.image)

    def get_vanilla_backprop_gradient(self, idx, sorted_idx=False):
        """

        Return:
            PIL image - The gradient image
            Numpy array - The gradient of original image (range: [0-1])
        """
        if sorted_idx:
            # implement sorted_idx at here to keep idx consistancy
            # idx = sorted_idx(idx)
            pass

        if self._vbackprop_gradient_idx == idx:
            return self._vbackprop_gradient

        if not self._forwarded:
            self.output = self.model(self.input)
            self.gradCAM.input = self.input
            self.vBackprop.input = self.input
            self._forwarded = True

        if self._model_idx != idx:
            self.model.zero_grad()
            self.output.backward(gradient=utils.one_hot_tensor(idx, self.num_classes), retain_graph=True)
            self._model_idx = idx

        self._vbackprop_gradient = self.vBackprop.get_input_gradient(idx, backward=False)
        self._vbackprop_gradient_idx = idx;

        return self._vbackprop_gradient

    # def get_vanilla_backprop_saliency(self, idx, sorted_idx=False):
    #     gradient = self.get_vanilla_backprop_gradient(idx, sorted_idx)
    #
    #     return utils.normalize(gradient)

    def get_guided_backprop_gradient(self, idx, sorted_idx=False):
        if sorted_idx:
            # implement sorted_idx at here to keep idx consistancy
            # idx = sorted_idx(idx)
            pass

        if self._gbackprop_gradient_idx == idx:
            return self._gbackprop_gradient

        if not self._forwarded_gbackprop:
            self.output_gbackprop = self.gBackprop.model(self.ginput)
            self.gBackprop.input = self.ginput
            self._forwarded_gbackprop = True

        if self._gmodel_idx != idx:
            self.gBackprop.model.zero_grad()
            self.output_gbackprop.backward(gradient=utils.one_hot_tensor(idx, self.num_classes), retain_graph=True)
            self._gmodel_idx = idx

        self._gbackprop_gradient = self.gBackprop.get_input_gradient(idx, backward=False)
        self._gbackprop_gradient_idx = idx;

        return self._gbackprop_gradient

    # def get_guided_backprop_saliency(self, idx, sorted_idx=False):
    #     gradient = self.get_guided_backprop_gradient(idx, sorted_idx)
    #
    #     return utils.normalize(gradient)
    #
    def get_guided_gramcam_saliency(self, idx, sorted_idx=False):
        _, gradcam = self.get_gradcam_heatmap(idx, sorted_idx)
        _, gradient = self.get_guided_backprop_gradient(idx, sorted_idx)

        # gradient = utils.normalize(gradient) # this make the background black in guided gradcam
        result = (gradient.transpose(2, 0, 1)*gradcam).transpose(1, 2, 0)
        result = utils.normalize(result)
        return Image.fromarray((result*255).astype('uint8')), result



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Transfer Learning Train')
    parser.add_argument('image', help='path to the input image')
    parser.add_argument('--model', default='resnet152', metavar='model_name',
                        help='name of the model to visualize')
    parser.add_argument('--class-idx', default='top3', metavar='ID1,ID2',
                        help='the class index for CAM visualizing, \
                              can be comma seperated index or the string "top1" or "top3"')
    args = parser.parse_args()


    cnn = torchvision.models.__dict__[args.model](pretrained=True)
    input_size = utils.get_input_size(args.model)
    target_layer = utils.get_conv_layer(args.model)
    preprocess = transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize(
           mean=[0.485, 0.456, 0.406],
           std=[0.229, 0.224, 0.225]
        )
    ])
    class_name = json.load(open('data/class_name.json', 'r'))

    img_pil = Image.open(args.image)
    img_pil = img_pil.resize((input_size,input_size))

    visualizer = Visualize(cnn, preprocess, target_layer, retainModel=False)

    visualizer.input_image(img_pil)
    x = visualizer.get_prediction_output()
    score = x.cpu().numpy()[0]

    if args.class_idx == 'top1' or args.class_idx == 'top3':
        i = int(args.class_idx[3])

        tmp_score = score.copy()

        class_idx = []
        for i in range(i):
            class_idx.append(tmp_score.argmax())
            tmp_score[class_idx[-1]] = -1000
    else:
        class_idx = [int(i) for i in args.class_idx.split(',')]

    for idx in class_idx:
        print(idx, score[idx], class_name[idx])

        img = [
            visualizer.get_gradcam_heatmap(idx)[0],
            visualizer.get_guided_backprop_gradient(idx)[0],
            visualizer.get_vanilla_backprop_gradient(idx)[0],
            visualizer.get_guided_gramcam_saliency(idx)[0]
        ]
        title = ["Grad-CAM", "Guided Backpropagation", "Backpropagation", "Guided Grad-CAM"]
        fig = PLT.figure(class_name[idx].split(",")[0])

        for i in range(4):
            ax = fig.add_subplot(221+i)
            ax.axis('off')
            ax.imshow(img[i])
            ax.set_title(title[i])

        PLT.suptitle(class_name[idx]+" Score: "+str(x[0][idx])[:5], fontsize=18)
        PLT.show()
