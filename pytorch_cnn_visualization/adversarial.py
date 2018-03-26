import torch, torchvision
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

from . import utils

class Adversarial:
    """
    Generate adversarial example for a CNN using fast gradient sign method or
    gradient ascend method
    """

    def __init__(self, model, transform, cuda=False):
        self.model = model
        self.transform = transform

    def generate(self, img, target=-1, eps=1, epoch=5, method=0, lr=0.01):
        """ Generate the adversarial example

        Argument:
            img (PIL Image) - the (unprocessed) input image
            target (integer, default:-1) - the target class for the adversarial
                                           attack, negative for untarget attack
            eps (float, default:1) - epsilon, in range of [0-255], the amount of
                                     pixel value changes allows for each pixel
            epoch (integer, default:5) - number of iteration
            method (0 or 1, default:0) - 0 for fast gradient sign method
                                         1 for gradient ascend
            lr (float, default:0.1) - learning rate, for gradient ascend only

        Return:
            adv_img (PIL Image) - the generated adversarial example
        """

        ori_img = img
        PILtoTensor = torchvision.transforms.ToTensor()
        TensortoPIL = torchvision.transforms.ToPILImage()

        eps = eps/255.0
        if method == 0:
            # fast gradient sign method
            # each iteration change eps/iteration, total change eps
            eps = eps/epoch

        if target < 0:
            # untargeted attack, move away from target
            score = self.model(Variable(img_tensor))
            target = score.cpu().numpy()[0].argmax()
            targeted = False
        else:
            targeted = True

        img_tensor = self.transform(img).unsqueeze_(0) # add dimension as "batch"
        total_adv_noise = img_tensor.clone().zero_()
        adv_img = img
        for _ in range(epoch):
            img_tensor = self.transform(adv_img).unsqueeze_(0) # add dimension as "batch"
            img_var = Variable(img_tensor, requires_grad=True)
            img_var.grad = None
            output = self.model(img_var)
            self.model.zero_grad()
            loss = F.cross_entropy(output, Variable(torch.LongTensor([target])))
            loss.backward()
            gradient = img_var.grad.data

            if method == 0:     # fast gradient sign method
                adv_noise = eps*torch.sign(gradient)
            else:               # gradient ascend
                adv_noise = lr*gradient

            if targeted:
                adv_noise = -adv_noise

            total_adv_noise += adv_noise

            if method != 0:     # gradient ascent
                total_adv_noise.clamp_(-eps, eps)

            adv_img = PILtoTensor(ori_img) + total_adv_noise[0]
            adv_img.clamp_(0, 1)
            adv_img = TensortoPIL(adv_img)

        # adv_noise_img = torchvision.transforms.ToPILImage()(total_adv_noise[0])
        return adv_img
