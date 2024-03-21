
import os
import random
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from methods.base import TTAMethod
from utils.visualization import save_col_preds
import inspect

logger = logging.getLogger(__name__)

def debug_log(count=0):
    print("▶▷▷ debug check : ", count)

class SymCrossEntropy(nn.Module):
    def __init__(self, alpha=1, beta=0.1, num_classes=14, ignore_index=255):
        super(SymCrossEntropy, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, ignore_index=self.ignore_index)
        
        targets_adjusted = targets.clone()
        targets_adjusted[targets == self.ignore_index] = 0
        targets_adjusted = targets_adjusted.unsqueeze(-1)
        
        targets_one_hot = F.one_hot(targets_adjusted, num_classes=self.num_classes)
        targets_one_hot = targets_one_hot.squeeze(-2).permute(0, 3, 1, 2).float()
        
        mask = targets.unsqueeze(1) == self.ignore_index
        mask_expanded = mask.expand_as(targets_one_hot)
        
        targets_one_hot[mask_expanded] = 0.0
        
        # Calculate reverse cross entropy loss
        softmax_inputs = F.softmax(inputs, dim=1)
        rce_loss = -(targets_one_hot * torch.log(softmax_inputs + 1e-6)).sum(dim=1).mean()
        debug_log(9)
        
        # Combine losses
        total_loss = self.alpha * ce_loss + self.beta * rce_loss
        return total_loss
    
class SCELoss(nn.Module):
    def __init__(self, alpha, beta, num_classes=10, ignore_index=255):
        super(SCELoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def forward(self, pred, labels):
        # CCE
        ce = F.cross_entropy(pred, labels, ignore_index=self.ignore_index)

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss

class GTTA(TTAMethod):
    def __init__(self, model, optimizer, 
                crop_size, steps, episodic, adain_model, src_loader, adain_loader, steps_adain, device, save_dir,
                lambda_ce_trg=0.1, num_classes=14, ignore_label=255, style_transfer=True):
        super().__init__(model, optimizer, crop_size, steps, episodic)

        self.adain_model = adain_model
        self.src_loader = src_loader
        self.adain_loader = adain_loader

        self.src_loader_iter = iter(self.src_loader)
        self.adain_loader_iter = iter(self.adain_loader)

        self.steps_adain = steps_adain
        self.device = device

        self.lambda_ce_trg = lambda_ce_trg
        self.num_classes = num_classes

        self.ignore_label = ignore_label
        self.use_style_transfer = style_transfer
        
        # self.cross_entropy = torch.nn.CrossEntropyLoss(ignore_index=self.ignore_label)
        self.cross_entropy = SymCrossEntropy(alpha=0.1, beta=1.0, num_classes=self.num_classes, ignore_index=self.ignore_label)
        self.avg_conf = torch.tensor(0.9).cuda()
        self.save_dir = save_dir

        # initialize some other variables
        self.class_weights = None
        self.dataset_means = [[torch.tensor([]) for _ in range(self.num_classes)] for _ in range(2)]
        self.dataset_stds = [[torch.tensor([]) for _ in range(self.num_classes)] for _ in range(2)]

    def forward(self, x):
        if self.episodic:
            self.reset()

        self.model.train()

        # generate the filtered pseudo-labels
        pseudos_thr = self.create_pseudo_labels(x)

        if self.use_style_transfer:
            # extract class-wise moments
            self.extract_moments(x, pseudo_label=pseudos_thr)

            # train style transfer model
            self.adain_model.train()
            for _ in range(self.steps_adain):
                self.forward_and_adapt_adain(optimizer=self.adain_model.optimizer_dec)
            self.adain_model.eval()

        # train segmentation model
        for _ in range(self.steps):
            self.forward_and_adapt(x, pseudos_thr)

        self.model.eval()
        return self.model(x)

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, img_test, pseudo_labels):
        # sample source batch
        try:
            imgs_src, labels_src, files_src = next(self.src_loader_iter)
        except StopIteration:
            self.src_loader_iter = iter(self.src_loader)
            imgs_src, labels_src, files_src = next(self.src_loader_iter)

        imgs_src, labels_src = imgs_src.to(self.device), labels_src.to(self.device).long()

        if self.use_style_transfer:
            # do style transfer and normalize output for segmentation model
            with torch.no_grad():
                gen_imgs = self.adain_model(input=[imgs_src, labels_src],
                                            moments_list=[self.dataset_means, self.dataset_stds])

                # prepare the images for the segmentation model
                gen_imgs = torch.cat([gen_imgs[:1], imgs_src[1:]], dim=0) if random.random() <= 0.2 else gen_imgs
        else:
            gen_imgs = imgs_src

        self.optimizer.zero_grad()
        logits = self.model(gen_imgs)
        loss_src = self.cross_entropy(logits, labels_src)
        debug_log(10)
        loss_src.backward()
        debug_log(11)
        self.optimizer.step()
        debug_log(12)
        
        # create test batch
        imgs_test_cropped = torch.tensor([], device=self.device)
        pseudos_cropped = torch.tensor([], device=self.device)
        _, _, height, width = imgs_src.shape
        for _ in range(2):
            x1 = random.randint(0, img_test.shape[-1] - width)
            y1 = random.randint(0, img_test.shape[-2] - height)
            imgs_test_cropped = torch.cat([imgs_test_cropped, img_test[:, :, y1:y1 + height, x1:x1 + width]], dim=0)
            pseudos_cropped = torch.cat([pseudos_cropped, pseudo_labels[:, y1:y1 + height, x1:x1 + width]], dim=0)

        # do self-training with pseudo-labeled target images
        self.optimizer.zero_grad()
        debug_log(13)
        loss_trg = F.cross_entropy(input=self.model(imgs_test_cropped),
                                    target=pseudos_cropped.long(),
                                    ignore_index=self.ignore_label)
        print("debug check loss_trg", loss_trg.item())
        loss_trg *= self.lambda_ce_trg
        loss_trg.backward()
        self.optimizer.step()
        
    def adjust_labels(labels, max_label):
        """
        Adjust labels to ensure they are within the expected range.

        Args:
            labels (torch.Tensor): The labels tensor.
            max_label (int): The maximum label value (num_classes - 1).

        Returns:
            torch.Tensor: Adjusted labels tensor.
        """
        # Example mapping logic, adjust according to your needs
        # This is a simple direct mapping, but you might need something more sophisticated
        factor = max_label / 255
        adjusted_labels = (labels.float() * factor).long()
        return adjusted_labels
    
    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt_adain(self, optimizer):
        try:
            imgs_src, labels_src, ids_src = next(self.adain_loader_iter)
        except StopIteration:
            self.adain_loader_iter = iter(self.adain_loader)
            imgs_src, labels_src, ids_src = next(self.adain_loader_iter)

        optimizer.zero_grad()
        gen_imgs, loss_content, loss_style = self.adain_model(input=[imgs_src.to(self.device), labels_src.to(self.device).long()],
                                                                moments_list=[self.dataset_means, self.dataset_stds])

        loss = loss_content + 0.1 * loss_style
        loss.backward()
        optimizer.step()

    @torch.no_grad()
    def create_pseudo_labels(self, img_test, save_thr_pseudos=False):
        # create pseudo-labels
        output = self.model(img_test)
        confidences, pseudo_labels = torch.max(output.softmax(dim=1), dim=1) # get the confidence and the pseudo-label for each pixel
        # torch.max returns the maximum value and the index of the maximum value. 

        # filter unreliable samples
        momentum = 0.9
        self.avg_conf = momentum * self.avg_conf + (1 - momentum) * confidences.mean()
        mask = torch.where(confidences < torch.sqrt(self.avg_conf)) # torch.where returns the indices of the elements that satisfy the condition

        # create filtered pseudo-labels
        pseudo_labels_thr = pseudo_labels.clone()
        pseudo_labels_thr[mask] = self.ignore_label 
        #ignore_label means that the pixel is not used for training. because it is not reliable enough when the confidence is lower than the average confidence
        # so to make the model more robust, we ignore the pixels with low confidence as set the pseudo-labels to ignore_label(255).
        # 255 value 

        # save the predictions
        # if save_thr_pseudos:
        #     preds_dir_path = os.path.join(self.save_dir, "pseudos_thr")
        #     os.makedirs(preds_dir_path, exist_ok=True)
        #     save_col_preds(preds_dir_path, [str(self.counter)], pseudo_labels_thr.clone(), images=img_test.clone())

        return pseudo_labels_thr

    @torch.no_grad()
    def extract_moments(self, img_test, pseudo_label):
        # extract the class-wise moments
        out_adain = self.adain_model(input=[img_test.to(self.device), pseudo_label.to(device=self.device, dtype=torch.long)])

        for i_adain_layer, (means, stds, classes) in enumerate(out_adain):  # iterate through the adain layers
            for i_sample in range(means.shape[0]):  # iterate through all samples of one batch
                for class_nr in classes[i_sample]:  # iterate through all classes contained in one sample
                    # only add moment, if the class is present in both adain layers
                    if class_nr in out_adain[-1][-1][i_sample]:
                        self.dataset_means[i_adain_layer][class_nr] = torch.cat(
                            [self.dataset_means[i_adain_layer][class_nr], means[i_sample, class_nr, :].unsqueeze(0).cpu()], dim=0)
                        self.dataset_stds[i_adain_layer][class_nr] = torch.cat(
                            [self.dataset_stds[i_adain_layer][class_nr], stds[i_sample, class_nr, :].unsqueeze(0).cpu()], dim=0)

