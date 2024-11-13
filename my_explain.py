import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import visualization
import cv2
from tqdm import tqdm
from scipy.ndimage import binary_opening, binary_closing, label
import torch
import torchvision.transforms as transforms
from PIL import Image
import CLIP.clip as clip
import json
import os
import torch.nn.functional as F

import torch
import numpy as np

import torch

def compute_equivalent_attention(attn_probs, gradients):
    """
    Computes the equivalent attention matrix.
    
    Args:
    attn_probs (torch.Tensor): Attention probabilities.
    gradients (torch.Tensor): Gradients of the attention probabilities.
    
    Returns:
    torch.Tensor: The equivalent attention matrix.
    """
    A = attn_probs * gradients
    A = torch.clamp(A, min=0)  # Only positive relevance
    A = A.mean(dim=1)  # Average over heads
    A = A / A.sum(dim=-1, keepdim=True)  # Row normalization
    return A

def forward_hook(module, input, output):
    module.input = input[0]
    module.output = output

def register_hooks(model):
    hooks = []
    for blk in model.visual.transformer.resblocks.children():
        hook = blk.register_forward_hook(forward_hook)
        hooks.append(hook)
    return hooks

def update_relevance_map(R, equivalent_attention, one_hot, Y, Y_prime):
    """
    Updates the relevance map.
    
    Args:
    R (torch.Tensor): The current relevance map.
    equivalent_attention (torch.Tensor): The equivalent attention matrix.
    one_hot (torch.Tensor): The one-hot encoded target.
    Y (torch.Tensor): The input tokens to the current layer.
    Y_prime (torch.Tensor): The output tokens of the current layer.
    
    Returns:
    torch.Tensor: The updated relevance map.
    """
    # Calculate alpha and beta
    Y_grad = torch.autograd.grad(one_hot, Y, retain_graph=True)[0]
    Y_prime_grad = torch.autograd.grad(one_hot, Y_prime, retain_graph=True)[0]

    alpha = (Y_grad * Y).sum(dim=-1) / ((Y_grad * Y).sum(dim=-1) + (Y_prime_grad * Y_prime).sum(dim=-1))
    beta = 1 - alpha

    equivalent_attention = equivalent_attention.expand(R.size(0), -1, -1)

    R_Y_prime_X = torch.bmm(equivalent_attention, R)
    R_Y_prime_X = R_Y_prime_X.mean(dim=1)  # Average over the heads dimension

    # Ensure alpha and beta have the correct dimensions
    alpha = alpha.view(-1, 1, 1)
    beta = beta.view(-1, 1, 1)
    
    R = torch.nn.functional.softmax(R, dim=-1)

    R = alpha * R + beta * R_Y_prime_X
    return R

def update_relevance_map_sum(R, equivalent_attention, one_hot, Y, Y_prime, alpha_sum):
    """
    Updates the relevance map using cumulative alpha from previous blocks.
    
    Args:
    R (torch.Tensor): The current relevance map.
    equivalent_attention (torch.Tensor): The equivalent attention matrix.
    one_hot (torch.Tensor): The one-hot encoded target.
    Y (torch.Tensor): The input tokens to the current layer.
    Y_prime (torch.Tensor): The output tokens of the current layer.
    alpha_sum (torch.Tensor): Sum of alpha values from previous blocks.
    
    Returns:
    torch.Tensor: The updated relevance map.
    """
    # Calculate current block's contribution
    Y_grad = torch.autograd.grad(one_hot, Y, retain_graph=True)[0]
    Y_prime_grad = torch.autograd.grad(one_hot, Y_prime, retain_graph=True)[0]
    
    current_alpha = (Y_grad * Y).sum(dim=-1) / ((Y_grad * Y).sum(dim=-1) + (Y_prime_grad * Y_prime).sum(dim=-1))
    
    # Convert tensors to float32 before multiplication
    equivalent_attention = equivalent_attention.float()
    R = R.float()
    
    equivalent_attention = equivalent_attention.expand(R.size(0), -1, -1)
    R_Y_prime_X = torch.bmm(equivalent_attention, R)
    R_Y_prime_X = R_Y_prime_X.mean(dim=1)  # Average over the heads dimension
    
    R = torch.nn.functional.softmax(R, dim=-1)
    
    # Calculate alpha and beta using current_alpha and alpha_sum
    alpha = current_alpha / (alpha_sum + current_alpha)
    beta = 1 - alpha
    
    # Ensure alpha and beta have the correct dimensions
    alpha = alpha.view(-1, 1, 1)
    beta = beta.view(-1, 1, 1)
    
    # Update R using alpha and beta
    R = alpha * R + beta * R_Y_prime_X
    
    return R, current_alpha

def interpret_image(image, texts, model, device, start_layer=-1):
    batch_size = texts.shape[0]
    images = image.repeat(batch_size, 1, 1, 1)
    logits_per_image, _ = model(images, texts)
    probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()
    index = [i for i in range(batch_size)]
    one_hot = np.zeros((logits_per_image.shape[0], logits_per_image.shape[1]), dtype=np.float32)
    one_hot[torch.arange(logits_per_image.shape[0]), index] = 1
    one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    one_hot = torch.sum(one_hot.cuda() * logits_per_image)
    model.zero_grad()

    image_attn_blocks = list(dict(model.visual.transformer.resblocks.named_children()).values())
    if start_layer == -1:
        # calculate index of last layer
        start_layer = len(image_attn_blocks) - 1

    num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
    R = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype).to(device)
    R = R.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
    for i, blk in enumerate(image_attn_blocks):
        if i < start_layer:
            continue
        grad = torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True)[0].detach()
        cam = blk.attn_probs.detach()
        cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
        cam = grad * cam
        cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
        cam = cam.clamp(min=0).mean(dim=1)
        R = R + torch.bmm(cam, R)
    image_relevance = R[:, 0, 1:]

    return image_relevance

def interpret_image_ours(model, image, texts, device, start_layer=-1):
    """
    Interpret image using weighted relevance across transformer blocks.
    
    Args:
    model: The CLIP model
    image: Input image tensor
    texts: Input text tensor
    device: Device to run computations on
    start_layer: Starting layer for interpretation (default: -1)
    """
    batch_size = texts.shape[0]
    images = image.repeat(batch_size, 1, 1, 1)
    logits_per_image, _ = model(images, texts)
    probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()
    index = [i for i in range(batch_size)]
    one_hot = np.zeros((logits_per_image.shape[0], logits_per_image.shape[1]), dtype=np.float32)
    one_hot[torch.arange(logits_per_image.shape[0]), index] = 1
    one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    one_hot = torch.sum(one_hot.cuda() * logits_per_image)
    model.zero_grad()

    image_attn_blocks = list(dict(model.visual.transformer.resblocks.named_children()).values())
    if start_layer == -1:
        start_layer = (len(image_attn_blocks) - 1) // 2

    num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
    R = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype).to(device)
    R = R.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
    
    # Initialize alpha sum
    alpha_sum = torch.zeros(batch_size).to(device)
    
    for i, blk in enumerate(image_attn_blocks):
        if i < start_layer:
            continue

        grad = torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True)[0].detach()
        cam = blk.attn_probs.detach()

        # Compute the equivalent attention matrix
        equivalent_attention = compute_equivalent_attention(cam, grad)

        # Update the relevance map with cumulative alpha
        Y = blk.input
        Y_prime = blk.output
        R, current_alpha = update_relevance_map_sum(R, equivalent_attention, one_hot, Y, Y_prime, alpha_sum)
        
        # Update alpha sum for next iteration
        alpha_sum = alpha_sum + current_alpha

    R = R * (-1)
    # Extract the relevance of the image tokens
    image_relevance = R[:, 0, 1:]
    return image_relevance


def interpret_image_mine(image, texts, model, device, start_layer=-1):
    threshold = 0.009
    batch_size = texts.shape[0]
    images = image.repeat(batch_size, 1, 1, 1)
    logits_per_image, _ = model(images, texts)
    probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()
    index = [i for i in range(batch_size)]
    one_hot = np.zeros((logits_per_image.shape[0], logits_per_image.shape[1]), dtype=np.float32)
    one_hot[torch.arange(logits_per_image.shape[0]), index] = 1
    one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    one_hot = torch.sum(one_hot.cuda() * logits_per_image)
    model.zero_grad()

    image_attn_blocks = list(dict(model.visual.transformer.resblocks.named_children()).values())
    if start_layer == -1:
        # calculate index of last layer
        start_layer = len(image_attn_blocks) - 1
        # start_layer = (len(image_attn_blocks) - 1) // 2

    num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
    R = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype).to(device)
    R = R.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
    for i, blk in enumerate(image_attn_blocks):
        if i < start_layer:
            continue
        grad = torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True)[0].detach()
        cam = blk.attn_probs.detach()
        cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
        cam = grad * cam
        cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
        cam = cam.clamp(min=0).mean(dim=1)

    # Apply thresholding to cam tensor
    thresholded_cam = cam.clone()  # Create a copy of cam tensor
    thresholded_cam[thresholded_cam < threshold] = 0  # Set values below threshold to 0
    R = R + torch.bmm(thresholded_cam, R)
    image_relevance = R[:, 0, 1:]

    return image_relevance


def interpret_image_weighted(image, texts, model, device, start_layer=-1):
    batch_size = texts.shape[0]
    images = image.repeat(batch_size, 1, 1, 1)
    logits_per_image, _ = model(images, texts)
    probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()
    index = [i for i in range(batch_size)]
    one_hot = np.zeros((logits_per_image.shape[0], logits_per_image.shape[1]), dtype=np.float32)
    one_hot[torch.arange(logits_per_image.shape[0]), index] = 1
    one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    one_hot = torch.sum(one_hot.cuda() * logits_per_image)
    model.zero_grad()

    image_attn_blocks = list(dict(model.visual.transformer.resblocks.named_children()).values())
    if start_layer == -1:
        start_layer = (len(image_attn_blocks) - 1) #// 2

    num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
    R = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype).to(device)
    R = R.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
    
    for i, blk in enumerate(image_attn_blocks):
        if i < start_layer:
            continue

        grad = torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True)[0].detach()  # Extract gradients
        cam = blk.attn_probs.detach()  # Detach attention probabilities

        # Compute the equivalent attention matrix
        equivalent_attention = compute_equivalent_attention(cam, grad)

        # Update the relevance map
        Y = blk.input  # Assuming Y is the current layer's input (this needs to be stored/accessible)
        Y_prime = blk.output  # Assuming Y_prime is the output of the current layer (this needs to be stored/accessible)
        R = update_relevance_map(R, equivalent_attention, one_hot, Y, Y_prime)
    
    # Extract the relevance of the image tokens
    image_relevance = R[:, 0, 1:]
    return image_relevance



def interpret_text(texts, model, device, start_layer_text=-1):
    batch_size = texts.shape[0]
    logits_per_image, logits_per_text = model(None, texts)
    probs = logits_per_text.softmax(dim=-1).detach().cpu().numpy()
    index = [i for i in range(batch_size)]
    one_hot = np.zeros((logits_per_text.shape[0], logits_per_text.shape[1]), dtype=np.float32)
    one_hot[torch.arange(logits_per_text.shape[0]), index] = 1
    one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    one_hot = torch.sum(one_hot.cuda() * logits_per_text)
    model.zero_grad()

    text_attn_blocks = list(dict(model.transformer.resblocks.named_children()).values())
    if start_layer_text == -1:
        # calculate index of last layer
        start_layer_text = len(text_attn_blocks) - 1

    num_tokens = text_attn_blocks[0].attn_probs.shape[-1]
    R_text = torch.eye(num_tokens, num_tokens, dtype=text_attn_blocks[0].attn_probs.dtype).to(device)
    R_text = R_text.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
    for i, blk in enumerate(text_attn_blocks):
        if i < start_layer_text:
            continue
        grad = torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True)[0].detach()
        cam = blk.attn_probs.detach()
        cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
        cam = grad * cam
        cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
        cam = cam.clamp(min=0).mean(dim=1)
        R_text = R_text + torch.bmm(cam, R_text)
    text_relevance = R_text

    return text_relevance


