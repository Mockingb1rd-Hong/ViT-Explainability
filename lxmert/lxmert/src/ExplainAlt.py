import numpy as np
import torch
import copy
from torch.autograd import grad


#All of rules was consicdered about the batch situation!

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    R = alpha * R + beta * R_Y_prime_X
    return R

def compute_rollout_attention(all_layer_matrices, start_layer=0):
    # adding residual consideration
    num_tokens = all_layer_matrices[0].shape[1]
    eye = torch.eye(num_tokens).to(all_layer_matrices[0].device)
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
    matrices_aug = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
                          for i in range(len(all_layer_matrices))]
    joint_attention = matrices_aug[start_layer]
    for i in range(start_layer+1, len(matrices_aug)):
        joint_attention = matrices_aug[i].matmul(joint_attention)
    return joint_attention

# rule 5 from paper
def avg_heads(cam, grad):
    cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
    grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
    cam = grad * cam
    cam = cam.clamp(min=0).mean(dim=0)
    return cam

# rules 6 + 7 from paper
def apply_self_attention_rules(R_ss, R_sq, cam_ss):
    R_sq_addition = torch.matmul(cam_ss, R_sq)
    R_ss_addition = torch.matmul(cam_ss, R_ss)
    return R_ss_addition, R_sq_addition

# rules 10 + 11 from paper
def apply_mm_attention_rules(R_ss, R_qq, R_qs, cam_sq, apply_normalization=True, apply_self_in_rule_10=True):
    R_ss_normalized = R_ss
    R_qq_normalized = R_qq
    if apply_normalization:
        R_ss_normalized = handle_residual(R_ss)
        R_qq_normalized = handle_residual(R_qq)
    R_sq_addition = torch.matmul(R_ss_normalized.t(), torch.matmul(cam_sq, R_qq_normalized))
    if not apply_self_in_rule_10:
        R_sq_addition = cam_sq
    R_ss_addition = torch.matmul(cam_sq, R_qs)
    return R_sq_addition, R_ss_addition

# normalization- eq. 8+9
def handle_residual(orig_self_attention):
    self_attention = orig_self_attention.clone()
    diag_idx = range(self_attention.shape[-1])
    # computing R hat
    self_attention -= torch.eye(self_attention.shape[-1]).to(self_attention.device)
    assert self_attention[diag_idx, diag_idx].min() >= 0
    # normalizing R hat
    self_attention = self_attention / self_attention.sum(dim=-1, keepdim=True)
    self_attention += torch.eye(self_attention.shape[-1]).to(self_attention.device)
    return self_attention

class GeneratorOurs:
    def __init__(self, model_usage, save_visualization=False):
        self.model_usage = model_usage
        self.save_visualization = save_visualization

    def handle_self_attention_lang(self, blocks):
        for blk in blocks:
            grad = blk.attention.self.get_attn_gradients().detach()
            if self.use_lrp:
                cam = blk.attention.self.get_attn_cam().detach()
            else:
                cam = blk.attention.self.get_attn().detach()
            cam = avg_heads(cam, grad)
            R_t_t_add, R_t_i_add = apply_self_attention_rules(self.R_t_t, self.R_t_i, cam)
            self.R_t_t += R_t_t_add
            self.R_t_i += R_t_i_add

    def handle_self_attention_image(self, blocks):
        for blk in blocks:
            grad = blk.attention.self.get_attn_gradients().detach()
            if self.use_lrp:
                cam = blk.attention.self.get_attn_cam().detach()
            else:
                cam = blk.attention.self.get_attn().detach()
            cam = avg_heads(cam, grad)
            R_i_i_add, R_i_t_add = apply_self_attention_rules(self.R_i_i, self.R_i_t, cam)
            self.R_i_i += R_i_i_add
            self.R_i_t += R_i_t_add

    def handle_co_attn_self_lang(self, block):
        grad = block.lang_self_att.self.get_attn_gradients().detach()
        if self.use_lrp:
            cam = block.lang_self_att.self.get_attn_cam().detach()
        else:
            cam = block.lang_self_att.self.get_attn().detach()
        cam = avg_heads(cam, grad)
        R_t_t_add, R_t_i_add = apply_self_attention_rules(self.R_t_t, self.R_t_i, cam)
        self.R_t_t += R_t_t_add
        self.R_t_i += R_t_i_add

    def handle_co_attn_self_image(self, block):
        grad = block.visn_self_att.self.get_attn_gradients().detach()
        if self.use_lrp:
            cam = block.visn_self_att.self.get_attn_cam().detach()
        else:
            cam = block.visn_self_att.self.get_attn().detach()
        cam = avg_heads(cam, grad)
        R_i_i_add, R_i_t_add = apply_self_attention_rules(self.R_i_i, self.R_i_t, cam)
        self.R_i_i += R_i_i_add
        self.R_i_t += R_i_t_add

    def handle_co_attn_lang(self, block):
        if self.use_lrp:
            cam_t_i = block.visual_attention.att.get_attn_cam().detach()
        else:
            cam_t_i = block.visual_attention.att.get_attn().detach()
        grad_t_i = block.visual_attention.att.get_attn_gradients().detach()
        cam_t_i = avg_heads(cam_t_i, grad_t_i)
        R_t_i_addition, R_t_t_addition = apply_mm_attention_rules(self.R_t_t, self.R_i_i, self.R_i_t, cam_t_i,
                                                                  apply_normalization=self.normalize_self_attention,
                                                                  apply_self_in_rule_10=self.apply_self_in_rule_10)
        return R_t_i_addition, R_t_t_addition

    def handle_co_attn_image(self, block):
        if self.use_lrp:
            cam_i_t = block.visual_attention_copy.att.get_attn_cam().detach()
        else:
            cam_i_t = block.visual_attention_copy.att.get_attn().detach()
        grad_i_t = block.visual_attention_copy.att.get_attn_gradients().detach()
        cam_i_t = avg_heads(cam_i_t, grad_i_t)
        R_i_t_addition, R_i_i_addition = apply_mm_attention_rules(self.R_i_i, self.R_t_t, self.R_t_i, cam_i_t,
                                                                  apply_normalization=self.normalize_self_attention,
                                                                  apply_self_in_rule_10=self.apply_self_in_rule_10)
        return R_i_t_addition, R_i_i_addition

    def generate_ours(self, input, index=None, use_lrp=True, normalize_self_attention=True, apply_self_in_rule_10=True, method_name="ours"):
        self.use_lrp = use_lrp
        self.normalize_self_attention = normalize_self_attention
        self.apply_self_in_rule_10 = apply_self_in_rule_10
        kwargs = {"alpha": 1}
        output = self.model_usage.forward(input).question_answering_score
        model = self.model_usage.model

        # initialize relevancy matrices
        text_tokens = self.model_usage.text_len
        image_bboxes = self.model_usage.image_boxes_len

        # text self attention matrix
        self.R_t_t = torch.eye(text_tokens, text_tokens).to(model.device)
        # image self attention matrix
        self.R_i_i = torch.eye(image_bboxes, image_bboxes).to(model.device)
        # impact of images on text
        self.R_t_i = torch.zeros(text_tokens, image_bboxes).to(model.device)
        # impact of text on images
        self.R_i_t = torch.zeros(image_bboxes, text_tokens).to(model.device)


        if index is None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)

        model.zero_grad()
        one_hot.backward(retain_graph=True)
        if self.use_lrp:
            model.relprop(torch.tensor(one_hot_vector).to(output.device), **kwargs)

        # language self attention
        blocks = model.lxmert.encoder.layer
        self.handle_self_attention_lang(blocks)

        # image self attention
        blocks = model.lxmert.encoder.r_layers
        self.handle_self_attention_image(blocks)

        # cross attn layers
        blocks = model.lxmert.encoder.x_layers
        for i, blk in enumerate(blocks):
            # in the last cross attention module, only the text cross modal
            # attention has an impact on the CLS token, since it's the first
            # token in the language tokens
            if i == len(blocks) - 1:
                break
            # cross attn- first for language then for image
            R_t_i_addition, R_t_t_addition = self.handle_co_attn_lang(blk)
            R_i_t_addition, R_i_i_addition = self.handle_co_attn_image(blk)

            self.R_t_i += R_t_i_addition
            self.R_t_t += R_t_t_addition
            self.R_i_t += R_i_t_addition
            self.R_i_i += R_i_i_addition

            # language self attention
            self.handle_co_attn_self_lang(blk)

            # image self attention
            self.handle_co_attn_self_image(blk)


        # take care of last cross attention layer- only text
        blk = model.lxmert.encoder.x_layers[-1]
        # cross attn- first for language then for image
        R_t_i_addition, R_t_t_addition = self.handle_co_attn_lang(blk)
        self.R_t_i += R_t_i_addition
        self.R_t_t += R_t_t_addition

        # language self attention
        self.handle_co_attn_self_lang(blk)

        # disregard the [CLS] token itself
        self.R_t_t[0,0] = 0
        return self.R_t_t, self.R_t_i