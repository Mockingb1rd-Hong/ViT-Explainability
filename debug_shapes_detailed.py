import torch
from PIL import Image
import numpy as np
import CLIP.clip as clip
from my_explain import compute_equivalent_attention, register_hooks, interpret_image, interpret_image_weighted, interpret_image_ours

def print_tensor_info(name, tensor):
    """Helper function to print tensor information"""
    print(f"\n{name}:")
    print(f"  Shape: {tensor.shape}")
    if len(tensor.shape) >= 2:
        print(f"  First dimension (batch_size): {tensor.shape[0]}")
        print(f"  Second dimension: {tensor.shape[1]}")
        if len(tensor.shape) >= 3:
            print(f"  Third dimension: {tensor.shape[2]}")
    print(f"  Type: {tensor.dtype}")
    print(f"  Device: {tensor.device}")
    if isinstance(tensor, torch.Tensor):
        print(f"  Requires grad: {tensor.requires_grad}")
        print(f"  Min value: {tensor.min().item():.4f}")
        print(f"  Max value: {tensor.max().item():.4f}")

def perturb_image(image, image_relevance, percentage):
    # Flatten the relevance scores to a 1D array
    relevance_flat = image_relevance.flatten()

    # Calculate the number of tokens to remove based on the percentage
    num_tokens = relevance_flat.shape[0]
    num_to_remove = int(num_tokens * percentage)

    # Find the indices of the most important tokens based on relevance scores
    indices_to_remove = torch.topk(relevance_flat, num_to_remove, largest=True).indices

    # Create a binary mask for the tokens to remove
    binary_mask = torch.zeros_like(relevance_flat, dtype=torch.bool)
    binary_mask[indices_to_remove] = True

    # Reshape the binary mask to the original image relevance shape
    binary_mask = binary_mask.view_as(image_relevance)

    # Upsample the binary mask to match the image dimensions
    binary_mask = torch.nn.functional.interpolate(binary_mask.unsqueeze(0).unsqueeze(0).float(), size=(image.shape[2], image.shape[3]), mode='nearest').bool()

    # Expand the binary mask to match the image dimensions
    binary_mask = binary_mask.expand(image.shape[0], image.shape[1], -1, -1)

    # Apply the binary mask to the image by setting the selected tokens to zero
    perturbed_image = image.clone()
    perturbed_image[binary_mask] = 0

    return perturbed_image

def debug_interpretation_pipeline():
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    
    # Load sample image and text
    img_path = "CLIP/glasses.png"
    image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
    texts = ["sunglasses"]
    text = clip.tokenize(texts).to(device)
    
    print("\n=== Input Shapes ===")
    print_tensor_info("Image", image)
    print_tensor_info("Text", text)
    
    # Forward pass
    print("\n=== Forward Pass ===")
    hooks = register_hooks(model)
    with torch.enable_grad():
        logits_per_image, _ = model(image, text)
        print_tensor_info("Logits", logits_per_image)
        
        # Get attention from first block
        blk = model.visual.transformer.resblocks[0]
        print("\n=== Attention Block Information ===")
        print(f"Number of attention blocks: {len(model.visual.transformer.resblocks)}")
        
        # Print shapes for each interpretation method
        print("\n=== Interpretation Method Shapes ===")
        
        print("\n1. Standard Interpretation")
        R_image = interpret_image(image, text, model, device, start_layer=0)
        print_tensor_info("Standard R_image", R_image)
        
        print("\n2. Weighted Interpretation")
        R_image_weighted = interpret_image_weighted(image, text, model, device, start_layer=0)
        print_tensor_info("Weighted R_image", R_image_weighted)
        
        print("\n3. Our Interpretation")
        R_image_ours = interpret_image_ours(model, image, text, device)
        print_tensor_info("Our R_image", R_image_ours)
        
        # Print differences between methods
        print("\n=== Differences Between Methods ===")
        if R_image.shape == R_image_weighted.shape == R_image_ours.shape:
            print(f"All methods produce same shape: {R_image.shape}")
            print(f"Mean absolute difference (Standard vs Weighted): {(R_image - R_image_weighted).abs().mean().item():.4f}")
            print(f"Mean absolute difference (Standard vs Ours): {(R_image - R_image_ours).abs().mean().item():.4f}")
            print(f"Mean absolute difference (Weighted vs Ours): {(R_image_weighted - R_image_ours).abs().mean().item():.4f}")
        else:
            print("Warning: Methods produce different shapes!")
            print(f"Standard shape: {R_image.shape}")
            print(f"Weighted shape: {R_image_weighted.shape}")
            print(f"Ours shape: {R_image_ours.shape}")
        
        # Clean up hooks
        for hook in hooks:
            hook.remove()

def analyze_relevance_maps():
    # Print sorted values to see distribution
    print("\nSorted relevance values:")
    print("Standard:")
    print(torch.sort(R_image[0])[0])
    
    print("\nWeighted:")
    print(torch.sort(R_image_weighted[0])[0])
    
    print("\nOurs:")
    print(torch.sort(R_image_ours[0])[0])
    
    # Print token rankings
    print("\nMost important token indices:")
    print("Standard:", torch.topk(R_image[0], 5).indices)
    print("Weighted:", torch.topk(R_image_weighted[0], 5).indices)
    print("Ours:", torch.topk(R_image_ours[0], 5).indices)
    
    # Compare rankings between methods
    standard_order = torch.argsort(R_image[0], descending=True)
    weighted_order = torch.argsort(R_image_weighted[0], descending=True)
    ours_order = torch.argsort(R_image_ours[0], descending=True)
    
    # Stack the orders for correlation computation
    correlation_matrix = torch.stack([
        standard_order.float(),
        weighted_order.float(),
        ours_order.float()
    ])
    
    # Compute correlation matrix
    corr = torch.corrcoef(correlation_matrix)
    
    print("\nRank correlation:")
    print("Standard vs Weighted:", corr[0,1].item())
    print("Standard vs Ours:", corr[0,2].item())
    print("Weighted vs Ours:", corr[1,2].item())
    
    # Print percentile distribution
    print("\nPercentile distribution:")
    percentiles = [0, 25, 50, 75, 100]
    for method, relevance in [("Standard", R_image), ("Weighted", R_image_weighted), ("Ours", R_image_ours)]:
        print(f"\n{method}:")
        for p in percentiles:
            # Ensure relevance is float for quantile computation
            relevance_float = relevance[0].float()
            value = torch.quantile(relevance_float, p/100)
            print(f"{p}th percentile: {value.item():.4f}")

def print_value_distribution(relevance_map, method_name):
    """Prints the value distribution of the relevance map."""
    print(f"\n=== {method_name} Value Distribution ===")
    print(f"  Min value: {relevance_map.min().item():.4f}")
    print(f"  Max value: {relevance_map.max().item():.4f}")
    print(f"  Mean value: {relevance_map.mean().item():.4f}")
    
    # Print percentiles
    percentiles = [0, 25, 50, 75, 100]
    for p in percentiles:
        value = torch.quantile(relevance_map, p / 100).item()
        print(f"  {p}th percentile: {value:.4f}")

def main():
    print("Starting debug analysis of shapes and sizes...")
    
    # Run the interpretation pipeline and store results
    global R_image, R_image_weighted, R_image_ours  # Make these global so analyze_relevance_maps can access them
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    
    # Load sample image and text
    img_path = "CLIP/glasses.png"
    image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
    texts = ["sunglasses"]
    text = clip.tokenize(texts).to(device)
    
    # Get relevance maps
    hooks = register_hooks(model)
    with torch.enable_grad():
        R_image = interpret_image(image, text, model, device, start_layer=0)
        R_image_weighted = interpret_image_weighted(image, text, model, device, start_layer=0)
        R_image_ours = interpret_image_ours(model, image, text, device)
    
    # Clean up hooks
    for hook in hooks:
        hook.remove()
    
    # Run both analyses
    debug_interpretation_pipeline()
    print("\nAnalyzing relevance distributions...")
    analyze_relevance_maps()
    
    print("\nDebug analysis complete!")
    
    # In the main function, after computing the relevance maps
    print_value_distribution(R_image, "Standard Interpretation")
    print_value_distribution(R_image_weighted, "Weighted Interpretation")
    print_value_distribution(R_image_ours, "Our Interpretation")

if __name__ == "__main__":
    main() 