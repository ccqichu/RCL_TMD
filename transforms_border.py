"""
Border/Center Region Transformation Utilities for Robustness Evaluation

This module implements image transformations that manipulate border and center regions
to evaluate model's dependency on different spatial regions.

Transformations include:
- border_only: Keep border region, mask/blur center
- center_only: Keep center region, mask/blur border
- edge_mask: Mask border region (equivalent to center_only)
- crop_resize: Crop border and resize back to original size

Author: Auto-generated for CID-DIMM robustness evaluation
"""

import numpy as np
from PIL import Image, ImageFilter
import torch
from typing import Union, Tuple, Optional


def _get_fill_value(img: Image.Image, fill_mode: str = "per_image_mean") -> Tuple[int, int, int]:
    """
    Get the fill value based on fill_mode.

    Args:
        img: PIL Image (RGB)
        fill_mode: one of ["per_image_mean", "gray", "imagenet_mean"]

    Returns:
        (R, G, B) tuple of fill values
    """
    if fill_mode == "per_image_mean":
        # Compute per-channel mean for this image
        img_array = np.array(img)  # [H, W, 3]
        mean_rgb = img_array.mean(axis=(0, 1))
        return tuple(int(x) for x in mean_rgb)
    elif fill_mode == "gray":
        return (127, 127, 127)
    elif fill_mode == "imagenet_mean":
        # ImageNet mean in RGB order
        return (123, 117, 104)
    else:
        raise ValueError(f"Unknown fill_mode: {fill_mode}")


def _apply_blur(img: Image.Image, mask: Image.Image, blur_sigma: float) -> Image.Image:
    """
    Apply Gaussian blur to regions where mask is white (255).

    Args:
        img: original RGB image
        mask: binary mask (0=keep original, 255=blur)
        blur_sigma: Gaussian blur sigma

    Returns:
        Blurred image
    """
    blurred = img.filter(ImageFilter.GaussianBlur(radius=blur_sigma))
    # Composite: use original where mask is black, blurred where mask is white
    # PIL Image.composite(im1, im2, mask): im1 where mask=255, im2 where mask=0
    result = Image.composite(blurred, img, mask)
    return result


def border_only(
    img: Union[Image.Image, torch.Tensor],
    k: float,
    fill_mode: str = "per_image_mean",
    blur_sigma: Optional[float] = None
) -> Union[Image.Image, torch.Tensor]:
    """
    Keep border region, replace center region with fill/blur.

    Border region B_k is defined as:
        - Top: [0:bh, :]
        - Bottom: [H-bh:H, :]
        - Left: [:, 0:bw]
        - Right: [:, W-bw:W]
    where bh = round(k*H), bw = round(k*W)

    Args:
        img: PIL Image (RGB) or torch.Tensor [C, H, W]
        k: border width ratio (0 < k < 0.5)
        fill_mode: "per_image_mean", "gray", or "imagenet_mean"
        blur_sigma: if not None, use blur instead of fill

    Returns:
        Transformed image in same format as input
    """
    is_tensor = isinstance(img, torch.Tensor)
    if is_tensor:
        # Convert tensor to PIL for processing
        img = Image.fromarray((img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))

    W, H = img.size  # PIL uses (width, height)
    bw = round(k * W)
    bh = round(k * H)

    # Create a copy
    result = img.copy()

    # Define center region to be masked/blurred
    # Center: [bh:H-bh, bw:W-bw]
    center_box = (bw, bh, W - bw, H - bh)

    if blur_sigma is not None:
        # Use blur: create mask for center region
        mask = Image.new('L', (W, H), 0)  # All black (keep original)
        # Draw white rectangle on center region (to be blurred)
        from PIL import ImageDraw
        draw = ImageDraw.Draw(mask)
        draw.rectangle(center_box, fill=255)
        result = _apply_blur(img, mask, blur_sigma)
    else:
        # Use fill
        fill_rgb = _get_fill_value(img, fill_mode)
        # Paste filled rectangle on center
        fill_patch = Image.new('RGB', (center_box[2] - center_box[0], center_box[3] - center_box[1]), fill_rgb)
        result.paste(fill_patch, center_box[:2])

    if is_tensor:
        # Convert back to tensor
        result = torch.from_numpy(np.array(result)).permute(2, 0, 1).float() / 255.0

    return result


def center_only(
    img: Union[Image.Image, torch.Tensor],
    k: float,
    fill_mode: str = "per_image_mean",
    blur_sigma: Optional[float] = None
) -> Union[Image.Image, torch.Tensor]:
    """
    Keep center region, replace border region with fill/blur.

    Border region B_k (to be masked) includes:
        - Top: [0:bh, :]
        - Bottom: [H-bh:H, :]
        - Left: [:, 0:bw]
        - Right: [:, W-bw:W]

    Args:
        img: PIL Image (RGB) or torch.Tensor [C, H, W]
        k: border width ratio (0 < k < 0.5)
        fill_mode: "per_image_mean", "gray", or "imagenet_mean"
        blur_sigma: if not None, use blur instead of fill

    Returns:
        Transformed image in same format as input
    """
    is_tensor = isinstance(img, torch.Tensor)
    if is_tensor:
        img = Image.fromarray((img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))

    W, H = img.size
    bw = round(k * W)
    bh = round(k * H)

    result = img.copy()

    if blur_sigma is not None:
        # Create mask for border region
        mask = Image.new('L', (W, H), 0)
        from PIL import ImageDraw
        draw = ImageDraw.Draw(mask)
        # Top border
        draw.rectangle((0, 0, W, bh), fill=255)
        # Bottom border
        draw.rectangle((0, H - bh, W, H), fill=255)
        # Left border
        draw.rectangle((0, bh, bw, H - bh), fill=255)
        # Right border
        draw.rectangle((W - bw, bh, W, H - bh), fill=255)
        result = _apply_blur(img, mask, blur_sigma)
    else:
        # Fill border regions
        fill_rgb = _get_fill_value(img, fill_mode)
        # Convert to numpy for easier manipulation
        arr = np.array(result)
        # Top
        arr[:bh, :] = fill_rgb
        # Bottom
        arr[H-bh:, :] = fill_rgb
        # Left (excluding corners already filled)
        arr[bh:H-bh, :bw] = fill_rgb
        # Right (excluding corners already filled)
        arr[bh:H-bh, W-bw:] = fill_rgb
        result = Image.fromarray(arr)

    if is_tensor:
        result = torch.from_numpy(np.array(result)).permute(2, 0, 1).float() / 255.0

    return result


def edge_mask(
    img: Union[Image.Image, torch.Tensor],
    k: float,
    fill_mode: str = "per_image_mean",
    blur_sigma: Optional[float] = None
) -> Union[Image.Image, torch.Tensor]:
    """
    Mask edge/border region (semantically equivalent to center_only).

    This is an alias for center_only for clarity in experimental naming.

    Args:
        img: PIL Image (RGB) or torch.Tensor [C, H, W]
        k: border width ratio
        fill_mode: "per_image_mean", "gray", or "imagenet_mean"
        blur_sigma: if not None, use blur instead of fill

    Returns:
        Transformed image in same format as input
    """
    return center_only(img, k, fill_mode, blur_sigma)


def crop_resize(
    img: Union[Image.Image, torch.Tensor],
    k: float,
    out_size: Optional[Tuple[int, int]] = None,
    interpolation: str = "bilinear"
) -> Union[Image.Image, torch.Tensor]:
    """
    Crop out k% border and resize back to original (or specified) size.

    This simulates the effect of "zooming in" by removing border information.

    Args:
        img: PIL Image (RGB) or torch.Tensor [C, H, W]
        k: border crop ratio (0 < k < 0.5)
        out_size: output size (W, H); if None, use original size
        interpolation: "bilinear", "nearest", "bicubic"

    Returns:
        Cropped and resized image in same format as input
    """
    is_tensor = isinstance(img, torch.Tensor)
    if is_tensor:
        img = Image.fromarray((img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))

    W, H = img.size
    if out_size is None:
        out_size = (W, H)

    bw = round(k * W)
    bh = round(k * H)

    # Crop to center region [bw:W-bw, bh:H-bh]
    crop_box = (bw, bh, W - bw, H - bh)
    cropped = img.crop(crop_box)

    # Resize back to out_size
    interp_map = {
        "nearest": Image.NEAREST,
        "bilinear": Image.BILINEAR,
        "bicubic": Image.BICUBIC
    }
    resample = interp_map.get(interpolation, Image.BILINEAR)
    result = cropped.resize(out_size, resample=resample)

    if is_tensor:
        result = torch.from_numpy(np.array(result)).permute(2, 0, 1).float() / 255.0

    return result


# ============================================================================
# Test utilities (for debugging/validation)
# ============================================================================

def _test_transforms():
    """
    Quick sanity check for transformations.
    """
    # Create a simple test image (gradient pattern)
    W, H = 224, 224
    x = np.linspace(0, 255, W).astype(np.uint8)
    y = np.linspace(0, 255, H).astype(np.uint8)
    X, Y = np.meshgrid(x, y)

    # RGB gradient
    img_array = np.stack([X, Y, (X + Y) // 2], axis=-1).astype(np.uint8)
    img = Image.fromarray(img_array)

    k = 0.1

    # Test all transforms
    print("Testing border_only...")
    b_only = border_only(img, k, fill_mode="per_image_mean")
    assert b_only.size == img.size

    print("Testing center_only...")
    c_only = center_only(img, k, fill_mode="gray")
    assert c_only.size == img.size

    print("Testing edge_mask...")
    e_mask = edge_mask(img, k, blur_sigma=8.0)
    assert e_mask.size == img.size

    print("Testing crop_resize...")
    cropped = crop_resize(img, k, out_size=(224, 224))
    assert cropped.size == (224, 224)

    print("✅ All transform tests passed!")

    # Visual inspection (optional, save to file)
    # b_only.save("/tmp/border_only.png")
    # c_only.save("/tmp/center_only.png")
    # e_mask.save("/tmp/edge_mask.png")
    # cropped.save("/tmp/crop_resize.png")


if __name__ == "__main__":
    _test_transforms()
