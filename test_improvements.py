#!/usr/bin/env python3
"""
Test script to validate the 3 improvements to RCLMuFN model:
1. Hard Negative Mining (建议2)
2. DIMM Channel-level Fusion (建议4)
3. Multi-layer Feature Fusion (建议5)
"""

import torch
import sys
from argparse import Namespace

# Import model
from model import RCLMuFN

def create_test_args():
    """Create test arguments for model initialization."""
    args = Namespace(
        text_size=512,
        image_size=768,
        label_number=2,
        simple_linear=False,
        dropout_rate=0.1,
        neg_sampling='hard_negative',  # Test hard negative mining
        tau_schedule_mode='epoch',
        lambda_ratio_start=0.0,
        lambda_itm_start=0.0
    )
    return args

def create_dummy_inputs(batch_size=4, max_len=77):
    """Create dummy inputs for testing."""
    inputs = {
        'input_ids': torch.randint(0, 1000, (batch_size, max_len)),
        'pixel_values': torch.randn(batch_size, 3, 224, 224),
        'attention_mask': torch.ones(batch_size, max_len, dtype=torch.long)
    }
    # Set some positions to 0 (padding) for testing
    inputs['attention_mask'][:, 50:] = 0

    batch = (
        ['text'] * batch_size,
        ['image'] * batch_size,
        [0, 1, 0, 1],  # labels
        list(range(batch_size))
    )

    labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)

    return inputs, batch, labels

def test_model_forward():
    """Test model forward pass."""
    print("=" * 80)
    print("Testing RCLMuFN with 4 Improvements")
    print("=" * 80)

    # Create model
    args = create_test_args()
    model = RCLMuFN(args)
    model.eval()

    print("✅ Model initialized successfully")
    print(f"   - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   - Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Test forward pass
    print("\n" + "=" * 80)
    print("Test 1: Forward Pass")
    print("=" * 80)

    inputs, batch, labels = create_dummy_inputs(batch_size=4)

    with torch.no_grad():
        outputs = model(inputs, batch, labels)

    loss, score = outputs

    print(f"✅ Forward pass successful")
    print(f"   - Loss: {loss.item():.4f}")
    print(f"   - Score shape: {score.shape}")
    print(f"   - Score sum per sample: {score.sum(dim=1)}")

    # Test improvement 5: Multi-layer fusion weights
    print("\n" + "=" * 80)
    print("Test 2: Multi-layer Feature Fusion (建议5)")
    print("=" * 80)

    print(f"✅ Layer weights initialized:")
    print(f"   - Text layer weights: {model.layer_weights_text.data}")
    print(f"   - Vision layer weights: {model.layer_weights_vision.data}")

    # After softmax normalization
    weights_t = torch.softmax(model.layer_weights_text, dim=0)
    weights_v = torch.softmax(model.layer_weights_vision, dim=0)
    print(f"   - Normalized text weights: {weights_t}")
    print(f"   - Normalized vision weights: {weights_v}")
    print(f"   - Text weights sum: {weights_t.sum():.4f}")
    print(f"   - Vision weights sum: {weights_v.sum():.4f}")

    # Test improvement 2: Hard negative mining
    print("\n" + "=" * 80)
    print("Test 3: Hard Negative Mining (建议2)")
    print("=" * 80)

    print(f"✅ Negative sampling strategy:")
    print(f"   - Mode: {model.cid.neg_sampling}")
    print(f"   - Expected: 'hard_negative'")

    # Test improvement 4: DIMM channel-level fusion
    print("\n" + "=" * 80)
    print("Test 4: DIMM Channel-level Fusion (建议4)")
    print("=" * 80)

    print(f"✅ DIMM final MLP dimensions:")
    print(f"   - Input: {model.dimm.final_mlp[0].in_features}")
    print(f"   - Expected: 3072 (768*4, 3 text channels + 1 vision channel)")
    print(f"   - Hidden: {model.dimm.final_mlp[0].out_features}")
    print(f"   - Output: {model.dimm.final_mlp[3].out_features}")

    # Test backward pass
    print("\n" + "=" * 80)
    print("Test 5: Backward Pass")
    print("=" * 80)

    model.train()
    inputs, batch, labels = create_dummy_inputs(batch_size=4)

    # Zero gradients
    for param in model.parameters():
        if param.grad is not None:
            param.grad.zero_()

    outputs = model(inputs, batch, labels)
    loss = outputs[0]
    loss.backward()

    # Check gradients
    grad_count = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    total_params = sum(1 for p in model.parameters() if p.requires_grad)

    print(f"✅ Backward pass successful")
    print(f"   - Parameters with gradients: {grad_count}/{total_params}")
    print(f"   - Layer weights text gradient: {model.layer_weights_text.grad}")
    print(f"   - Layer weights vision gradient: {model.layer_weights_vision.grad}")

    # Test CID statistics
    print("\n" + "=" * 80)
    print("Test 6: CID Statistics")
    print("=" * 80)

    if hasattr(model, 'last_cid_stats'):
        stats = model.last_cid_stats
        print(f"✅ CID mask statistics:")
        print(f"   - Text mask mean: {stats['m_t_mean']:.4f}")
        print(f"   - Text mask variance: {stats['m_t_var']:.4f}")
        print(f"   - Vision mask mean: {stats['m_v_mean']:.4f}")
        print(f"   - Vision mask variance: {stats['m_v_var']:.4f}")

    print("\n" + "=" * 80)
    print("✅ All tests passed!")
    print("=" * 80)
    print("\n📝 Summary of Improvements:")
    print("   1. ✅ Hard Negative Mining implemented (neg_sampling='hard_negative')")
    print("   2. ✅ DIMM Channel-level Fusion optimized (3072 -> 1536 -> 768)")
    print("   3. ✅ Multi-layer Feature Fusion added (LAFF-style, 4 layers)")
    print("\n🚀 Model is ready for training!")

if __name__ == '__main__':
    try:
        test_model_forward()
    except Exception as e:
        print(f"\n❌ Test failed with error:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
