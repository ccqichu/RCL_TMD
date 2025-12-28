import torch
from model import RCLMuFN

class Args:
    def __init__(self):
        self.device = '0'
        self.model = 'RCLMuFN'
        self.text_name = 'text_final'
        self.simple_linear = False
        self.num_train_epochs = 10
        self.train_batch_size = 4
        self.dev_batch_size = 4
        self.label_number = 2
        self.text_size = 512
        self.image_size = 768
        self.adam_epsilon = 1e-8
        self.optimizer_name = 'adam'
        self.learning_rate = 5e-4
        self.clip_learning_rate = 1e-6
        self.max_len = 77
        self.layers = 3
        self.max_grad_norm = 5.0
        self.weight_decay = 0.05
        self.warmup_proportion = 0.2
        self.dropout_rate = 0.1
        self.output_dir = '../output_dir/'
        self.limit = None
        self.seed = 42
        self.lambda_ratio_start = 0.0
        self.lambda_ratio_end = 1e-3
        self.lambda_itm_start = 0.0
        self.lambda_itm_end = 1e-3
        self.lambda_warmup_epochs = 1
        self.lambda_ramp_epochs = 3
        self.lambda_schedule = 'linear'
        self.neg_sampling = 'label_aware'
        self.tau_schedule_mode = 'epoch'

args = Args()

print('=' * 60)
print('Checking parameter reduction after removing post-fusion attention')
print('=' * 60)

model = RCLMuFN(args)

# Count total parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'\n📊 Total parameters: {total_params:,}')
print(f'📊 Trainable parameters: {trainable_params:,}')

# Check if post_gate exists
has_post_gate = hasattr(model, 'post_gate')
has_post_ln = hasattr(model, 'post_ln')

print(f'\n✅ post_gate removed: {not has_post_gate}')
print(f'✅ post_ln retained: {has_post_ln}')

# Estimate parameter reduction
# post_gate was Linear(768*2, 768) = 1536*768 + 768 = 1,180,416 params
estimated_reduction = 1536 * 768 + 768
print(f'\n📉 Estimated parameter reduction: ~{estimated_reduction:,} params')
print(f'📉 Percentage reduction: ~{(estimated_reduction/trainable_params)*100:.2f}%')

print('\n' + '=' * 60)
print('✅ Parameter check complete!')
print('=' * 60)
