#!/usr/bin/env python3
"""QLoRA 구현 테스트"""

from qwen_advanced import AdvancedQwenConfig, AdvancedQwenLM, QLoRA, QLoRALinear
import torch

print('=== QLoRA 테스트 ===')
qlora = QLoRA(in_features=768, out_features=768, rank=8, quantize=True)
x = torch.randn(2, 10, 768)
out = qlora(x)
print(f'✅ QLoRA 순전파: {out.shape}')

print('\n=== QLoRALinear 테스트 ===')
qlora_linear = QLoRALinear(768, 768, use_qlora=True, quantize_weight=True)
out = qlora_linear(x)
print(f'✅ QLoRALinear 순전파: {out.shape}')

# 학습 파라미터 확인
params = qlora_linear.get_training_params()
print(f'✅ 학습 가능 파라미터: {list(params.keys())}')

print('\n=== 전체 모델 QLoRA 테스트 ===')
config = AdvancedQwenConfig(use_qlora=True, use_lora=True)
model = AdvancedQwenLM(config)
input_ids = torch.randint(0, 50000, (2, 10))
outputs = model(input_ids)
if isinstance(outputs, dict):
    logits = outputs['logits'] if 'logits' in outputs else outputs.get('output', None)
    print(f'✅ 모델 출력: {logits.shape}')
else:
    print(f'✅ 모델 출력: {outputs.shape}')
print(f'✅ Config use_qlora: {config.use_qlora}')
print(f'✅ Config qlora_nf4: {config.qlora_nf4}')

# 메모리 추정
print('\n=== 메모리 효율성 ===')
total_params = sum(p.numel() for p in model.parameters())
lora_params = 0
for name, param in model.named_parameters():
    if 'lora' in name:
        lora_params += param.numel()

print(f'전체 파라미터: {total_params:,}')
print(f'LoRA 파라미터: {lora_params:,}')
if lora_params > 0:
    print(f'학습 가능 비율: {lora_params/total_params*100:.2f}%')

print('\n🎉 QLoRA 구현 완료!')
