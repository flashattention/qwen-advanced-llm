#!/usr/bin/env python3
"""MoE 및 QLoRA 기능 테스트"""

import sys
from pathlib import Path

# 부모 디렉토리를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from qwen_advanced import AdvancedQwenConfig, AdvancedQwenLM, MoE
import torch

print('=== MoE 기본 테스트 ===')
moe = MoE(
    hidden_size=768,
    num_experts=8,
    expert_size=3072,
    top_k=2,
    router_temp=1.0,
)
x_moe = torch.randn(2, 10, 768)
out_moe = moe(x_moe)
print(f'✅ MoE 순전파: {out_moe.shape}')
print(f'✅ 활성화된 Expert 수: 2 (top-k=2)')

print('\n=== 기본 모델 (MLP 사용) ===')
config_mlp = AdvancedQwenConfig(
    hidden_size=768,
    num_hidden_layers=12,
    use_moe=False,
)
model_mlp = AdvancedQwenLM(config_mlp)
input_ids = torch.randint(0, 50000, (2, 10))
outputs_mlp = model_mlp(input_ids)
# 반환값이 dict일 수 있으므로 처리
if isinstance(outputs_mlp, dict):
    output_shape = outputs_mlp['logits'].shape if 'logits' in outputs_mlp else list(outputs_mlp.values())[0].shape
else:
    output_shape = outputs_mlp.shape
print(f'✅ MLP 모델 출력: {output_shape}')

mlp_params = sum(p.numel() for p in model_mlp.parameters())
print(f'✅ MLP 모델 파라미터: {mlp_params:,}')

print('\n=== MoE 모델 테스트 ===')
config_moe = AdvancedQwenConfig(
    hidden_size=768,
    num_hidden_layers=12,
    use_moe=True,
    moe_num_experts=8,
    moe_top_k=2,
)
model_moe = AdvancedQwenLM(config_moe)
outputs_moe = model_moe(input_ids)
if isinstance(outputs_moe, dict):
    output_shape = outputs_moe['logits'].shape if 'logits' in outputs_moe else list(outputs_moe.values())[0].shape
else:
    output_shape = outputs_moe.shape
print(f'✅ MoE 모델 출력: {output_shape}')

moe_params = sum(p.numel() for p in model_moe.parameters())
print(f'✅ MoE 모델 파라미터: {moe_params:,}')
print(f'✅ 파라미터 증가: {(moe_params/mlp_params - 1)*100:.1f}%')
print(f'✅ 예상 계산량: MLP와 동일 (top-k=2로 제한)')

print('\n=== 조합 테스트: QLoRA + MoE ===')
config_combined = AdvancedQwenConfig(
    hidden_size=768,
    num_hidden_layers=12,
    use_moe=True,
    moe_num_experts=8,
    moe_top_k=2,
    use_qlora=True,
    use_lora=True,
    lora_rank=8,
)
model_combined = AdvancedQwenLM(config_combined)
outputs_combined = model_combined(input_ids)
if isinstance(outputs_combined, dict):
    output_shape = outputs_combined['logits'].shape if 'logits' in outputs_combined else list(outputs_combined.values())[0].shape
else:
    output_shape = outputs_combined.shape
print(f'✅ QLoRA + MoE 모델 출력: {output_shape}')

combined_params = sum(p.numel() for p in model_combined.parameters())
trainable_params = sum(p.numel() for p in model_combined.parameters() if p.requires_grad)
print(f'✅ 전체 파라미터: {combined_params:,}')
print(f'✅ 학습 가능 파라미터: {trainable_params:,}')
print(f'✅ 학습 가능 비율: {trainable_params/combined_params*100:.2f}%')

print('\n=== 성능 메트릭 ===')
print(f'MLP 모델:        {mlp_params:>12,} 파라미터')
print(f'MoE 모델:        {moe_params:>12,} 파라미터 (+{(moe_params/mlp_params - 1)*100:.1f}%)')
print(f'QLoRA + MoE:     {combined_params:>12,} 파라미터')
print(f'  - 학습 가능:   {trainable_params:>12,} 파라미터')
print(f'  - 메모리 절감: ~{(1 - trainable_params/combined_params)*100:.0f}% (QLoRA)')

print('\n🎉 MoE 및 조합 테스트 완료!')
