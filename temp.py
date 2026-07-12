import torch
state = torch.load('models/dm_count_shb.pth', map_location='cpu', weights_only=False)
if isinstance(state, dict):
    print('Keys:', list(state.keys())[:10])
    print('Type:', type(state))
else:
    print('Direct state dict, keys:', list(state.keys())[:5])
print('Total keys:', len(state) if isinstance(state, dict) else 'N/A')
