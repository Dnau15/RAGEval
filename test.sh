micromamba activate rageval
python -c "
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

print('Step 1: importing torch...')
import torch
print(f'  torch {torch.__version__}, MPS={torch.backends.mps.is_available()}')

print('Step 2: importing SentenceTransformer...')
from sentence_transformers import SentenceTransformer
print('  OK')

print('Step 3: loading model...')
m = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
print('  OK')

print('Step 4: encoding 1 doc...')
e = m.encode(['test sentence'], convert_to_numpy=True)
print(f'  shape={e.shape}')

print('Step 5: encoding 100 docs...')
docs = ['test sentence number ' + str(i) for i in range(100)]
e2 = m.encode(docs, batch_size=16, convert_to_numpy=True)
print(f'  shape={e2.shape}')

print('ALL STEPS PASSED')
"