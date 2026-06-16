import urllib.request, os
from pathlib import Path

# Remove the old potentially corrupted file
old = Path('models/sam2_hiera_l.pt')
if old.exists():
    os.remove(old)
    print('Removed old weights.')

print('Downloading sam2_hiera_l weights...')
urllib.request.urlretrieve(
    'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt',
    'models/sam2_hiera_l.pt'
)
print('Done.')
