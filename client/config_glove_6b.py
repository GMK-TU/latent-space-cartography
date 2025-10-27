#!/usr/bin/env python
# configurations of GloVe word embedding on Wikipedia data (6B)

dset = 'glove'
data_type = 'text'
dims = [50, 100, 200, 300]
metric = 'cosine'

# MySQL table schema
schema_meta = 'i, name'
schema_header = None

img_rows, img_cols, img_chns = None, None, None
img_mode = None