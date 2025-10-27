#!/usr/bin/env python
# configurations of GloVe word embedding on Wikipedia data (6B)

dset = 'glove'
data_type = 'text'
dims = [50]
metric = 'cosine'

# MySQL table schema
schema_meta = 'i, name'
schema_header = None
