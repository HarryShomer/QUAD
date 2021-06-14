# HyperRelKG

For obj training samples we have SAMPLES / BS = Batches. For qual training the number of samples is very unlikely to be the same. 
So we can modify BS for quals to ensrure that QUAL_SAMPLES / BS = Batch.
Need:
    - Own iterator (instead of both using self.i)
    - next() can return both batches

Relations or Quals...
- Index for subjects
- Labels for subjects
- No need to encode twice...instead have model return encoded x and r w/o passing to transformer
- Mask: see https://github.com/migalkin/StarE/blob/master/models/models_statements.py.
