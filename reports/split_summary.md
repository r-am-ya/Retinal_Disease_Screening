# Split summary

Original images were stratified per class with seed 42 into approximate 70/15/15 train/validation/test partitions. Exact perceptual-hash duplicate clusters were kept intact. Patient IDs are unavailable, so patient-level leakage cannot be ruled out.

Augmentation linkage was below 95%; no augmented image was used.

| Class | Train | Validation | Test | Augmented train | Augmented % train |
|---|---:|---:|---:|---:|---:|
| central_serous_chorioretinopathy | 71 | 15 | 15 | 0 | 0.00% |
| diabetic_retinopathy | 1056 | 227 | 226 | 0 | 0.00% |
| disc_edema | 89 | 19 | 19 | 0 | 0.00% |
| glaucoma | 944 | 203 | 202 | 0 | 0.00% |
| healthy | 717 | 154 | 153 | 0 | 0.00% |
| macular_scar | 311 | 67 | 66 | 0 | 0.00% |
| myopia | 350 | 75 | 75 | 0 | 0.00% |
| retinal_detachment | 87 | 19 | 19 | 0 | 0.00% |
| retinitis_pigmentosa | 97 | 21 | 21 | 0 | 0.00% |
