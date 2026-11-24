# Data inspection report

## Counts

| Class | Original | Expected | Augmented | Status |
|---|---:|---:|---:|---|
| central_serous_chorioretinopathy | 101 | 101 | 606 | OK |
| diabetic_retinopathy | 1509 | 1509 | 3444 | OK |
| disc_edema | 127 | 127 | 762 | OK |
| glaucoma | 1349 | 1349 | 2880 | OK |
| healthy | 1024 | 1024 | 2676 | OK |
| macular_scar | 444 | 444 | 1937 | OK |
| myopia | 500 | 500 | 2251 | OK |
| retinal_detachment | 125 | 125 | 750 | OK |
| retinitis_pigmentosa | 139 | 139 | 834 | OK |
| pterygium | 17 | 17 | 102 | OK |

## Image details

### central_serous_chorioretinopathy
- Samples: CSCR1.jpg, CSCR10.jpg, CSCR100.jpg, CSCR101.jpg, CSCR11.jpg
- Extensions: {'.jpg': 101}
- Size min/median/max: W 2004/2004/2004, H 1690/1690/1690
### diabetic_retinopathy
- Samples: DR1.jpg, DR10.jpg, DR100.jpg, DR1000.jpg, DR1001.jpg
- Extensions: {'.jpg': 1509}
- Size min/median/max: W 2004/2004/2004, H 1690/1690/1690
### disc_edema
- Samples: Disc Edema1.jpg, Disc Edema10.jpg, Disc Edema100.jpg, Disc Edema101.jpg, Disc Edema102.jpg
- Extensions: {'.jpg': 127}
- Size min/median/max: W 2004/2004/2004, H 1690/1690/1690
### glaucoma
- Samples: Glaucoma1.jpg, Glaucoma10.jpg, Glaucoma100.jpg, Glaucoma1000.jpg, Glaucoma1001.jpg
- Extensions: {'.jpg': 1349}
- Size min/median/max: W 2004/2004/2004, H 1690/1690/1690
### healthy
- Samples: Healthy1.jpg, Healthy10.jpg, Healthy100.jpg, Healthy1000.jpg, Healthy1001.jpg
- Extensions: {'.jpg': 1024}
- Size min/median/max: W 2004/2004/2004, H 1690/1690/1690
### macular_scar
- Samples: Macular Scar1.jpg, Macular Scar10.jpg, Macular Scar100.jpg, Macular Scar101.jpg, Macular Scar102.jpg
- Extensions: {'.jpg': 444}
- Size min/median/max: W 2004/2004/2004, H 1690/1690/1690
### myopia
- Samples: Myopia1.jpg, Myopia10.jpg, Myopia100.jpg, Myopia101.jpg, Myopia102.jpg
- Extensions: {'.jpg': 500}
- Size min/median/max: W 2004/2004/2004, H 1690/1690/1690
### retinal_detachment
- Samples: Retinal Detachment1.jpg, Retinal Detachment10.jpg, Retinal Detachment100.jpg, Retinal Detachment101.jpg, Retinal Detachment102.jpg
- Extensions: {'.jpg': 125}
- Size min/median/max: W 2004/2004/2004, H 1690/1690/1690
### retinitis_pigmentosa
- Samples: Retinitis Pigmentosa1.jpg, Retinitis Pigmentosa10.jpg, Retinitis Pigmentosa100.jpg, Retinitis Pigmentosa101.jpg, Retinitis Pigmentosa102.jpg
- Extensions: {'.jpg': 139}
- Size min/median/max: W 2004/2004/2004, H 1690/1690/1690
### pterygium
- Samples: Pterygium1.jpg, Pterygium10.jpg, Pterygium11.jpg, Pterygium12.jpg, Pterygium13.jpg
- Extensions: {'.jpg': 17}
- Size min/median/max: W 2004/2004/2004, H 1690/1690/1690

## Integrity

- Corrupt/unreadable original images: 0
- Exact duplicate files within original: 484

## Augmentation linkage

- Linked 13109/16140 non-Pterygium augmented images (81.22%).
- Link method: exact filename stem first; remaining images use DCT perceptual hashes compared against original flip/rotation variants (Hamming distance ≤10).
- central_serous_chorioretinopathy: 415/606 (68.48%) linked.
- diabetic_retinopathy: 3027/3444 (87.89%) linked.
- disc_edema: 556/762 (72.97%) linked.
- glaucoma: 2454/2880 (85.21%) linked.
- healthy: 2247/2676 (83.97%) linked.
- macular_scar: 1421/1937 (73.36%) linked.
- myopia: 1785/2251 (79.30%) linked.
- retinal_detachment: 610/750 (81.33%) linked.
- retinitis_pigmentosa: 594/834 (71.22%) linked.

## WARNING

**Augmented-to-original linkage is below 95%; the splitter will exclude augmented files and use on-the-fly augmentation only.**

## Limitations

Near-duplicate checks use transform-tolerant perceptual hashes. Patient identifiers are unavailable, so patient-level separation cannot be verified.
