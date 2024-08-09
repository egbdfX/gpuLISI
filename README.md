# gpuLISI

We have developed a GPU-accelerated LISI. The CPU version of LISI is displayed on [LISI](https://github.com/egbdfX/Intensity-sensitive-IQAs?tab=readme-ov-file#lisi-lisipy). Please see our paper in Section [Reference](https://github.com/egbdfX/FastImagingTrigger/tree/main#reference) for more information.

## User guidance

**Step 1:**
Make sure GCCcore, CUDA, and CFITSIO are avaiable. If you see a warning saying ```/usr/bin/ld.gold: warning: /apps/system/easybuild/software/GCCcore/11.2.0/lib/gcc/x86_64-pc-linux-gnu/11.2.0/crtbegin.o: unknown program property type 0xc0010002 in .note.gnu.property section```, you would need to make sure Python is also available.

**Step 2:**
Run the Makefile by ```make```. Note that this Makefile is written for NVIDIA H100. If you are using other GPU, you would need to make sure the CUDA arch is matching.

**Step 3:**
Run the code by ```./sharedlibrary_gpu dif1.fits dif2.fits snap1.fits```, where ```dif1.fits``` and ```dif2.fits``` are the two difference images (FITS files), and ```snap1.fits``` is the reference snapshot image (FITS file). The three input images should have the same size.

**Step 4:**
The code will output a FITS file named ```output_tLISI.fits```, which is the output tLISI matrix.

## Contact
If you have any questions or need further assistance, please feel free to contact at [egbdfmusic1@gmail.com](mailto:egbdfmusic1@gmail.com).

## Reference

**When referencing this code, please cite our related paper:**

X. Li, K. Adámek, W. Armour, "GPU-accelerated fast imaging trigger for source localisation in transient detection," 2024.

## License

Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
