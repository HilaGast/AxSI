AxSI pipeline written in MATLAB will be updated here.
AxSI_main.m contain the main script which call all others.
Resulted files:
- FA - Fractional Anisotropy as calculated using DTI.
- MD - Mean Diffusivity as calculated using DTI.
- ADD - averaged Axon Diameter Distribution (for each voxel)
- pfr - the probability of restricted tissue (for each voxel)
- ph - the probability of hindered tissue (for each voxel)
- pcsf - the probability of csf tissue (for each voxel)
- ADD_allvalues - the probability of each of 160 tested diameters in each voxel

If you use this analysis in your research, please quote:

Gast, H., Horowitz, A., Krupnik, R., Barazany, D., Lifshits, S., Ben-Amitay, S., & Assaf, Y. (2023). A Method for In-Vivo Mapping of Axonal Diameter Distributions in the Human Brain Using Diffusion-Based Axonal Spectrum Imaging (AxSI). Neuroinformatics, 1-14.
