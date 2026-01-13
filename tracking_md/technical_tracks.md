### 07/01/2026
> Torch issues : not detected 
environment renewal.
- `env.yaml` building + `cuda118` creation
- command run in addition : 
```bash
conda activate cuda118
conda env config vars set MKL_INTERFACE_LAYER=LP64
conda deactivate
conda activate cuda118
```