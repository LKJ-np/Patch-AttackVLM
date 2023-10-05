

## Prepare the VLMs in LAVIS Lib

There are two steps of adversarial attack for VLMs: (1) transfer-based attacking strategy and (2) query-based attacking strategy for the further improvement.

### Building a suitable LAVIS environment
```
conda create -n lavis python=3.8
conda activate lavis

git clone https://github.com/salesforce/LAVIS.git
cd LAVIS
pip install -e .
```
or following the steps [HERE](https://github.com/salesforce/LAVIS), and you can refer to the [ModelZoo](https://opensource.salesforce.com/LAVIS//latest/getting_started.html#model-zoo) for the possible model candidates.

## <b> Example: BLIP </b>

Here, we use BLIP for an example. For other models supported in the LAVIS library, please refer to their ```bash``` script (BLIP2, Img2Prompt, etc.) with similar commands as BLIP.
### Transfer-based attacking strategy

```
bash adv_img_transfer_blip.sh
```
the crafted adv images x_trans will be stored in `../_output_img/name_of_your_output_img_folder`. Then, we perform image-to-text and store the generated response of x_trans. This can be achieved by:

```
bash img2txt_blip.sh
```
where the generated responses will be stored in `./output_unidiffuser/name_of_your_output_txt_file.txt`. We will use them for pseudo-gradient estimation via RGF-estimator.

### Query-based attacking strategy (via RGF-estimator)

```
bash adv_img_query_blip.sh
```



# Bibtex
If you find this project useful in your research, please consider citing our paper:

```
@article{zhao2023evaluate,
  title={On Evaluating Adversarial Robustness of Large Vision-Language Models},
  author={Zhao, Yunqing and Pang, Tianyu and Du, Chao and Yang, Xiao and Li, Chongxuan and Cheung, Ngai-Man and Lin, Min},
  journal={arXiv preprint arXiv:2305.16934},
  year={2023}
}
```

Meanwhile, a relevant research that aims to [Embedding a Watermark to (multi-modal) Diffusion Models](https://github.com/yunqing-me/WatermarkDM):
```
@article{zhao2023recipe,
  title={A Recipe for Watermarking Diffusion Models},
  author={Zhao, Yunqing and Pang, Tianyu and Du, Chao and Yang, Xiao and Cheung, Ngai-Man and Lin, Min},
  journal={arXiv preprint arXiv:2303.10137},
  year={2023}
}
```