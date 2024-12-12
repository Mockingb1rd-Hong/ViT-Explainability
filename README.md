# ViT-Explainability

This repository is based on the code and methodology from Hila Chefer's [Transformer-MM-Explainability](https://github.com/hila-chefer/Transformer-MM-Explainability), which provides a framework for explainability in bi-modal and encoder-decoder transformers.

Install all the components in requirements.txt
pip install -r requirements.txt

Attention visualization
nicetry.py

Positive perturbation test
get_result.py

Methods are all in my_explain.py

Infos of image information in valid.json
[valid.json](https://nlp.cs.unc.edu/data/lxmert_data/vqa/valid.json)

dataset: MS_COCO_val2014

## Citation

If you use or reference this code in your work, please cite the original paper:

```bibtex
@InProceedings{Chefer_2021_ICCV,
   author    = {Chefer, Hila and Gur, Shir and Wolf, Lior},
   title     = {Generic Attention-Model Explainability for Interpreting Bi-Modal and Encoder-Decoder Transformers},
   booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
   month     = {October},
   year      = {2021},
   pages     = {397-406}
}

This version uses code block formatting for the BibTeX citation, making it easier to read and ensuring it displays correctly in Markdown viewers.





