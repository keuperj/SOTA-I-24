# SOTA-I-24
Materials for the ML SOTA course at HSO in SS24

<details>
<summary> <H2> Week 1 </H2><BR>
Image Classification
</summary>

* BOARD: [https://zoom-x.de/wb/doc/e0_IXOFrS7S3y4Q-Td-APA](https://zoom-x.de/wb/doc/e0_IXOFrS7S3y4Q-Td-APA)


### SotA Links + Materials
* [arxiv.org Preprints](https://arxiv.org/)
    * [Arxiv tag](https://arxivtag.com/)
    * [DL Monitor](https://deeplearn.org/)   
* [Scholar Inbox](https://www.scholar-inbox.com/)
* [AK on Twitter](https://twitter.com/_akhaliq)
* [Papers with Code](https://paperswithcode.com/sota)
* [Hugging Face](https://huggingface.co/models)
* [Zotero](https://www.zotero.org/)

### Image Classification
#### Benchmarks
* [ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)
* [ImageNet100](https://paperswithcode.com/sota/image-classification-on-imagenet-100)
*  ...
  
#### Baseline Models
* ResNet
   * [paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)
   * [code](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)
* Transformer
   * [paper](https://openreview.net/pdf?id=YicbFdNTTy)
   * [code](https://github.com/lucidrains/vit-pytorch) 

#### SOTA CNN
* [ConvNext v2](https://openaccess.thecvf.com/content/CVPR2023/papers/Woo_ConvNeXt_V2_Co-Designing_and_Scaling_ConvNets_With_Masked_Autoencoders_CVPR_2023_paper.pdf)
   * [code](https://github.com/facebookresearch/ConvNeXt-V2)
   * [ConvNext v1](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.pdf)

#### SOTA Transformer
* [Swin v2](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_Swin_Transformer_V2_Scaling_Up_Capacity_and_Resolution_CVPR_2022_paper.pdf)
   * [code](https://github.com/microsoft/Swin-Transformer) 

#### Leader Board
* [OmniVec](https://openaccess.thecvf.com/content/WACV2024/papers/Srivastava_OmniVec_Learning_Robust_Representations_With_Cross_Modal_Sharing_WACV_2024_paper.pdf)

</details>
<details>
<summary> <H2> Week 2 </H2><BR>
Image Classification with Foundation Models
</summary>

### Backbones
* [ConvNext v2](https://openaccess.thecvf.com/content/CVPR2023/papers/Woo_ConvNeXt_V2_Co-Designing_and_Scaling_ConvNets_With_Masked_Autoencoders_CVPR_2023_paper.pdf)
   * [code](https://github.com/facebookresearch/ConvNeXt-V2) 
* [Clipp V2](https://arxiv.org/pdf/2306.15658.pdf)
   * [code](https://github.com/UCSC-VLAA/CLIPA)
   * [CLIP v1 paper](https://arxiv.org/pdf/2103.00020.pdf)
* [Dino V2](https://arxiv.org/pdf/2304.07193.pdf)
   * [code](https://github.com/facebookresearch/dinov2)
   * [DINO V1 paper](https://arxiv.org/pdf/2104.14294.pdf)

### Self-Supervised 
* [Masked AutoEncoder](https://arxiv.org/pdf/2111.06377.pdf)

### SOTA FM Classification
* [Battle of the Backbones](https://openreview.net/pdf?id=1yOnfDpkVe)
   * [code](https://github.com/hsouri/Battle-of-the-Backbones)
* [ConvNet vs Transformer, Supervised vs CLIP: Beyond ImageNet Accuracy](https://arxiv.org/pdf/2311.09215.pdf)
   * [code](https://github.com/kirill-vish/Beyond-INet) 

</details>
<details>
<summary> <H2> Week 3 </H2><BR>
Object Detection
</summary>

### Classic Methods
* [Hough Forests](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=99582ce8439dce17d9d6f74eb54fc5c89dbe06d9)
* [Deformable Part Models](https://cs.brown.edu/people/pfelzens/papers/lsvm-pami.pdf)

### Yolo 
* [Yolo V1](https://openaccess.thecvf.com/content_cvpr_2016/papers/Redmon_You_Only_Look_CVPR_2016_paper.pdf)
   * [original DarkNet code](https://github.com/pjreddie/darknet)
   * [Pytorch code (not official)](https://github.com/tanjeffreyz/yolo-v1)    
* Yolo V8 has no paper! -> just a GitHub page 
   * [code](https://github.com/ultralytics/ultralytics)
* [Yolo V1 to V8 overview](https://arxiv.org/pdf/2304.00501.pdf)

### SOTA 
* [Benchmark: MSCoco](https://paperswithcode.com/sota/object-detection-on-coco-minival)
* [SOTA paper: DETRs with Collaborative Hybrid Assignments Training](https://openaccess.thecvf.com/content/ICCV2023/papers/Zong_DETRs_with_Collaborative_Hybrid_Assignments_Training_ICCV_2023_paper.pdf)
   * [code](https://github.com/Sense-X/Co-DETR)
   * [background: DETR paper](https://arxiv.org/pdf/2005.12872.pdf)
   * [background: Faster-R-CNN paper](https://arxiv.org/pdf/1506.01497.pdf)
 
</details>

<details>
<summary> <H2> Week 4 </H2><BR>
Segmentation I
</summary>

### Benchmarks
* [MS-COCO](https://paperswithcode.com/sota/instance-segmentation-on-coco)
     * [website](https://cocodataset.org/#home)
* [CityScapes](https://paperswithcode.com/sota/semantic-segmentation-on-cityscapes)
     * [website](https://www.cityscapes-dataset.com/dataset-overview/)

### Baseline Model
* [U-Net](https://arxiv.org/pdf/1505.04597v1.pdf)
     * [PyTorch Code](https://github.com/milesial/Pytorch-UNet)
     * [Annotated Code](https://nn.labml.ai/unet/index.html)

### SOTA
* [#1 MS-COCO: EVA: Exploring the Limits of Masked Visual Representation Learning at Scale](https://openaccess.thecvf.com/content/CVPR2023/papers/Fang_EVA_Exploring_the_Limits_of_Masked_Visual_Representation_Learning_at_CVPR_2023_paper.pdf)
     * [code](https://github.com/baaivision/EVA/tree/master/EVA-01)
* [#3 ScityScapes: InternImage: Exploring Large-Scale Vision Foundation Models with
Deformable Convolutions](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_InternImage_Exploring_Large-Scale_Vision_Foundation_Models_With_Deformable_Convolutions_CVPR_2023_paper.pdf)
     * [code](https://github.com/OpenGVLab/InternImage)


</details>
<details>
<summary> <H2> Week 5 </H2><BR>
Segmentation II
</summary>

### SOTA
* [Segment Anything (SAM)](https://arxiv.org/pdf/2304.02643.pdf)
     * [code](https://github.com/facebookresearch/segment-anything)
     * [Demo](https://segment-anything.com/demo)
     * [Colab Tutorial](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/how-to-segment-anything-with-sam.ipynb)    
* [Segment Everything Everywhere All at Once (SEEM)](https://openreview.net/pdf?id=UHBrWeFWlL)
     * [code](https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once)
* [Segment Like Me (Slime)](https://arxiv.org/pdf/2309.03179.pdf)
     * [code](https://github.com/aliasgharkhani/SLiMe)
     * [Colab Demo](https://colab.research.google.com/drive/1fpKx6b2hQGEx1GK269vOw_sKeV9Rpnuj?usp=sharing)

</details>

<details>
<summary> <H2> Week 6 </H2><BR>
Depth Estimation 
</summary>
   
### Overview


### Benchmark Monocular
* [NYU-v2 leaderboard](https://paperswithcode.com/sota/monocular-depth-estimation-on-nyu-depth-v2)
* [NYU-v2 website](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html)

### SOTA Monocular
* [Depth Anything](https://arxiv.org/pdf/2401.10891v2.pdf)
   * [code](https://depth-anything.github.io/) 
* [UniDepth - CVPR '24 + NYU-v2 #1](https://arxiv.org/pdf/2403.18913v1.pdf)
   * [code](https://github.com/lpiccinelli-eth/unidepth)    

</details>
<details>
<summary> <H2> Week 7 - Visual Question Answering </H2><BR>

### Benchmark
* [VQA v2](https://visualqa.org/)
   * [VQA paper](https://arxiv.org/pdf/1505.00468)
   * [VQU v2 paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Goyal_Making_the_v_CVPR_2017_paper.pdf)
   * [Papers with Coder Leaderboard](https://paperswithcode.com/sota/visual-question-answering-on-vqa-v2-test-dev)
 
### SOTA Paper
* [PALI: A JOINTLY-SCALED MULTILINGUAL LANGUAGE-IMAGE MODEL](https://openreview.net/pdf?id=mWVoBz4W0u)
   * [code](https://github.com/kyegomez/PALI) 
* [Image as a Foreign Language: BEiT Pretraining for All Vision and Vision-Language Tasks ](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Image_as_a_Foreign_Language_BEiT_Pretraining_for_Vision_and_CVPR_2023_paper)
   * [code](https://github.com/microsoft/unilm/tree/master/beit3)
   * [BEiT v1](https://openreview.net/pdf?id=p-BhZSz59o4)
   * [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805)


</details>

### Week 8 - Genearitve Models I - SOTA GANs 

### Benchmark
* [ImageNet 512x512](https://paperswithcode.com/sota/image-generation-on-imagenet-512x512)
* [Flickr-Faces-HQ (FFHQ)](https://paperswithcode.com/sota/image-generation-on-ffhq-256-x-256)
   * [Website](https://github.com/NVlabs/ffhq-dataset) 
#### FID Score
* [Paper](https://proceedings.neurips.cc/paper_files/paper/2017/file/8a1d694707eb0fefe65871369074926d-Paper.pdf)
* [Problems with FID](https://openreview.net/pdf?id=mLG96UpmbYz)  

### GAN overview
* [2024 Overview paper](https://iopscience.iop.org/article/10.1088/2632-2153/ad1f77/pdf)
  
### GAN SOTA
* [StyleGAN v2](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123670171.pdf)
   * [code](https://github.com/EvgenyKashin/stylegan2-distillation)
   * [StyleGAN v1](https://arxiv.org/pdf/1812.04948)
* [SAN](https://arxiv.org/pdf/2301.12811v4)
   * [code](https://github.com/sony/san)


