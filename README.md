# Deep Wilcoxon signed-rank test for Covid-19 classfication

## Overview 

![teaser](figures/flowchart.png)



The proposed method DWCC (Deep Wilcoxon signed-rank test for Covid-19 classfication) is for AI-enabled Medical Image Analysis Workshop and Covid-19 Diagnosis Competition (MIA-COV19D).

### Result Visualization

Negative CT scan:

![teaser](figures/neg.gif)

Positive CT scan:

![teaser](figures/pos.gif)

### Pretrained Model
[DWCC](https://www.dropbox.com/s/qkfy2q7a3r3kflm/DWCC.zip?dl=0)

[DWCC fine-tuning](https://www.dropbox.com/s/wzbg7w44qs8erjl/DWCC_finetuning.zip?dl=0)

---
### Prerequisites:

- Python 3.6
- timm 0.4.9
- scipy 1.5.4
- pytorch 1.7.1
- torchvision 0.8.2
- albumentations 1.0.0

<br/>

---
### Folder Structure

>```data/```             &nbsp; - &nbsp; dataset and dataloader <br/>
>```models/```      &nbsp; - &nbsp; Swin-Transformer <br/>

>```args.py```     &nbsp; - &nbsp; args<br/>
>```evaluation.py```            &nbsp; - &nbsp; evaluation for validation set when training <br/>
>```train.py```      &nbsp; - &nbsp; train the proposed model <br/>
>```test_inference.ipynb```       &nbsp; - &nbsp; inference for testing set <br/>
>```make_example.ipynb```     &nbsp; - &nbsp;  visualization for predicted result<br/>

<br/>

---
### Training

#### Start training
To train the model, use the following command:

```bash
python train.py path=covid/data epoch=100 train_batch=10 train_ct_batch=16 val_batch=8 lr=0.0001
```

Optional parameters (and default values):

>```path```: **```covid/data```** &nbsp; - &nbsp; root path for Covid-19 dataset<br/>
>```epoch```:  100&nbsp; - &nbsp; training epoch <br/>```train_batch```:  10&nbsp; - &nbsp;image batch size per CT scan <br/>```train_ct_batch```:  16&nbsp; - &nbsp;CT scan batch per iteration <br/>```val_batch```:  8&nbsp; - sample image size when evaluation <br/>```lr```:  0.0001&nbsp; - &nbsp; learning rate for AdamW optimizer <br/>

<br/>

---
### Reference

* [SwinTransformer](https://github.com/microsoft/Swin-Transformer)

