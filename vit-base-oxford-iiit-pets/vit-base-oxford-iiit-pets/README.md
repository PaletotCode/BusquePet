---
library_name: transformers
license: apache-2.0
base_model: google/vit-base-patch16-224
tags:
- image-classification
- generated_from_trainer
- zero-shot-image-classification
metrics:
- accuracy
model-index:
- name: vit-base-oxford-iiit-pets
  results: []
---

# vit-base-oxford-iiit-pets

This model is a fine-tuned version of [google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224) on the pcuenq/oxford-pets dataset.
It achieves the following results on the evaluation set:
- Loss: 0.1924
- Accuracy: 0.9445


## Model description

This model is a fine-tuned version of a pre-trained Vision Transformer (`google/vit-base-patch16-224`) for image classification on the Oxford-IIIT Pet Dataset. 
It uses transfer learning to adapt a generic vision model to identify 37 different cat and dog breeds. 
The model head is adjusted to output the number of classes in the dataset, and it is trained end-to-end using standard classification loss.

---

## Intended uses & limitations

**Intended Uses:**
- Educational demos on transfer learning and fine-tuning vision models.
- Pet breed classification in structured datasets similar to Oxford Pets.
- Comparative analysis with zero-shot models like CLIP.

**Limitations:**
- May not generalize well to breeds outside of the Oxford-IIIT dataset.
- Not suitable for real-world medical or safety-critical applications.
- Input images should be clear, centered, and close in style to the training data (cropped pet portraits).


## Training and evaluation data

The model is trained and evaluated on the [Oxford-IIIT Pet Dataset](https://huggingface.co/datasets/pcuenq/oxford-pets), which contains 7,349 images of cats and dogs spanning 37 different breeds. The dataset includes equal representation of pets and was split into training, validation, and test sets. Evaluation metrics used include accuracy, precision, and recall.


## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0003
- train_batch_size: 16
- eval_batch_size: 8
- seed: 42
- optimizer: Use adamw_torch with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: linear
- num_epochs: 5

### Training results

| Training Loss | Epoch | Step | Validation Loss | Accuracy |
|:-------------:|:-----:|:----:|:---------------:|:--------:|
| 0.3716        | 1.0   | 370  | 0.3013          | 0.9242   |
| 0.2048        | 2.0   | 740  | 0.2342          | 0.9310   |
| 0.1764        | 3.0   | 1110 | 0.2124          | 0.9350   |
| 0.1617        | 4.0   | 1480 | 0.2050          | 0.9350   |
| 0.1235        | 5.0   | 1850 | 0.2032          | 0.9350   |

## Zero-Shot Classification Evaluation (CLIP)

Evaluated the Oxford-IIIT Pet dataset using a **zero-shot image classification model**: [`openai/clip-vit-base-patch32`](https://huggingface.co/openai/clip-vit-base-patch32). 
Instead of training, the CLIP model was evaluated using a list of breed names (e.g., "Siamese", "Persian", "Chihuahua") as candidate labels for zero-shot classification.

### Evaluation Results:
- Accuracy: 0.8800
- Precision: 0.8768
- Recall: 0.8800

![image/png](https://cdn-uploads.huggingface.co/production/uploads/67cdb999dedccb89c84f908d/Xvso3JiEQFxKUT0ZwYYkE.png)

### Framework versions

- Transformers 4.50.0
- Pytorch 2.6.0+cu124
- Datasets 3.4.1
- Tokenizers 0.21.1