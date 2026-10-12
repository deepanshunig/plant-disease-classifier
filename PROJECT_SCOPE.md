# Plant disease classifier: project scope

Scope established on 12 October 2026. The implemented project and full held-out results are documented in [EVALUATION.md](EVALUATION.md).

## Objective

Build an image classifier and a local upload interface that predicts a crop and its leaf condition from one clear leaf photograph. The final deliverables are a trained model, an evaluation report, and a working Streamlit application.

Example: a tomato leaf photograph produces a label such as `Tomato___Early_blight`, displayed as "Tomato - Early blight". This example describes the output format, not a prediction from a trained model.

## Input and output

- Input: one JPEG or PNG photograph with a clearly visible leaf. Decode the file, validate that it is an image, and convert it to RGB.
- Model input: a normalized tensor with three color channels and a spatial size of 224 x 224.
- Output: one predicted crop-condition class, its readable crop and condition names, and the top three class scores.
- The application displays the uploaded image and an uncertainty message when a validation-selected rule calls for it. Scores describe the model's output; they are not guarantees of correctness.

## Supported classes

The planned scope is the 38 classes below, covering 14 crops. "Healthy" is supported only where it appears in this table.

| Crop | Supported conditions | Class count |
|---|---|---:|
| Apple | Apple scab; black rot; cedar apple rust; healthy | 4 |
| Blueberry | Healthy | 1 |
| Cherry, including sour cherry | Powdery mildew; healthy | 2 |
| Corn / maize | Cercospora leaf spot / gray leaf spot; common rust; northern leaf blight; healthy | 4 |
| Grape | Black rot; Esca / black measles; leaf blight / Isariopsis leaf spot; healthy | 4 |
| Orange | Huanglongbing / citrus greening | 1 |
| Peach | Bacterial spot; healthy | 2 |
| Bell pepper | Bacterial spot; healthy | 2 |
| Potato | Early blight; late blight; healthy | 3 |
| Raspberry | Healthy | 1 |
| Soybean | Healthy | 1 |
| Squash | Powdery mildew | 1 |
| Strawberry | Leaf scorch; healthy | 2 |
| Tomato | Bacterial spot; early blight; late blight; leaf mold; Septoria leaf spot; two-spotted spider mite damage; target spot; yellow leaf curl virus; mosaic virus; healthy | 10 |
| **Total** | | **38** |

The machine-readable label contract is `configs/supported_classes.json`. Class IDs are zero-based positions in its `class_names` array, matching the upstream builder's declared order. Preserve original dataset spelling in raw labels; use readable spelling in the interface. In Step 3, verify these names against the downloaded data and explicitly map source labels to this contract. Do not assume another loader uses the same integer IDs.

## Architecture decisions

- Dataset: original PlantVillage color images. Audit actual counts, metadata, revision, and usage terms when downloading in Step 3.
- Main model: ImageNet-pretrained ResNet18 with a new classifier producing 38 logits.
- Comparison model: a small CNN trained from scratch on identical splits.
- Training: PyTorch and Torchvision, first training the new classifier, then fine-tuning the final residual stage.
- Interface: Streamlit, calling a shared prediction module that also supports command-line inference.
- Packaging: save model weights, class mapping, preprocessing settings, and model version together.

## Data and evaluation commitments

1. Keep photographs of the same physical leaf in one split, and check duplicate overlap before training.
2. Preserve the upstream grouped test split where available. Create validation from the training portion by leaf group. Verify the supplied metadata rather than assuming grouping is complete.
3. Augment training images only. Use identical deterministic preprocessing for validation, test, and uploaded images.
4. Select model settings, checkpoints, and any uncertainty threshold using validation data. Keep test data reserved for final evaluation.
5. Report accuracy, macro F1, per-class precision and recall, class support, a confusion matrix, and prediction latency.
6. Report performance on a separate labeled set of real photographs when available. PlantVillage results alone do not establish field-photo performance.

## Completion criteria

The final project is complete when the data audit and split checks pass; the baseline and ResNet18 have documented results; the selected model can be saved and reloaded with consistent predictions; and a fresh environment can run the documented upload demo. The application must handle unreadable uploads clearly. The report must distinguish measured results from examples and state the coverage of external-photo evaluation.

The scope, class contract, and build sequence are implemented. Full-data training, held-out evaluation, packaged-model reload, upload handling, and app prediction checks have passed. Independent field-photo accuracy remains unmeasured.

## Sources

- [Upstream dataset repository](https://github.com/spMohanty/PlantVillage-Dataset)
- [Upstream class definitions](https://github.com/spMohanty/PlantVillage-Dataset/blob/master/plant_village.py)
- [Dataset card and leaf-group splitting guidance](https://huggingface.co/datasets/mohanty/PlantVillage)
- [Original study and external-image evaluation](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2016.01419/full)
