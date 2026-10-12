# Third-party notices

## PlantVillage data and example photograph

The training data, crop-condition labels, split metadata, and example photograph originate from the PlantVillage dataset by Sharada P. Mohanty, David P. Hughes, and Marcel Salathé and the contributors credited by the dataset.

- Original repository: [spMohanty/PlantVillage-Dataset](https://github.com/spMohanty/PlantVillage-Dataset)
- Original study: [Using Deep Learning for Image-Based Plant Disease Detection](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2016.01419/full)
- Dataset distribution used: [mohanty/PlantVillage on Hugging Face](https://huggingface.co/datasets/mohanty/PlantVillage/tree/9e97599868962bd0079b8db4b7f1efa9185fa1e7)
- Distribution revision: `9e97599868962bd0079b8db4b7f1efa9185fa1e7`
- Recorded upstream revision: `7f7ecc7e1eaca78107e3affe7cb5abd9427e139a`
- License declared by that distribution's dataset card: [Creative Commons Attribution-ShareAlike 3.0 Unported (CC BY-SA 3.0)](https://creativecommons.org/licenses/by-sa/3.0/)

`examples/apple_scab.jpg` is an unchanged copy of the PlantVillage color photograph `Apple___Apple_scab/0261a6e4-21f8-481a-8827-b674e6955644___FREC_Scab 3055.JPG`. The same photograph appears in the app screenshot `reports/demo.jpg`; the screenshot adds the classifier interface and predictions. The example photograph and its depiction retain the dataset attribution and declared license. The complete training image archive is downloaded separately rather than included in this repository.

Dataset provenance, archive size, pinned revisions, and the declared dataset license are recorded in `configs/dataset.json`. Generated split manifests and evaluation reports document how this project used the data.

## Model and software dependencies

The classifier uses ResNet18 initialized from the official Torchvision ImageNet weights and fine-tuned on the documented PlantVillage training split. PyTorch, Torchvision, Streamlit, and the other dependencies retain their own licenses. See the upstream projects and the installed packages for their license terms.

This notice records third-party attribution. It does not assign a new license to this repository's source code or establish license terms for the trained model weights.
