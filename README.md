# byol-embeddings

A repository for training an unsupervised BYOL model on ImageStream image data, and running basic visualisations of the resultant embeddings. Differs from the original BYOL paper due to slight alterations of transformations used (primarily removal of colour jitter). Uses WideResNet50 as the convolutional neural network backbone.

TODO:
- Implement large batch training optimiser.
- Ensemble representation generation.
