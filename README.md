## Fine-tune And host SDXL models cost-effectively with AWS Inferentia2

## Description

Fine Tune SDXL
The SDXL 1.0 is a text-to-image generation model developed by Stability AI, consisting of over 3 billion parameters. It comprises several key components, including a text encoder that converts input prompts into latent representations, and a U-Net model that generates images based on these latent representations through a diffusion process. Despite its impressive capabilities trained on public dataset,  AWS customers sometimes need to generate images for specific subject or style that are difficult or inefficient to describe in words. In that situation, fine tuning is a great option to improve the performance using customer’s own data.

One of the popular approach to fine tune SDXL is to leverage Dreambooth and LoRA (Low-Rank Adaptation) techniques. Dreambooth allows us to personalize the model by embedding a subject into its output domain using a unique identifier, effectively expanding its language-vision dictionary. This process leverages a technique called prior preservation, which retains the model's existing knowledge about the subject class (e.g., humans) while incorporating new information from the provided subject images. Concurrently, we utilized LoRA, an efficient fine-tuning method that attaches small adapter networks to specific layers of the pre-trained model, freezing most of its weights. By combining these techniques, we could generate a personalized model while tuning an order-of-magnitude fewer parameters, resulting in faster training times, reduced GPU utilization, and optimized storage requirements.

Host and invoke fine-tuned Model on Inf2
AWS Inferentia2 is purpose-built machine learning (ML) accelerator designed for inference workloads and delivers high-performance at up to 40% lower cost for generative AI workloads over other inference optimized instances on AWS. To leverage the full potential of Inf2 instances, you need AWS Neuron SDK. This is a software layer running atop the Inf2 hardware, enables end-to-end ML development lifecycle, from building new models to training, optimizing, and deploying them for production. The transformers-neuronx component, part of the AWS Neuron SDK, is specifically tailored for transformer decoder inference workflows, supporting a wide range of popular models, including the SDXL model. By compiling and hosting SDXL on Inf2 instances using transformers-neuronx, organizations can benefit from the exceptional performance and cost-efficiency offered by these specialized accelerators, while taking advantage of the seamless integration with popular deep learning frameworks like TensorFlow and PyTorch.



## Steps to follow

#### Fine-tune sdxl

1. Upload 10-12 selfies in `fine-tune-sdxl/data/` directory. 

2. Run the notebook `sdxl_fine_tune.ipynb` to fine tune SDXL model with your images. 


#### Compile on Inferentia

1. On an EC2 inferentia instance, install and run jupiter notebook.
2. Execute the scripts  `compile-on-inferentia/install-drivers.sh` and `compile-on-inferentia/install-pytorch-neuron.sh`
3. Copy the model weights generated in `fine-tune-sdxl/finetune_sttirum` directory to `compile-on-inferentia/lora` directory
4. Run the notebook `compile-on-inferentia/hf_pretrained_sdxl_base_1024_inference.ipynb` to compile the SDXL model on inferentia2 


## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.
